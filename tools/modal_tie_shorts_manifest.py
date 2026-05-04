#!/usr/bin/env python3
"""Build a bounded TIE_shorts manifest from HF metadata without full dataset materialization."""

from __future__ import annotations

import csv
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
HF_SECRET_NAME = os.environ.get("LOCALWISPR_MODAL_LORA_HF_SECRET_NAME", "huggingface-secret")
ARTIFACTS_DIR = Path("/artifacts")
DATASET_ID = "raianand/TIE_shorts"

app = modal.App("localwispr-tie-shorts-manifest")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub", "requests")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)


@dataclass(frozen=True)
class TieManifestConfig:
    manifest_name: str
    sample_limit: int = 10_000
    output_limit: int = 5_000
    seed: int = 42
    min_duration_seconds: float = 1.5
    max_duration_seconds: float = 10.0
    min_word_count: int = 4
    max_word_count: int = 24
    max_samples_per_speaker: int = 8
    balance_fields: tuple[str, ...] = ("Native Region", "Gender", "Discipline Group")
    export_audio: bool = True
    max_workers: int = 32
    download_read_timeout_seconds: float = 90.0
    download_retries: int = 3
    dry_run: bool = False


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _sanitize_artifact_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-._") or "artifact"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _text_key(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s']", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _word_count(text: str) -> int:
    return len(_text_key(text).split())


def _contains_date_like(text: str) -> bool:
    return bool(re.search(r"\b\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?\b", text))


def _contains_currency_or_amount(text: str) -> bool:
    return bool(re.search(r"[$₹€£]|\b(rs|rupees|dollars|usd|inr)\b", text, flags=re.IGNORECASE))


def _duration(row: dict[str, str]) -> float | None:
    for key in ("Speech_Duration_seconds", "duration_seconds", "duration"):
        value = row.get(key)
        if value is None or not str(value).strip():
            continue
        try:
            return float(value)
        except ValueError:
            continue
    for key, value in row.items():
        if "duration" not in key.lower():
            continue
        if value is None or not str(value).strip():
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _clean_csv_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(key or "").strip().lstrip("\ufeff"): str(value or "").strip()
        for key, value in row.items()
    }


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _row_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    normalized_targets = {_normalized_key(key) for key in keys}
    for key, value in row.items():
        if _normalized_key(key) in normalized_targets and value:
            return str(value).strip()
    return ""


def _transcript(row: dict[str, str]) -> str:
    value = _row_value(
        row,
        "Normalised_Transcript",
        "Normalised Transcript",
        "Normalized_Transcript",
        "Normalized Transcript",
        "Transcript",
        "transcript",
        "text",
    )
    if value:
        return value
    for key, value in row.items():
        normalized_key = key.lower().replace(" ", "_")
        if "transcript" in normalized_key and value:
            return str(value).strip()
    return ""


def _speaker_key(row: dict[str, str]) -> str:
    value = _row_value(row, "Speaker_ID", "Speaker ID", "speaker_id", "speaker")
    return value or "unknown"


def _balance_key(row: dict[str, str], fields: tuple[str, ...]) -> str:
    parts = []
    for field in fields:
        value = _row_value(row, field)
        if value:
            parts.append(f"{field}={value}")
    return "|".join(parts) if parts else "all"


def _score_row(row: dict[str, str], *, config: TieManifestConfig) -> tuple[bool, list[str], dict[str, Any]]:
    text = _transcript(row)
    duration = _duration(row)
    word_count = _word_count(text)
    reject_reasons: list[str] = []
    if not text:
        reject_reasons.append("empty_text")
    if word_count < config.min_word_count:
        reject_reasons.append("too_few_words")
    if word_count > config.max_word_count:
        reject_reasons.append("too_many_words")
    if duration is None:
        reject_reasons.append("missing_duration")
    else:
        if duration < config.min_duration_seconds:
            reject_reasons.append("too_short")
        if duration > config.max_duration_seconds:
            reject_reasons.append("too_long")
    if any(character.isdigit() for character in text):
        reject_reasons.append("format_sensitive")
    if _contains_date_like(text) or _contains_currency_or_amount(text):
        reject_reasons.append("format_sensitive")

    score = 100.0
    if duration is not None:
        if 3.0 <= duration <= 7.0:
            score += 10.0
        elif duration > 8.0:
            score -= 5.0
    if 8 <= word_count <= 18:
        score += 8.0
    return not reject_reasons, reject_reasons, {
        "duration_seconds": duration,
        "word_count": word_count,
        "score": score,
    }


def _audio_repo_path(row: dict[str, str], repo_files: set[str]) -> str | None:
    raw_id = str(row.get("ID") or "").strip()
    if not raw_id:
        return None
    candidates = []
    if raw_id.startswith("audio/"):
        candidates.append(raw_id)
    candidates.extend(
        [
            f"audio/{raw_id}",
            f"audio/{raw_id}.mp3",
            f"audio/{raw_id}.mp3.mp3",
        ]
    )
    for candidate in candidates:
        if candidate in repo_files:
            return candidate
    return None


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    config: TieManifestConfig,
) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    candidates = list(rows)
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: (-float(item["score_payload"]["score"]), item["tie_breaker"]))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(str(row["balance_key"]), []).append(row)
    active_keys = sorted(grouped, key=lambda key: (-float(grouped[key][0]["score_payload"]["score"]), key))
    selected: list[dict[str, Any]] = []
    speaker_counts: dict[str, int] = {}
    seen_texts: set[str] = set()
    while active_keys and len(selected) < config.output_limit:
        next_keys = []
        for key in active_keys:
            bucket = grouped[key]
            while bucket:
                candidate = bucket.pop(0)
                text_key = candidate["text_key"]
                speaker = str(candidate["speaker_key"])
                if text_key in seen_texts:
                    continue
                if config.max_samples_per_speaker > 0 and speaker_counts.get(speaker, 0) >= config.max_samples_per_speaker:
                    continue
                selected.append(candidate)
                seen_texts.add(text_key)
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                break
            if bucket:
                next_keys.append(key)
            if len(selected) >= config.output_limit:
                break
        active_keys = next_keys
    return selected


def _summarize_numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": ordered[round((len(ordered) - 1) * 0.50)],
        "p90": ordered[round((len(ordered) - 1) * 0.90)],
        "max": ordered[-1],
    }


@app.function(
    image=image,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def build_tie_manifest_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    import requests
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

    balance_fields = tuple(config_payload.get("balance_fields") or ("Native Region", "Gender", "Discipline Group"))
    config = TieManifestConfig(
        manifest_name=str(config_payload["manifest_name"]),
        sample_limit=int(config_payload.get("sample_limit", 10_000)),
        output_limit=int(config_payload.get("output_limit", 5_000)),
        seed=int(config_payload.get("seed", 42)),
        min_duration_seconds=float(config_payload.get("min_duration_seconds", 1.5)),
        max_duration_seconds=float(config_payload.get("max_duration_seconds", 10.0)),
        min_word_count=int(config_payload.get("min_word_count", 4)),
        max_word_count=int(config_payload.get("max_word_count", 24)),
        max_samples_per_speaker=int(config_payload.get("max_samples_per_speaker", 8)),
        balance_fields=balance_fields,
        export_audio=bool(config_payload.get("export_audio", True)),
        max_workers=int(config_payload.get("max_workers", 32)),
        download_read_timeout_seconds=float(config_payload.get("download_read_timeout_seconds", 90.0)),
        download_retries=int(config_payload.get("download_retries", 3)),
        dry_run=bool(config_payload.get("dry_run", False)),
    )
    token = os.environ.get("HF_TOKEN")
    run_id = f"{config.manifest_name}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    audio_dir = run_dir / "audio"
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.export_audio and not config.dry_run:
        audio_dir.mkdir(parents=True, exist_ok=True)

    def write_progress(payload: dict[str, Any], *, commit: bool = True) -> None:
        payload = {"stage": "tie_shorts_manifest", "run_id": run_id, "updated_at_utc": _now_iso(), **payload}
        _write_json(run_dir / "progress.json", payload)
        if commit:
            artifacts_volume.commit()

    write_progress({"status": "loading_metadata", "rows_seen": 0, "selected_rows": 0})
    metadata_path = hf_hub_download(DATASET_ID, "Metadata.csv", repo_type="dataset", token=token)
    repo_files = set(HfApi(token=token).list_repo_files(DATASET_ID, repo_type="dataset"))

    rng = random.Random(config.seed)
    candidates: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    rows_seen = 0
    missing_audio = 0
    with open(metadata_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        metadata_columns = [str(name or "").strip().lstrip("\ufeff") for name in (reader.fieldnames or [])]
        duration_values_all: list[float] = []
        word_counts_all: list[float] = []
        sample_metadata: list[dict[str, Any]] = []
        for source_index, row in enumerate(reader):
            row = _clean_csv_row(row)
            rows_seen += 1
            if config.sample_limit > 0 and rows_seen > config.sample_limit:
                break
            duration = _duration(row)
            text = _transcript(row)
            if duration is not None:
                duration_values_all.append(duration)
            if text:
                word_counts_all.append(float(_word_count(text)))
            if len(sample_metadata) < 5:
                sample_metadata.append(
                    {
                        "source_index": source_index,
                        "keys": sorted(row.keys()),
                        "id": row.get("ID"),
                        "duration": duration,
                        "word_count": _word_count(text),
                        "speaker": _row_value(row, "Speaker_ID", "Speaker ID"),
                        "native_region": _row_value(row, "Native_Region", "Native Region"),
                        "discipline_group": _row_value(row, "Discipline_Group", "Discipline Group"),
                    }
                )
            audio_repo_path = _audio_repo_path(row, repo_files)
            if audio_repo_path is None:
                missing_audio += 1
                continue
            keep, reject_reasons, score_payload = _score_row(row, config=config)
            for reason in reject_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            if not keep:
                continue
            text = _transcript(row)
            candidates.append(
                {
                    "source_index": source_index,
                    "row": dict(row),
                    "audio_repo_path": audio_repo_path,
                    "text": text,
                    "text_key": _text_key(text),
                    "speaker_key": _speaker_key(row),
                    "balance_key": _balance_key(row, config.balance_fields),
                    "score_payload": score_payload,
                    "tie_breaker": rng.random(),
                }
            )
            if rows_seen % 2000 == 0:
                write_progress(
                    {
                        "status": "scoring",
                        "rows_seen": rows_seen,
                        "candidate_rows": len(candidates),
                        "selected_rows": 0,
                    }
                )

    selected = _select_rows(candidates, config=config)
    write_progress(
        {
            "status": "selected",
            "rows_seen": rows_seen,
            "candidate_rows": len(candidates),
            "selected_rows": len(selected),
            "exported_rows": 0,
        }
    )

    download_errors: list[dict[str, Any]] = []

    def download_one(index_and_row: tuple[int, dict[str, Any]]) -> tuple[int, str]:
        selected_index, item = index_and_row
        suffix = Path(item["audio_repo_path"]).suffix or ".mp3"
        relative = f"audio/{selected_index:06d}{suffix}"
        destination = run_dir / relative
        temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
        url = hf_hub_url(DATASET_ID, item["audio_repo_path"], repo_type="dataset")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        last_error: Exception | None = None
        for attempt in range(1, max(1, config.download_retries) + 1):
            try:
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(15.0, config.download_read_timeout_seconds),
                ) as response:
                    response.raise_for_status()
                    bytes_written = 0
                    with temporary_destination.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            output.write(chunk)
                            bytes_written += len(chunk)
                if bytes_written <= 0:
                    raise RuntimeError("download produced an empty file")
                temporary_destination.replace(destination)
                return selected_index, relative
            except Exception as exc:  # noqa: BLE001 - error is reported in the manifest report.
                last_error = exc
                try:
                    temporary_destination.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt >= max(1, config.download_retries):
                    break
        raise RuntimeError(
            f"failed to download {item['audio_repo_path']} after {config.download_retries} attempts: {last_error!r}"
        )

    audio_relatives: dict[int, str] = {}
    if config.export_audio and not config.dry_run:
        with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as executor:
            futures = {
                executor.submit(download_one, (index, item)): index
                for index, item in enumerate(selected, start=1)
            }
            completed = 0
            for future in as_completed(futures):
                selected_index = futures[future]
                completed += 1
                try:
                    index, relative = future.result()
                    audio_relatives[index] = relative
                except Exception as exc:
                    download_errors.append({"manifest_index": selected_index, "error": repr(exc)})
                if completed == 1 or completed % 100 == 0 or completed == len(futures):
                    print(f"[tie:{run_id}] downloaded {completed}/{len(futures)}")
                    write_progress(
                        {
                            "status": "downloading_audio",
                            "rows_seen": rows_seen,
                            "candidate_rows": len(candidates),
                            "selected_rows": len(selected),
                            "exported_rows": completed,
                            "download_errors": len(download_errors),
                        }
                    )

    manifest_rows = []
    duration_values = []
    word_counts = []
    for selected_index, item in enumerate(selected, start=1):
        if config.export_audio and not config.dry_run and selected_index not in audio_relatives:
            continue
        audio_relative = audio_relatives.get(selected_index, "")
        score_payload = item["score_payload"]
        duration = score_payload["duration_seconds"]
        if duration is not None:
            duration_values.append(float(duration))
        word_counts.append(int(score_payload["word_count"]))
        source_row = item["row"]
        manifest_row = {
            "manifest_index": len(manifest_rows) + 1,
            "dataset_index": item["source_index"],
            "audio": f"{run_id}/{audio_relative}" if audio_relative else "",
            "text": item["text"],
            "raw_transcript": source_row.get("Transcript"),
            "source_dataset": DATASET_ID,
            "source_split": "train",
            "speaker_key": item["speaker_key"],
            "balance_key": item["balance_key"],
            "audio_repo_path": item["audio_repo_path"],
            "audio_file": audio_relative,
            "duration_seconds": duration,
            "word_count": score_payload["word_count"],
            "quality_score": score_payload["score"],
        }
        for field_name in (
            "ID",
            "Speaker_ID",
            "Speaker ID",
            "Gender",
            "Caste",
            "Year_Class",
            "Speech_Class",
            "Speech Class",
            "Discipline_Group",
            "Discipline Group",
            "Native_Region",
            "Native Region",
            "Topic",
        ):
            manifest_row[field_name] = _row_value(source_row, field_name)
        manifest_rows.append(manifest_row)

    manifest_jsonl_path = run_dir / "manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    train_jsonl_path = run_dir / "train.jsonl"
    with train_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(
                json.dumps(
                    {
                        "audio": row["audio"],
                        "text": row["text"],
                        "source_dataset": row["source_dataset"],
                        "dataset_index": row["dataset_index"],
                        "quality_score": row["quality_score"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")

    report = {
        "manifest_run_id": run_id,
        "created_at_utc": _now_iso(),
        "dataset": {
            "name": DATASET_ID,
            "metadata_columns": metadata_columns,
            "metadata_rows_seen": rows_seen,
            "candidate_rows": len(candidates),
            "selected_rows": len(selected),
            "exported_rows": len(manifest_rows),
            "missing_audio_rows": missing_audio,
            "sample_metadata": sample_metadata,
        },
        "selection": {
            "sample_limit": config.sample_limit,
            "output_limit": config.output_limit,
            "balance_fields": list(config.balance_fields),
            "max_samples_per_speaker": config.max_samples_per_speaker,
            "download_read_timeout_seconds": config.download_read_timeout_seconds,
            "download_retries": config.download_retries,
            "rejection_counts": dict(sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))),
            "duration_seconds": _summarize_numeric(duration_values),
            "word_count": _summarize_numeric([float(value) for value in word_counts]),
            "download_errors": download_errors[:20],
            "download_error_count": len(download_errors),
            "all_duration_seconds": _summarize_numeric(duration_values_all),
            "all_word_count": _summarize_numeric(word_counts_all),
        },
        "artifacts": {
            "manifest_dir": str(run_dir),
            "manifest_jsonl": str(manifest_jsonl_path),
            "training_jsonl": str(train_jsonl_path),
            "audio_dir": str(audio_dir) if config.export_audio else None,
            "progress_path": str(run_dir / "progress.json"),
        },
    }
    _write_json(run_dir / "report.json", report)
    write_progress(
        {
            "status": "complete",
            "rows_seen": rows_seen,
            "candidate_rows": len(candidates),
            "selected_rows": len(selected),
            "exported_rows": len(manifest_rows),
            "report_path": str(run_dir / "report.json"),
        }
    )
    artifacts_volume.commit()
    return report


@app.local_entrypoint()
def main(
    manifest_name: str = "tie-shorts-metadata-balanced",
    sample_limit: int = 10_000,
    output_limit: int = 5_000,
    seed: int = 42,
    min_duration_seconds: float = 1.5,
    max_duration_seconds: float = 10.0,
    min_word_count: int = 4,
    max_word_count: int = 24,
    max_samples_per_speaker: int = 8,
    balance_fields: str = "Native Region,Gender,Discipline Group",
    export_audio: bool = True,
    max_workers: int = 32,
    download_read_timeout_seconds: float = 90.0,
    download_retries: int = 3,
    dry_run: bool = False,
) -> None:
    fields = tuple(field.strip() for field in balance_fields.split(",") if field.strip())
    report = build_tie_manifest_remote.remote(
        {
            "manifest_name": manifest_name,
            "sample_limit": sample_limit,
            "output_limit": output_limit,
            "seed": seed,
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "min_word_count": min_word_count,
            "max_word_count": max_word_count,
            "max_samples_per_speaker": max_samples_per_speaker,
            "balance_fields": fields,
            "export_audio": export_audio,
            "max_workers": max_workers,
            "download_read_timeout_seconds": download_read_timeout_seconds,
            "download_retries": download_retries,
            "dry_run": dry_run,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
