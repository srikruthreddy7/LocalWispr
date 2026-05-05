#!/usr/bin/env python3
"""Build Modal-side mixed training manifests from existing artifact manifests."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App("localwispr-mix-training-manifests")
image = modal.Image.debian_slim(python_version="3.11")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)


@dataclass(frozen=True)
class SourceSpec:
    path: str
    limit: int
    label: str


@dataclass(frozen=True)
class MixConfig:
    mix_name: str
    sources: list[SourceSpec]
    seed: int = 42
    dedupe_text: bool = True
    max_samples_per_speaker: int = 0
    selection_strategy: str = "random"
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    min_word_count: int | None = None
    max_word_count: int | None = None
    reject_format_sensitive: bool = False
    min_selection_score: float | None = None
    min_base_cer: float | None = None
    max_base_cer: float | None = None
    max_teacher_cer: float | None = None


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return key, value
    return None, None


def _canonicalize_training_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize mixed sources to the local training loader's audio/text columns."""
    audio_key, audio_value = _first_present(
        row,
        ("audio", "audio_path", "audio_filepath", "path", "file"),
    )
    text_key, text_value = _first_present(
        row,
        ("text", "normalized_transcript", "transcript", "sentence", "raw_transcript"),
    )
    mixed_row = dict(row)
    if audio_key is not None:
        mixed_row["audio"] = audio_value
        mixed_row["source_audio_column"] = audio_key
    if text_key is not None:
        mixed_row["text"] = str(text_value)
        mixed_row["source_text_column"] = text_key
    return mixed_row


def _text_key(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("normalized_transcript") or row.get("transcript") or "")


def _word_count(row: dict[str, Any]) -> int:
    return len(_as_text(row).split())


def _duration_seconds(row: dict[str, Any]) -> float | None:
    for key in ("duration_seconds", "audio_seconds", "duration", "duration_sec"):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _metric_value(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _row_selection_score(row: dict[str, Any]) -> float:
    explicit = _metric_value(row, ("selection_score", "score", "hard_score"))
    if explicit is not None:
        return explicit
    base_cer = _metric_value(row, ("base_cer", "turbo_cer", "whisper_turbo_cer"))
    teacher_cer = _metric_value(row, ("teacher_cer", "large_v3_cer", "whisper_large_v3_cer"))
    if base_cer is not None and teacher_cer is not None:
        return base_cer - teacher_cer
    if base_cer is not None:
        return base_cer
    return 0.0


def _is_format_sensitive(row: dict[str, Any]) -> bool:
    text = _as_text(row)
    lowered = text.lower()
    if re.search(r"\d", text):
        return True
    if re.search(r"[$₹€£]|\b(?:rs|inr|usd|dollars?|rupees?)\b", lowered):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", lowered):
        return True
    if re.search(r"<[^>]+>|&[a-z]+;", lowered):
        return True
    return False


def _passes_quality_filters(row: dict[str, Any], config: MixConfig) -> tuple[bool, str | None]:
    duration = _duration_seconds(row)
    if config.min_duration_seconds is not None and (
        duration is None or duration < config.min_duration_seconds
    ):
        return False, "min_duration_seconds"
    if config.max_duration_seconds is not None and (
        duration is None or duration > config.max_duration_seconds
    ):
        return False, "max_duration_seconds"

    words = _word_count(row)
    if config.min_word_count is not None and words < config.min_word_count:
        return False, "min_word_count"
    if config.max_word_count is not None and words > config.max_word_count:
        return False, "max_word_count"

    if config.reject_format_sensitive and _is_format_sensitive(row):
        return False, "format_sensitive"

    score = _row_selection_score(row)
    if config.min_selection_score is not None and score < config.min_selection_score:
        return False, "min_selection_score"

    base_cer = _metric_value(row, ("base_cer", "turbo_cer", "whisper_turbo_cer"))
    if config.min_base_cer is not None and (base_cer is None or base_cer < config.min_base_cer):
        return False, "min_base_cer"
    if config.max_base_cer is not None and (base_cer is None or base_cer > config.max_base_cer):
        return False, "max_base_cer"

    teacher_cer = _metric_value(row, ("teacher_cer", "large_v3_cer", "whisper_large_v3_cer"))
    if config.max_teacher_cer is not None and (
        teacher_cer is None or teacher_cer > config.max_teacher_cer
    ):
        return False, "max_teacher_cer"

    return True, None


def _select_source_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    rng: random.Random,
    selection_strategy: str,
) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    if selection_strategy == "random":
        return rng.sample(rows, limit)
    if selection_strategy == "score":
        return sorted(
            rows,
            key=lambda row: (
                _row_selection_score(row),
                _duration_seconds(row) or 0.0,
                _word_count(row),
            ),
            reverse=True,
        )[:limit]
    raise ValueError("selection_strategy must be one of: random, score")


def _speaker_key(row: dict[str, Any]) -> str:
    for key in (
        "speaker_key",
        "client_id",
        "speaker_id",
        "Speaker_ID",
        "speaker",
        "user_id",
        "utt_spk",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{str(value).strip()}"
    return ""


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = []
    word_counts = []
    labels: dict[str, int] = {}
    scores = []
    for row in rows:
        label = str(row.get("mix_source") or "")
        labels[label] = labels.get(label, 0) + 1
        scores.append(_row_selection_score(row))
        try:
            durations.append(float(row["duration_seconds"]))
        except (KeyError, TypeError, ValueError):
            pass
        text = str(row.get("text") or "")
        if text:
            word_counts.append(len(text.split()))

    def numeric(values: list[float]) -> dict[str, float | int | None]:
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

    return {
        "rows": len(rows),
        "rows_by_source": dict(sorted(labels.items())),
        "duration_seconds": numeric(durations),
        "word_count": numeric([float(value) for value in word_counts]),
        "selection_score": numeric(scores),
    }


@app.function(
    image=image,
    timeout=60 * 20,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def mix_manifests_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = MixConfig(
        mix_name=str(config_payload["mix_name"]),
        sources=[SourceSpec(**item) for item in config_payload["sources"]],
        seed=int(config_payload.get("seed", 42)),
        dedupe_text=bool(config_payload.get("dedupe_text", True)),
        max_samples_per_speaker=int(config_payload.get("max_samples_per_speaker", 0)),
        selection_strategy=str(config_payload.get("selection_strategy") or "random"),
        min_duration_seconds=_as_float(config_payload.get("min_duration_seconds")),
        max_duration_seconds=_as_float(config_payload.get("max_duration_seconds")),
        min_word_count=(
            int(config_payload["min_word_count"])
            if config_payload.get("min_word_count") is not None
            else None
        ),
        max_word_count=(
            int(config_payload["max_word_count"])
            if config_payload.get("max_word_count") is not None
            else None
        ),
        reject_format_sensitive=bool(config_payload.get("reject_format_sensitive", False)),
        min_selection_score=_as_float(config_payload.get("min_selection_score")),
        min_base_cer=_as_float(config_payload.get("min_base_cer")),
        max_base_cer=_as_float(config_payload.get("max_base_cer")),
        max_teacher_cer=_as_float(config_payload.get("max_teacher_cer")),
    )
    rng = random.Random(config.seed)
    run_id = f"{config.mix_name}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    source_reports = []
    for source_index, source in enumerate(config.sources):
        source_path = Path(source.path)
        if not source_path.is_absolute():
            source_path = ARTIFACTS_DIR / source.path.lstrip("/")
        rows = [_canonicalize_training_row(row) for row in _load_jsonl(source_path)]
        filter_counts: dict[str, int] = {}
        filtered_rows = []
        for row in rows:
            passes, reason = _passes_quality_filters(row, config)
            if not passes:
                filter_counts[str(reason)] = filter_counts.get(str(reason), 0) + 1
                continue
            filtered_rows.append(row)
        selected = _select_source_rows(
            filtered_rows,
            limit=source.limit,
            rng=rng,
            selection_strategy=config.selection_strategy,
        )
        for row_index, row in enumerate(selected):
            mixed_row = dict(row)
            mixed_row["mix_source"] = source.label
            mixed_row["mix_source_manifest"] = str(source_path)
            mixed_row["mix_source_index"] = source_index
            mixed_row["mix_source_row_index"] = row_index
            mixed_row["mix_selection_score"] = _row_selection_score(row)
            selected_rows.append(mixed_row)
        source_reports.append(
            {
                "label": source.label,
                "path": str(source_path),
                "available_rows": len(rows),
                "rows_after_quality_filters": len(filtered_rows),
                "selected_rows": len(selected),
                "limit": source.limit,
                "filter_counts": filter_counts,
            }
        )

    rng.shuffle(selected_rows)
    if config.dedupe_text or config.max_samples_per_speaker > 0:
        filtered_rows: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        speaker_counts: dict[str, int] = {}
        filter_counts = {
            "duplicate_text": 0,
            "speaker_cap": 0,
        }
        for row in selected_rows:
            if config.dedupe_text:
                text_key = _text_key(row)
                if text_key and text_key in seen_texts:
                    filter_counts["duplicate_text"] += 1
                    continue
                if text_key:
                    seen_texts.add(text_key)
            if config.max_samples_per_speaker > 0:
                speaker_key = _speaker_key(row)
                if speaker_key:
                    speaker_count = speaker_counts.get(speaker_key, 0)
                    if speaker_count >= config.max_samples_per_speaker:
                        filter_counts["speaker_cap"] += 1
                        continue
                    speaker_counts[speaker_key] = speaker_count + 1
            filtered_rows.append(row)
        selected_rows = filtered_rows
    else:
        filter_counts = {}
    train_path = run_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    report = {
        "mix_run_id": run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "seed": config.seed,
        "sources": source_reports,
        "filters": {
            "dedupe_text": config.dedupe_text,
            "max_samples_per_speaker": config.max_samples_per_speaker,
            "selection_strategy": config.selection_strategy,
            "min_duration_seconds": config.min_duration_seconds,
            "max_duration_seconds": config.max_duration_seconds,
            "min_word_count": config.min_word_count,
            "max_word_count": config.max_word_count,
            "reject_format_sensitive": config.reject_format_sensitive,
            "min_selection_score": config.min_selection_score,
            "min_base_cer": config.min_base_cer,
            "max_base_cer": config.max_base_cer,
            "max_teacher_cer": config.max_teacher_cer,
            "filter_counts": filter_counts,
        },
        "summary": _summarize(selected_rows),
        "artifacts": {
            "mix_dir": str(run_dir),
            "training_jsonl": str(train_path),
            "report": str(run_dir / "report.json"),
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return report


def _parse_sources(raw_sources: str) -> list[dict[str, Any]]:
    sources = []
    for raw_source in raw_sources.split(","):
        raw_source = raw_source.strip()
        if not raw_source:
            continue
        parts = raw_source.split(":")
        if len(parts) != 3:
            raise ValueError("Each source must be path:limit:label")
        path, limit, label = parts
        sources.append({"path": path, "limit": int(limit), "label": label})
    if not sources:
        raise ValueError("At least one source is required")
    return sources


@app.local_entrypoint()
def main(
    mix_name: str,
    sources: str,
    seed: int = 42,
    dedupe_text: bool = True,
    max_samples_per_speaker: int = 0,
    selection_strategy: str = "random",
    min_duration_seconds: float | None = None,
    max_duration_seconds: float | None = None,
    min_word_count: int | None = None,
    max_word_count: int | None = None,
    reject_format_sensitive: bool = False,
    min_selection_score: float | None = None,
    min_base_cer: float | None = None,
    max_base_cer: float | None = None,
    max_teacher_cer: float | None = None,
) -> None:
    report = mix_manifests_remote.remote(
        {
            "mix_name": mix_name,
            "sources": _parse_sources(sources),
            "seed": seed,
            "dedupe_text": dedupe_text,
            "max_samples_per_speaker": max_samples_per_speaker,
            "selection_strategy": selection_strategy,
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "min_word_count": min_word_count,
            "max_word_count": max_word_count,
            "reject_format_sensitive": reject_format_sensitive,
            "min_selection_score": min_selection_score,
            "min_base_cer": min_base_cer,
            "max_base_cer": max_base_cer,
            "max_teacher_cer": max_teacher_cer,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
