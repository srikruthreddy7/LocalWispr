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
    for row in rows:
        label = str(row.get("mix_source") or "")
        labels[label] = labels.get(label, 0) + 1
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
        rows = _load_jsonl(source_path)
        if source.limit <= 0:
            selected = rows
        elif len(rows) <= source.limit:
            selected = rows
        else:
            selected = rng.sample(rows, source.limit)
        for row_index, row in enumerate(selected):
            mixed_row = _canonicalize_training_row(row)
            mixed_row["mix_source"] = source.label
            mixed_row["mix_source_manifest"] = str(source_path)
            mixed_row["mix_source_index"] = source_index
            mixed_row["mix_source_row_index"] = row_index
            selected_rows.append(mixed_row)
        source_reports.append(
            {
                "label": source.label,
                "path": str(source_path),
                "available_rows": len(rows),
                "selected_rows": len(selected),
                "limit": source.limit,
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
) -> None:
    report = mix_manifests_remote.remote(
        {
            "mix_name": mix_name,
            "sources": _parse_sources(sources),
            "seed": seed,
            "dedupe_text": dedupe_text,
            "max_samples_per_speaker": max_samples_per_speaker,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
