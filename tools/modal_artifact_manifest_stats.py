#!/usr/bin/env python3
"""Modal-side aggregate stats for artifact manifests.

This keeps gated dataset manifests inside the Modal artifacts volume and only
prints aggregate counts locally.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App("localwispr-artifact-manifest-stats")
image = modal.Image.debian_slim(python_version="3.11")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)


@dataclass
class ManifestStatsConfig:
    run_id: str
    manifest_name: str = "manifest.jsonl"
    text_field: str = "normalized_transcript"
    top_k: int = 12


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(q: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
        return ordered[index]

    return {
        "count": len(values),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mean": mean(values),
    }


def _split_values(value: Any) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


@app.function(
    image=image,
    timeout=60 * 5,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def manifest_stats_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = ManifestStatsConfig(**config_payload)
    manifest_path = ARTIFACTS_DIR / config.run_id / config.manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = 0
    fields: set[str] = set()
    counters: dict[str, Counter[str]] = {
        "gender": Counter(),
        "language": Counter(),
        "state": Counter(),
        "district": Counter(),
        "contains_digit": Counter(),
        "contains_date_like": Counter(),
        "contains_currency_or_amount": Counter(),
        "contains_markup": Counter(),
        "quality_warnings": Counter(),
        "quality_reject_reasons": Counter(),
    }
    durations: list[float] = []
    word_counts: list[float] = []
    text_word_counts: list[float] = []
    empty_text = 0

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            fields.update(row.keys())
            for field_name in (
                "gender",
                "language",
                "state",
                "district",
                "contains_digit",
                "contains_date_like",
                "contains_currency_or_amount",
                "contains_markup",
            ):
                counters[field_name][str(row.get(field_name) or "")] += 1
            for warning in _split_values(row.get("quality_warnings")):
                counters["quality_warnings"][warning] += 1
            for reason in _split_values(row.get("quality_reject_reasons")):
                counters["quality_reject_reasons"][reason] += 1

            try:
                durations.append(float(row["duration_seconds"]))
            except (KeyError, TypeError, ValueError):
                pass
            try:
                word_counts.append(float(row["word_count"]))
            except (KeyError, TypeError, ValueError):
                pass

            text = str(row.get(config.text_field) or row.get("text") or "")
            if not text.strip():
                empty_text += 1
            else:
                text_word_counts.append(float(len(text.split())))

    return {
        "run_id": config.run_id,
        "manifest": str(manifest_path),
        "rows": rows,
        "fields": sorted(fields),
        "empty_text": empty_text,
        "duration_seconds": _summarize(durations),
        "word_count_field": _summarize(word_counts),
        "text_word_count": _summarize(text_word_counts),
        "top_counts": {
            name: counter.most_common(config.top_k)
            for name, counter in counters.items()
        },
    }


@app.local_entrypoint()
def main(
    run_id: str,
    manifest_name: str = "manifest.jsonl",
    text_field: str = "normalized_transcript",
    top_k: int = 12,
) -> None:
    report = manifest_stats_remote.remote(
        {
            "run_id": run_id,
            "manifest_name": manifest_name,
            "text_field": text_field,
            "top_k": top_k,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
