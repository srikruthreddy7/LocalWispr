#!/usr/bin/env python3
"""Strict non-Svarah promotion gate for ASR accent candidates.

This tool reads pairwise prediction artifacts from the Modal artifacts volume
and decides whether a candidate model/adapter is allowed to advance. It is
intentionally stricter than the router scripts: a candidate must beat the base
overall while staying inside per-source and per-run regression budgets.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App("localwispr-accent-validation-gate")
image = modal.Image.debian_slim(python_version="3.11")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d")
CURRENCY_RE = re.compile(r"[$₹€£]|(?:\b(?:rs|inr|usd|dollars?|rupees?)\b)", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)


@dataclass(frozen=True)
class GateConfig:
    run_ids: list[str]
    candidate_labels: list[str]
    gate_name: str = "accent-validation-gate"
    manifest_name: str = "pairwise_predictions.jsonl"
    baseline_label: str = "base"
    min_total_samples: int = 500
    min_samples_per_run: int = 50
    min_samples_per_source: int = 50
    min_overall_wer_gain: float = 0.00025
    max_overall_cer_regression: float = 0.0
    max_run_wer_regression: float = 0.0
    max_run_cer_regression: float = 0.00025
    max_source_wer_regression: float = 0.0005
    max_source_cer_regression: float = 0.00075
    max_wer_worse_to_better_ratio: float = 1.0
    max_cer_worse_to_better_ratio: float = 1.0
    require_all_runs_pass: bool = True
    require_all_sources_pass: bool = True
    forbidden_dataset_patterns: list[str] = field(default_factory=lambda: ["svarah"])
    allow_forbidden_datasets: bool = False
    top_examples: int = 20
    max_text_chars: int = 180


@dataclass(frozen=True)
class Prediction:
    label: str
    prediction: str


@dataclass(frozen=True)
class GateRow:
    run_id: str
    index: int
    reference: str
    metadata: dict[str, Any]
    base: Prediction
    candidates: dict[str, Prediction]


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text or "")]


def _levenshtein_distance(left: list[str] | str, right: list[str] | str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _word_ops(reference: str, prediction: str) -> int:
    return _levenshtein_distance(_tokens(reference), _tokens(prediction))


def _char_ops(reference: str, prediction: str) -> int:
    return _levenshtein_distance(reference, prediction)


def _clip(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _matches_forbidden_dataset(value: Any, config: GateConfig) -> str | None:
    if config.allow_forbidden_datasets:
        return None
    text = str(value or "").lower()
    for pattern in config.forbidden_dataset_patterns:
        normalized = pattern.strip().lower()
        if normalized and normalized in text:
            return pattern
    return None


def _collect_datasetish_values(payload: Any, *, parent_key: str = "") -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key).lower()
            next_parent = f"{parent_key}.{key_text}" if parent_key else key_text
            if any(token in key_text for token in ("dataset", "corpus", "split", "source")):
                values.append(str(value))
            if isinstance(value, (dict, list)):
                values.extend(_collect_datasetish_values(value, parent_key=next_parent))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_collect_datasetish_values(item, parent_key=parent_key))
    return values


def _run_forbidden_dataset_match(run_id: str, config: GateConfig) -> str | None:
    report_path = ARTIFACTS_DIR / run_id / "report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for value in _collect_datasetish_values(report):
        match = _matches_forbidden_dataset(value, config)
        if match:
            return match
    return None


def _flat_label_prediction(row: dict[str, Any], label: str) -> str | None:
    return _first_present(
        row,
        (
            f"{label}__normalized_prediction",
            f"{label}__prediction",
            f"{label}_normalized_prediction",
            f"{label}_prediction",
        ),
    )


def _nested_label_prediction(row: dict[str, Any], label: str) -> str | None:
    adapters = row.get("adapters")
    if not isinstance(adapters, dict):
        return None
    payload = adapters.get(label)
    if not isinstance(payload, dict):
        return None
    return _first_present(payload, ("normalized_prediction", "prediction"))


def _prediction_for_label(row: dict[str, Any], label: str) -> str | None:
    if label == "base":
        return _first_present(
            row,
            (
                "base_normalized_prediction",
                "base_prediction",
                "base__normalized_prediction",
                "base__prediction",
            ),
        )
    nested = _nested_label_prediction(row, label)
    if nested is not None:
        return nested
    return _flat_label_prediction(row, label)


def _infer_source(row: GateRow) -> str:
    metadata = row.metadata
    for key in (
        "source_dataset",
        "dataset",
        "source",
        "mix_source",
        "source_config",
        "config",
        "split",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{str(value).strip()}"
    return f"run:{row.run_id}"


def _row_buckets(row: GateRow) -> list[str]:
    buckets = [_infer_source(row)]
    words = len(_tokens(row.reference))
    if words <= 4:
        buckets.append("words:<=4")
    elif words <= 8:
        buckets.append("words:5-8")
    elif words <= 14:
        buckets.append("words:9-14")
    else:
        buckets.append("words:>14")

    duration = None
    for key in ("duration_seconds", "audio_seconds"):
        try:
            duration = float(row.metadata.get(key))
            break
        except (TypeError, ValueError):
            pass
    if duration is not None:
        if duration <= 3:
            buckets.append("duration:<=3")
        elif duration <= 6:
            buckets.append("duration:3-6")
        elif duration <= 10:
            buckets.append("duration:6-10")
        else:
            buckets.append("duration:>10")

    if DIGIT_RE.search(row.reference):
        buckets.append("text:digit")
    if CURRENCY_RE.search(row.reference):
        buckets.append("text:currency")
    if DATE_RE.search(row.reference):
        buckets.append("text:date")
    return buckets


def _load_rows(config: GateConfig) -> tuple[list[GateRow], dict[str, Any]]:
    rows: list[GateRow] = []
    load_report: dict[str, Any] = {}
    for run_id in config.run_ids:
        path = ARTIFACTS_DIR / run_id / config.manifest_name
        run_report = {
            "path": str(path),
            "exists": path.exists(),
            "rows": 0,
            "usable_rows": 0,
            "missing_base": 0,
            "missing_candidate_by_label": Counter(),
            "forbidden_dataset_match": None,
        }
        load_report[run_id] = run_report
        forbidden_match = _run_forbidden_dataset_match(run_id, config)
        if forbidden_match:
            run_report["forbidden_dataset_match"] = forbidden_match
            continue
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                run_report["rows"] += 1
                reference = str(
                    payload.get("normalized_reference") or payload.get("reference") or ""
                )
                if not reference or not _tokens(reference):
                    continue
                base_prediction = _prediction_for_label(payload, config.baseline_label)
                if base_prediction is None:
                    run_report["missing_base"] += 1
                    continue
                candidates: dict[str, Prediction] = {}
                for label in config.candidate_labels:
                    candidate_prediction = _prediction_for_label(payload, label)
                    if candidate_prediction is None:
                        run_report["missing_candidate_by_label"][label] += 1
                        continue
                    candidates[label] = Prediction(label=label, prediction=candidate_prediction)
                if not candidates:
                    continue
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                forbidden_row_match = None
                for value in _collect_datasetish_values(metadata):
                    forbidden_row_match = _matches_forbidden_dataset(value, config)
                    if forbidden_row_match:
                        break
                if forbidden_row_match:
                    run_report["forbidden_dataset_match"] = forbidden_row_match
                    continue
                rows.append(
                    GateRow(
                        run_id=run_id,
                        index=int(payload.get("index") or run_report["rows"] - 1),
                        reference=reference,
                        metadata=metadata,
                        base=Prediction(label=config.baseline_label, prediction=base_prediction),
                        candidates=candidates,
                    )
                )
                run_report["usable_rows"] += 1
        run_report["missing_candidate_by_label"] = dict(run_report["missing_candidate_by_label"])
    return rows, load_report


def _empty_totals() -> Counter[str]:
    return Counter(
        {
            "samples": 0,
            "ref_words": 0,
            "ref_chars": 0,
            "base_wer_ops": 0,
            "base_cer_ops": 0,
            "candidate_wer_ops": 0,
            "candidate_cer_ops": 0,
            "wer_improved": 0,
            "wer_worsened": 0,
            "wer_unchanged": 0,
            "cer_improved": 0,
            "cer_worsened": 0,
            "cer_unchanged": 0,
        }
    )


def _add_row(totals: Counter[str], row: GateRow, candidate: Prediction) -> dict[str, int]:
    ref_words = max(1, len(_tokens(row.reference)))
    ref_chars = max(1, len(row.reference))
    base_wer_ops = _word_ops(row.reference, row.base.prediction)
    candidate_wer_ops = _word_ops(row.reference, candidate.prediction)
    base_cer_ops = _char_ops(row.reference, row.base.prediction)
    candidate_cer_ops = _char_ops(row.reference, candidate.prediction)
    totals["samples"] += 1
    totals["ref_words"] += ref_words
    totals["ref_chars"] += ref_chars
    totals["base_wer_ops"] += base_wer_ops
    totals["candidate_wer_ops"] += candidate_wer_ops
    totals["base_cer_ops"] += base_cer_ops
    totals["candidate_cer_ops"] += candidate_cer_ops
    totals["wer_improved"] += int(candidate_wer_ops < base_wer_ops)
    totals["wer_worsened"] += int(candidate_wer_ops > base_wer_ops)
    totals["wer_unchanged"] += int(candidate_wer_ops == base_wer_ops)
    totals["cer_improved"] += int(candidate_cer_ops < base_cer_ops)
    totals["cer_worsened"] += int(candidate_cer_ops > base_cer_ops)
    totals["cer_unchanged"] += int(candidate_cer_ops == base_cer_ops)
    return {
        "base_wer_ops": base_wer_ops,
        "candidate_wer_ops": candidate_wer_ops,
        "base_cer_ops": base_cer_ops,
        "candidate_cer_ops": candidate_cer_ops,
        "ref_words": ref_words,
        "ref_chars": ref_chars,
    }


def _metrics(totals: Counter[str]) -> dict[str, Any]:
    samples = int(totals["samples"])
    base_wer = totals["base_wer_ops"] / max(1, totals["ref_words"])
    candidate_wer = totals["candidate_wer_ops"] / max(1, totals["ref_words"])
    base_cer = totals["base_cer_ops"] / max(1, totals["ref_chars"])
    candidate_cer = totals["candidate_cer_ops"] / max(1, totals["ref_chars"])
    return {
        "samples": samples,
        "base_wer": base_wer,
        "candidate_wer": candidate_wer,
        "delta_wer": candidate_wer - base_wer,
        "base_cer": base_cer,
        "candidate_cer": candidate_cer,
        "delta_cer": candidate_cer - base_cer,
        "wer_counts": {
            "improved": int(totals["wer_improved"]),
            "worsened": int(totals["wer_worsened"]),
            "unchanged": int(totals["wer_unchanged"]),
        },
        "cer_counts": {
            "improved": int(totals["cer_improved"]),
            "worsened": int(totals["cer_worsened"]),
            "unchanged": int(totals["cer_unchanged"]),
        },
        "wer_worse_to_better_ratio": totals["wer_worsened"] / max(1, totals["wer_improved"]),
        "cer_worse_to_better_ratio": totals["cer_worsened"] / max(1, totals["cer_improved"]),
    }


def _gate_failures(
    *,
    overall: dict[str, Any],
    per_run: dict[str, dict[str, Any]],
    per_source: dict[str, dict[str, Any]],
    config: GateConfig,
) -> list[str]:
    failures: list[str] = []
    if overall["samples"] < config.min_total_samples:
        failures.append(
            f"total samples {overall['samples']} < min_total_samples {config.min_total_samples}"
        )
    if overall["delta_wer"] > -config.min_overall_wer_gain:
        failures.append(
            f"overall WER gain {-overall['delta_wer']:.8f} < required {config.min_overall_wer_gain:.8f}"
        )
    if overall["delta_cer"] > config.max_overall_cer_regression:
        failures.append(
            f"overall CER delta {overall['delta_cer']:.8f} > max {config.max_overall_cer_regression:.8f}"
        )
    if overall["wer_worse_to_better_ratio"] > config.max_wer_worse_to_better_ratio:
        failures.append(
            "WER worsened/improved ratio "
            f"{overall['wer_worse_to_better_ratio']:.4f} > {config.max_wer_worse_to_better_ratio:.4f}"
        )
    if overall["cer_worse_to_better_ratio"] > config.max_cer_worse_to_better_ratio:
        failures.append(
            "CER worsened/improved ratio "
            f"{overall['cer_worse_to_better_ratio']:.4f} > {config.max_cer_worse_to_better_ratio:.4f}"
        )

    for run_id, metrics in per_run.items():
        if metrics["samples"] < config.min_samples_per_run:
            failures.append(
                f"run {run_id} samples {metrics['samples']} < min_samples_per_run {config.min_samples_per_run}"
            )
            continue
        if not config.require_all_runs_pass:
            continue
        if metrics["delta_wer"] > config.max_run_wer_regression:
            failures.append(
                f"run {run_id} WER delta {metrics['delta_wer']:.8f} > max {config.max_run_wer_regression:.8f}"
            )
        if metrics["delta_cer"] > config.max_run_cer_regression:
            failures.append(
                f"run {run_id} CER delta {metrics['delta_cer']:.8f} > max {config.max_run_cer_regression:.8f}"
            )

    for source, metrics in per_source.items():
        if metrics["samples"] < config.min_samples_per_source:
            continue
        if not config.require_all_sources_pass:
            continue
        if metrics["delta_wer"] > config.max_source_wer_regression:
            failures.append(
                f"source {source} WER delta {metrics['delta_wer']:.8f} > max {config.max_source_wer_regression:.8f}"
            )
        if metrics["delta_cer"] > config.max_source_cer_regression:
            failures.append(
                f"source {source} CER delta {metrics['delta_cer']:.8f} > max {config.max_source_cer_regression:.8f}"
            )
    return failures


def _example(
    row: GateRow,
    candidate: Prediction,
    row_metrics: dict[str, int],
    *,
    max_text_chars: int,
) -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "index": row.index,
        "source": _infer_source(row),
        "delta_wer_ops": row_metrics["candidate_wer_ops"] - row_metrics["base_wer_ops"],
        "delta_cer_ops": row_metrics["candidate_cer_ops"] - row_metrics["base_cer_ops"],
        "reference": _clip(row.reference, max_text_chars),
        "base": _clip(row.base.prediction, max_text_chars),
        "candidate": _clip(candidate.prediction, max_text_chars),
    }


def _evaluate_candidate(rows: list[GateRow], label: str, config: GateConfig) -> dict[str, Any]:
    overall_totals = _empty_totals()
    per_run_totals: dict[str, Counter[str]] = {}
    per_source_totals: dict[str, Counter[str]] = {}
    per_bucket_totals: dict[str, Counter[str]] = {}
    regression_examples: list[dict[str, Any]] = []
    improvement_examples: list[dict[str, Any]] = []
    usable_rows = 0
    missing_rows = 0

    for row in rows:
        candidate = row.candidates.get(label)
        if candidate is None:
            missing_rows += 1
            continue
        usable_rows += 1
        row_metrics = _add_row(overall_totals, row, candidate)
        per_run_totals.setdefault(row.run_id, _empty_totals())
        _add_row(per_run_totals[row.run_id], row, candidate)
        source = _infer_source(row)
        per_source_totals.setdefault(source, _empty_totals())
        _add_row(per_source_totals[source], row, candidate)
        for bucket in _row_buckets(row):
            per_bucket_totals.setdefault(bucket, _empty_totals())
            _add_row(per_bucket_totals[bucket], row, candidate)

        delta_cer_ops = row_metrics["candidate_cer_ops"] - row_metrics["base_cer_ops"]
        example = _example(row, candidate, row_metrics, max_text_chars=config.max_text_chars)
        if delta_cer_ops > 0:
            regression_examples.append(example)
        elif delta_cer_ops < 0:
            improvement_examples.append(example)

    overall = _metrics(overall_totals)
    per_run = {run_id: _metrics(totals) for run_id, totals in sorted(per_run_totals.items())}
    per_source = {
        source: _metrics(totals)
        for source, totals in sorted(per_source_totals.items())
    }
    per_bucket = {
        bucket: _metrics(totals)
        for bucket, totals in sorted(per_bucket_totals.items())
    }
    failures = _gate_failures(
        overall=overall,
        per_run=per_run,
        per_source=per_source,
        config=config,
    )
    regression_examples.sort(
        key=lambda item: (int(item["delta_cer_ops"]), int(item["delta_wer_ops"])),
        reverse=True,
    )
    improvement_examples.sort(
        key=lambda item: (int(item["delta_cer_ops"]), int(item["delta_wer_ops"])),
    )
    return {
        "label": label,
        "passed": not failures,
        "failures": failures,
        "usable_rows": usable_rows,
        "missing_rows": missing_rows,
        "overall": overall,
        "per_run": per_run,
        "per_source": per_source,
        "per_bucket": per_bucket,
        "top_regressions": regression_examples[: config.top_examples],
        "top_improvements": improvement_examples[: config.top_examples],
    }


def _rank_key(result: dict[str, Any]) -> tuple[int, float, float, float]:
    return (
        0 if result["passed"] else 1,
        float(result["overall"]["delta_wer"]),
        float(result["overall"]["delta_cer"]),
        float(result["overall"]["wer_worse_to_better_ratio"]),
    )


@app.function(
    image=image,
    timeout=60 * 20,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def validation_gate_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = GateConfig(**config_payload)
    if not config.run_ids:
        raise ValueError("run_ids is required")
    if not config.candidate_labels:
        raise ValueError("candidate_labels is required")

    rows, load_report = _load_rows(config)
    candidate_results = [_evaluate_candidate(rows, label, config) for label in config.candidate_labels]
    forbidden_runs = {
        run_id: payload.get("forbidden_dataset_match")
        for run_id, payload in load_report.items()
        if isinstance(payload, dict) and payload.get("forbidden_dataset_match")
    }
    if forbidden_runs and not config.allow_forbidden_datasets:
        for result in candidate_results:
            result["passed"] = False
            result["failures"] = list(result.get("failures") or []) + [
                f"forbidden dataset pattern matched in run artifacts: {forbidden_runs}"
            ]
    candidate_results.sort(key=_rank_key)
    passing = [result["label"] for result in candidate_results if result["passed"]]

    run_id = f"{config.gate_name}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "created_at_utc": _now_iso(),
        "config": asdict(config),
        "loaded": {
            "rows": len(rows),
            "per_run": load_report,
        },
        "decision": {
            "passed": bool(passing),
            "selected_label": passing[0] if passing else None,
            "passing_labels": passing,
            "read": (
                "At least one candidate passed the non-Svarah promotion gate."
                if passing
                else "No candidate passed; do not run Svarah or production promotion from this result."
            ),
        },
        "candidates": candidate_results,
        "artifacts": {
            "run_dir": str(run_dir),
            "report_path": str(run_dir / "report.json"),
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return report


@app.local_entrypoint()
def main(
    run_ids: str,
    candidate_labels: str,
    gate_name: str = "accent-validation-gate",
    manifest_name: str = "pairwise_predictions.jsonl",
    baseline_label: str = "base",
    min_total_samples: int = 500,
    min_samples_per_run: int = 50,
    min_samples_per_source: int = 50,
    min_overall_wer_gain: float = 0.00025,
    max_overall_cer_regression: float = 0.0,
    max_run_wer_regression: float = 0.0,
    max_run_cer_regression: float = 0.00025,
    max_source_wer_regression: float = 0.0005,
    max_source_cer_regression: float = 0.00075,
    max_wer_worse_to_better_ratio: float = 1.0,
    max_cer_worse_to_better_ratio: float = 1.0,
    require_all_runs_pass: bool = True,
    require_all_sources_pass: bool = True,
    forbidden_dataset_patterns: str = "svarah",
    allow_forbidden_datasets: bool = False,
    top_examples: int = 20,
    max_text_chars: int = 180,
) -> None:
    report = validation_gate_remote.remote(
        {
            "run_ids": _parse_csv(run_ids),
            "candidate_labels": _parse_csv(candidate_labels),
            "gate_name": gate_name,
            "manifest_name": manifest_name,
            "baseline_label": baseline_label,
            "min_total_samples": min_total_samples,
            "min_samples_per_run": min_samples_per_run,
            "min_samples_per_source": min_samples_per_source,
            "min_overall_wer_gain": min_overall_wer_gain,
            "max_overall_cer_regression": max_overall_cer_regression,
            "max_run_wer_regression": max_run_wer_regression,
            "max_run_cer_regression": max_run_cer_regression,
            "max_source_wer_regression": max_source_wer_regression,
            "max_source_cer_regression": max_source_cer_regression,
            "max_wer_worse_to_better_ratio": max_wer_worse_to_better_ratio,
            "max_cer_worse_to_better_ratio": max_cer_worse_to_better_ratio,
            "require_all_runs_pass": require_all_runs_pass,
            "require_all_sources_pass": require_all_sources_pass,
            "forbidden_dataset_patterns": _parse_csv(forbidden_dataset_patterns),
            "allow_forbidden_datasets": allow_forbidden_datasets,
            "top_examples": top_examples,
            "max_text_chars": max_text_chars,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
