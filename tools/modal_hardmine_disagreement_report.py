#!/usr/bin/env python3
"""Modal-side disagreement report for hard-mined Whisper manifests.

This keeps gated audio and manifests inside the Modal artifacts volume. It
prints aggregate error patterns and a capped set of short examples for rows
where large-v3 is close to the reference and turbo is not.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App("localwispr-hardmine-disagreement-report")
image = modal.Image.debian_slim(python_version="3.11")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
    "your",
}
NUMBER_CANONICAL = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}
SPELLING_CANONICAL = {
    "colour": "color",
    "colours": "colors",
    "coloured": "colored",
    "theatre": "theater",
    "theatres": "theaters",
    "centre": "center",
    "centres": "centers",
    "defence": "defense",
    "defences": "defenses",
    "offence": "offense",
    "offences": "offenses",
    "metre": "meter",
    "metres": "meters",
    "litre": "liter",
    "litres": "liters",
    "travelling": "traveling",
    "travelled": "traveled",
    "cancelled": "canceled",
    "cancelling": "canceling",
    "favour": "favor",
    "favours": "favors",
    "favourite": "favorite",
    "favourites": "favorites",
}
CONTRACTION_FRAGMENTS = {
    "can",
    "couldn",
    "didn",
    "doesn",
    "don",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "let",
    "mustn",
    "shan",
    "shouldn",
    "that",
    "there",
    "wasn",
    "weren",
    "wouldn",
}


@dataclass
class DisagreementReportConfig:
    run_ids: list[str]
    source_kind: str = "hardmine"
    manifest_name: str = "manifest.jsonl"
    turbo_label: str = "turbo"
    teacher_label: str = "teacher"
    teacher_max_cer: float = 0.02
    turbo_min_cer: float = 0.04
    teacher_max_wer: float | None = None
    turbo_min_wer: float | None = None
    max_rows_per_run: int | None = None
    min_word_count: int | None = None
    max_word_count: int | None = None
    max_examples: int = 20
    top_k: int = 30
    max_text_chars: int = 180
    include_examples: bool = True


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


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


def _tokens(text: Any) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(str(text or ""))]


def _clip(text: Any, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index_left, char_left in enumerate(left, start=1):
        current = [index_left]
        for index_right, char_right in enumerate(right, start=1):
            current.append(
                min(
                    previous[index_right] + 1,
                    current[index_right - 1] + 1,
                    previous[index_right - 1] + (char_left != char_right),
                )
            )
        previous = current
    return previous[-1]


def _char_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    return 1.0 - (_levenshtein_distance(left, right) / denominator)


def _canonical_token(token: str) -> str:
    lowered = token.lower().strip()
    lowered = lowered.replace("'", "")
    lowered = NUMBER_CANONICAL.get(lowered, lowered)
    lowered = SPELLING_CANONICAL.get(lowered, lowered)
    return lowered


def _is_format_pair(turbo_token: str, ref_token: str) -> bool:
    turbo = turbo_token.lower()
    ref = ref_token.lower()
    if _canonical_token(turbo) == _canonical_token(ref):
        return True
    if turbo.replace("'", "").startswith(ref.replace("'", "")) and ref.replace("'", "") in CONTRACTION_FRAGMENTS:
        return True
    if ref.replace("'", "").startswith(turbo.replace("'", "")) and turbo.replace("'", "") in CONTRACTION_FRAGMENTS:
        return True
    return False


def _bucket(value: float | None, buckets: tuple[float, ...]) -> str:
    if value is None:
        return "unknown"
    for upper in buckets:
        if value <= upper:
            return f"<= {upper:g}"
    return f"> {buckets[-1]:g}"


def _row_passes(row: dict[str, Any], config: DisagreementReportConfig) -> bool:
    teacher_cer = _as_float(row.get("teacher_cer"))
    turbo_cer = _as_float(row.get("turbo_cer"))
    if teacher_cer is None or turbo_cer is None:
        return False
    if teacher_cer > config.teacher_max_cer:
        return False
    if turbo_cer < config.turbo_min_cer:
        return False

    teacher_wer = _as_float(row.get("teacher_wer"))
    turbo_wer = _as_float(row.get("turbo_wer"))
    if config.teacher_max_wer is not None and (
        teacher_wer is None or teacher_wer > config.teacher_max_wer
    ):
        return False
    if config.turbo_min_wer is not None and (
        turbo_wer is None or turbo_wer < config.turbo_min_wer
    ):
        return False
    word_count = len(_tokens(row.get("text") or row.get("raw_transcript") or ""))
    if config.min_word_count is not None and word_count < config.min_word_count:
        return False
    if config.max_word_count is not None and word_count > config.max_word_count:
        return False
    return True


def _normalize_row_for_report(
    row: dict[str, Any],
    *,
    run_id: str,
    config: DisagreementReportConfig,
) -> dict[str, Any]:
    source_kind = config.source_kind.strip().lower()
    if source_kind == "hardmine":
        return row
    if source_kind != "pairwise":
        raise ValueError("source_kind must be one of: hardmine, pairwise")

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

    def prediction_for(label: str) -> dict[str, Any]:
        if label == "base":
            return {
                "prediction": row.get("base_prediction"),
                "normalized_prediction": row.get("base_normalized_prediction"),
                "wer": row.get("base_wer"),
                "cer": row.get("base_cer"),
            }

        adapters = row.get("adapters") if isinstance(row.get("adapters"), dict) else {}
        adapter_payload = adapters.get(label)
        if isinstance(adapter_payload, dict):
            return {
                "prediction": adapter_payload.get("prediction"),
                "normalized_prediction": adapter_payload.get("normalized_prediction"),
                "wer": adapter_payload.get("wer"),
                "cer": adapter_payload.get("cer"),
            }

        return {
            "prediction": row.get(f"{label}__prediction"),
            "normalized_prediction": row.get(f"{label}__normalized_prediction"),
            "wer": row.get(f"{label}__wer"),
            "cer": row.get(f"{label}__cer"),
        }

    turbo_payload = prediction_for(config.turbo_label)
    teacher_payload = prediction_for(config.teacher_label)
    normalized_reference = row.get("normalized_reference") or row.get("reference") or ""
    return {
        "manifest_index": row.get("index"),
        "dataset_index": row.get("index"),
        "text": normalized_reference,
        "raw_transcript": row.get("reference") or normalized_reference,
        "source_dataset": metadata.get("source_dataset") or metadata.get("dataset") or run_id,
        "source_config": metadata.get("source_config") or metadata.get("config"),
        "source_split": metadata.get("source_split") or metadata.get("split"),
        "turbo_prediction": turbo_payload.get("normalized_prediction")
        or turbo_payload.get("prediction"),
        "teacher_prediction": teacher_payload.get("normalized_prediction")
        or teacher_payload.get("prediction"),
        "turbo_wer": turbo_payload.get("wer"),
        "turbo_cer": turbo_payload.get("cer"),
        "teacher_wer": teacher_payload.get("wer"),
        "teacher_cer": teacher_payload.get("cer"),
        "selection_score": (
            (_as_float(turbo_payload.get("cer")) or 0.0)
            - (_as_float(teacher_payload.get("cer")) or 0.0)
        ),
        **{f"metadata_{key}": value for key, value in metadata.items() if isinstance(key, str)},
    }


def _collect_token_diffs(
    reference: str,
    turbo: str,
    counters: dict[str, Counter[str]],
) -> dict[str, int]:
    ref_tokens = _tokens(reference)
    turbo_tokens = _tokens(turbo)
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=turbo_tokens, autojunk=False)
    op_counts = Counter()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        op_counts[tag] += max(i2 - i1, j2 - j1)
        ref_span = ref_tokens[i1:i2]
        turbo_span = turbo_tokens[j1:j2]

        if tag == "delete":
            for token in ref_span:
                counters["deleted_reference_tokens"][token] += 1
            if ref_span:
                counters["deleted_reference_phrases"][" ".join(ref_span[:4])] += 1
        elif tag == "insert":
            for token in turbo_span:
                counters["inserted_turbo_tokens"][token] += 1
            if turbo_span:
                counters["inserted_turbo_phrases"][" ".join(turbo_span[:4])] += 1
        else:
            pairs = list(zip(ref_span, turbo_span))
            for ref_token, turbo_token in pairs:
                counters["substitution_pairs"][f"{turbo_token} -> {ref_token}"] += 1
                similarity = _char_similarity(ref_token, turbo_token)
                if _is_format_pair(turbo_token, ref_token):
                    counters["format_artifact_pairs"][f"{turbo_token} -> {ref_token}"] += 1
                    op_counts["format_like"] += 1
                    continue
                op_counts["semantic_substitution"] += 1
                counters["semantic_substitution_pairs"][f"{turbo_token} -> {ref_token}"] += 1
                if ref_token not in STOPWORDS and turbo_token not in STOPWORDS:
                    counters["content_word_substitution_pairs"][f"{turbo_token} -> {ref_token}"] += 1
                if similarity >= 0.55:
                    counters["close_soundalike_pairs"][f"{turbo_token} -> {ref_token}"] += 1
                    counters["close_content_pairs"][f"{turbo_token} -> {ref_token}"] += 1
                if len(ref_token) <= 3 or len(turbo_token) <= 3:
                    counters["short_word_substitutions"][f"{turbo_token} -> {ref_token}"] += 1
            if len(ref_span) != len(turbo_span):
                counters["uneven_substitution_spans"][
                    f"{' '.join(turbo_span[:4])} -> {' '.join(ref_span[:4])}"
                ] += 1

    return dict(op_counts)


def _categorize_row(row: dict[str, Any], op_counts: dict[str, int]) -> list[str]:
    categories: list[str] = []
    text = str(row.get("text") or row.get("raw_transcript") or "")
    turbo = str(row.get("turbo_prediction") or "")
    if DIGIT_RE.search(text) or row.get("contains_digit") is True:
        categories.append("numeric_reference")
    if _as_float(row.get("duration_seconds")) is not None:
        categories.append(f"duration_{_bucket(_as_float(row.get('duration_seconds')), (3, 6, 10, 15, 25))}")
    word_count = len(_tokens(text))
    categories.append(f"words_{_bucket(float(word_count), (4, 8, 14, 24, 40))}")
    if op_counts.get("delete", 0) > op_counts.get("insert", 0) + op_counts.get("replace", 0):
        categories.append("mostly_deletion")
    elif op_counts.get("insert", 0) > op_counts.get("delete", 0) + op_counts.get("replace", 0):
        categories.append("mostly_insertion")
    elif op_counts.get("replace", 0):
        categories.append("has_substitution")
    if op_counts.get("format_like", 0) and op_counts.get("format_like", 0) >= op_counts.get("semantic_substitution", 0):
        categories.append("format_heavy")
    if op_counts.get("semantic_substitution", 0):
        categories.append("semantic_substitution")
    if len(_tokens(turbo)) < max(1, len(_tokens(text)) // 2):
        categories.append("turbo_too_short")
    return categories


def _example_payload(
    run_id: str,
    row: dict[str, Any],
    categories: list[str],
    max_text_chars: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "source_dataset": row.get("source_dataset"),
        "dataset_index": row.get("dataset_index"),
        "source_split": row.get("source_split"),
        "turbo_cer": _as_float(row.get("turbo_cer")),
        "teacher_cer": _as_float(row.get("teacher_cer")),
        "turbo_wer": _as_float(row.get("turbo_wer")),
        "teacher_wer": _as_float(row.get("teacher_wer")),
        "selection_score": _as_float(row.get("selection_score")),
        "categories": categories[:6],
        "reference": _clip(row.get("text") or row.get("raw_transcript"), max_text_chars),
        "turbo": _clip(row.get("turbo_prediction"), max_text_chars),
        "large_v3": _clip(row.get("teacher_prediction"), max_text_chars),
    }


@app.function(
    image=image,
    timeout=60 * 15,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def disagreement_report_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = DisagreementReportConfig(**config_payload)
    per_run: dict[str, Any] = {}
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    turbo_cer_values: list[float] = []
    teacher_cer_values: list[float] = []
    turbo_wer_values: list[float] = []
    teacher_wer_values: list[float] = []
    selected_by_source = Counter()
    selected_by_run = Counter()
    selected_by_category = Counter()
    total_rows = 0
    selected_rows = 0

    for run_id in config.run_ids:
        manifest_path = ARTIFACTS_DIR / run_id / config.manifest_name
        run_payload: dict[str, Any] = {
            "manifest": str(manifest_path),
            "exists": manifest_path.exists(),
            "rows": 0,
            "selected_rows": 0,
            "missing_prediction_rows": 0,
        }
        per_run[run_id] = run_payload
        if not manifest_path.exists():
            continue

        run_turbo_cer: list[float] = []
        run_teacher_cer: list[float] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if config.max_rows_per_run is not None and run_payload["rows"] >= config.max_rows_per_run:
                    break
                if not line.strip():
                    continue
                raw_row = json.loads(line)
                row = _normalize_row_for_report(raw_row, run_id=run_id, config=config)
                run_payload["rows"] += 1
                total_rows += 1
                if not row.get("turbo_prediction") or not row.get("teacher_prediction"):
                    run_payload["missing_prediction_rows"] += 1
                    continue
                if not _row_passes(row, config):
                    continue

                selected_rows += 1
                run_payload["selected_rows"] += 1
                selected_by_run[run_id] += 1
                source = str(row.get("source_dataset") or "unknown")
                selected_by_source[source] += 1

                turbo_cer = _as_float(row.get("turbo_cer"))
                teacher_cer = _as_float(row.get("teacher_cer"))
                turbo_wer = _as_float(row.get("turbo_wer"))
                teacher_wer = _as_float(row.get("teacher_wer"))
                if turbo_cer is not None:
                    turbo_cer_values.append(turbo_cer)
                    run_turbo_cer.append(turbo_cer)
                if teacher_cer is not None:
                    teacher_cer_values.append(teacher_cer)
                    run_teacher_cer.append(teacher_cer)
                if turbo_wer is not None:
                    turbo_wer_values.append(turbo_wer)
                if teacher_wer is not None:
                    teacher_wer_values.append(teacher_wer)

                reference = str(row.get("text") or row.get("raw_transcript") or "")
                turbo = str(row.get("turbo_prediction") or "")
                op_counts = _collect_token_diffs(reference, turbo, counters)
                categories = _categorize_row(row, op_counts)
                for category in categories:
                    selected_by_category[category] += 1
                counters["line_numbers_by_run"][f"{run_id}:{line_number}"] += 1

                if config.include_examples:
                    score = _as_float(row.get("selection_score"))
                    sort_score = score if score is not None else (turbo_cer or 0.0) - (teacher_cer or 0.0)
                    example = _example_payload(run_id, row, categories, config.max_text_chars)
                    example["_sort_score"] = sort_score
                    examples.append(example)

        run_payload["selected_turbo_cer"] = _summarize(run_turbo_cer)
        run_payload["selected_teacher_cer"] = _summarize(run_teacher_cer)

    examples = sorted(
        examples,
        key=lambda item: (
            -float(item.get("_sort_score") or 0.0),
            str(item.get("run_id") or ""),
            str(item.get("dataset_index") or ""),
        ),
    )
    capped_examples = []
    seen_sources = Counter()
    for example in examples:
        source = str(example.get("source_dataset") or "unknown")
        if seen_sources[source] >= max(3, config.max_examples // max(1, len(selected_by_source) or 1)):
            continue
        seen_sources[source] += 1
        example.pop("_sort_score", None)
        capped_examples.append(example)
        if len(capped_examples) >= config.max_examples:
            break
    if len(capped_examples) < config.max_examples:
        used = {(item["run_id"], item.get("dataset_index")) for item in capped_examples}
        for example in examples:
            key = (example["run_id"], example.get("dataset_index"))
            if key in used:
                continue
            example.pop("_sort_score", None)
            capped_examples.append(example)
            if len(capped_examples) >= config.max_examples:
                break

    return {
        "config": {
            "run_ids": config.run_ids,
            "source_kind": config.source_kind,
            "manifest_name": config.manifest_name,
            "turbo_label": config.turbo_label,
            "teacher_label": config.teacher_label,
            "teacher_max_cer": config.teacher_max_cer,
            "turbo_min_cer": config.turbo_min_cer,
            "teacher_max_wer": config.teacher_max_wer,
            "turbo_min_wer": config.turbo_min_wer,
            "min_word_count": config.min_word_count,
            "max_word_count": config.max_word_count,
        },
        "rows": {
            "total_scanned": total_rows,
            "selected_large_v3_clean_turbo_wrong": selected_rows,
        },
        "per_run": per_run,
        "selected_by_run": selected_by_run.most_common(),
        "selected_by_source": selected_by_source.most_common(),
        "selected_by_category": selected_by_category.most_common(config.top_k),
        "metrics": {
            "turbo_cer": _summarize(turbo_cer_values),
            "teacher_cer": _summarize(teacher_cer_values),
            "turbo_wer": _summarize(turbo_wer_values),
            "teacher_wer": _summarize(teacher_wer_values),
            "mean_cer_gap": (
                _mean(turbo_cer_values) - _mean(teacher_cer_values)
                if turbo_cer_values and teacher_cer_values
                else None
            ),
        },
        "top_error_patterns": {
            "substitution_pairs": counters["substitution_pairs"].most_common(config.top_k),
            "semantic_substitution_pairs": counters["semantic_substitution_pairs"].most_common(config.top_k),
            "content_word_substitution_pairs": counters["content_word_substitution_pairs"].most_common(config.top_k),
            "close_soundalike_pairs": counters["close_soundalike_pairs"].most_common(config.top_k),
            "close_content_pairs": counters["close_content_pairs"].most_common(config.top_k),
            "format_artifact_pairs": counters["format_artifact_pairs"].most_common(config.top_k),
            "short_word_substitutions": counters["short_word_substitutions"].most_common(config.top_k),
            "deleted_reference_tokens": counters["deleted_reference_tokens"].most_common(config.top_k),
            "deleted_reference_phrases": counters["deleted_reference_phrases"].most_common(config.top_k),
            "inserted_turbo_tokens": counters["inserted_turbo_tokens"].most_common(config.top_k),
            "inserted_turbo_phrases": counters["inserted_turbo_phrases"].most_common(config.top_k),
            "uneven_substitution_spans": counters["uneven_substitution_spans"].most_common(config.top_k),
        },
        "examples": capped_examples,
    }


@app.local_entrypoint()
def main(
    run_ids: str,
    source_kind: str = "hardmine",
    manifest_name: str = "manifest.jsonl",
    turbo_label: str = "turbo",
    teacher_label: str = "teacher",
    teacher_max_cer: float = 0.02,
    turbo_min_cer: float = 0.04,
    teacher_max_wer: float | None = None,
    turbo_min_wer: float | None = None,
    max_rows_per_run: int | None = None,
    min_word_count: int | None = None,
    max_word_count: int | None = None,
    max_examples: int = 20,
    top_k: int = 30,
    max_text_chars: int = 180,
    include_examples: bool = True,
) -> None:
    parsed_run_ids = [run_id.strip() for run_id in run_ids.split(",") if run_id.strip()]
    report = disagreement_report_remote.remote(
        {
            "run_ids": parsed_run_ids,
            "source_kind": source_kind,
            "manifest_name": manifest_name,
            "turbo_label": turbo_label,
            "teacher_label": teacher_label,
            "teacher_max_cer": teacher_max_cer,
            "turbo_min_cer": turbo_min_cer,
            "teacher_max_wer": teacher_max_wer,
            "turbo_min_wer": turbo_min_wer,
            "max_rows_per_run": max_rows_per_run,
            "min_word_count": min_word_count,
            "max_word_count": max_word_count,
            "max_examples": max_examples,
            "top_k": top_k,
            "max_text_chars": max_text_chars,
            "include_examples": include_examples,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
