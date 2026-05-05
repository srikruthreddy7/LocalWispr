#!/usr/bin/env python3
"""Build Cohere-specific hard-example manifests from pairwise ASR artifacts.

The intended use is:

1. Run a non-Svarah bakeoff with base Cohere and a stronger teacher.
2. Feed the resulting pairwise_predictions.jsonl here.
3. Train Cohere only on rows where base Cohere is wrong and the teacher is clean.

The output is a training JSONL with local audio paths preserved inside the
Modal artifacts volume.
"""

from __future__ import annotations

import difflib
import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App("localwispr-pairwise-hard-mine-manifest")
image = modal.Image.debian_slim(python_version="3.11")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d")
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")
CURRENCY_RE = re.compile(r"[$₹€£]|\b(?:rs|inr|usd|dollars?|rupees?|amount|balance|account|bank)\b", re.IGNORECASE)
MARKUP_RE = re.compile(r"<[^>]+>|&[a-z]+;|\[[^\]]+\]|\{[^}]+\}")
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


@dataclass(frozen=True)
class SourceSpec:
    run_id: str
    limit: int
    label: str


@dataclass(frozen=True)
class HardMineConfig:
    manifest_name: str = "pairwise_predictions.jsonl"
    output_name: str = "cohere-hardmine"
    sources: list[SourceSpec] | None = None
    base_label: str = "cohere_base"
    teacher_label: str = "whisper_large_v3"
    target_text_source: str = "teacher"
    seed: int = 42
    min_base_cer: float = 0.04
    max_base_cer: float = 0.35
    max_teacher_cer: float = 0.02
    min_selection_score: float = 0.03
    min_word_count: int = 4
    max_word_count: int = 18
    min_duration_seconds: float = 1.5
    max_duration_seconds: float = 8.0
    max_samples_per_speaker: int = 8
    dedupe_text: bool = True
    reject_format_sensitive: bool = True
    require_content_error: bool = True


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parse_sources(value: str) -> list[dict[str, Any]]:
    sources = []
    for raw_source in value.split(","):
        raw_source = raw_source.strip()
        if not raw_source:
            continue
        parts = raw_source.split(":")
        if len(parts) != 3:
            raise ValueError("Each source must be run_id:limit:label")
        run_id, limit, label = parts
        sources.append({"run_id": run_id, "limit": int(limit), "label": label})
    if not sources:
        raise ValueError("At least one source is required")
    return sources


def _tokens(text: Any) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(str(text or ""))]


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


def _wer(reference: str, prediction: str) -> float:
    return _levenshtein_distance(_tokens(reference), _tokens(prediction)) / max(1, len(_tokens(reference)))


def _cer(reference: str, prediction: str) -> float:
    return _levenshtein_distance(reference, prediction) / max(1, len(reference))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _prediction_for_label(row: dict[str, Any], label: str) -> str | None:
    if label == "base":
        return _first_present(row, ("base_normalized_prediction", "base_prediction"))
    adapters = row.get("adapters") if isinstance(row.get("adapters"), dict) else {}
    adapter_payload = adapters.get(label)
    if isinstance(adapter_payload, dict):
        value = _first_present(adapter_payload, ("normalized_prediction", "prediction"))
        if value is not None:
            return value
    return _first_present(
        row,
        (
            f"{label}__normalized_prediction",
            f"{label}__prediction",
            f"{label}_normalized_prediction",
            f"{label}_prediction",
        ),
    )


def _metric_for_label(row: dict[str, Any], label: str, metric: str, reference: str, prediction: str) -> float:
    adapters = row.get("adapters") if isinstance(row.get("adapters"), dict) else {}
    adapter_payload = adapters.get(label)
    if isinstance(adapter_payload, dict):
        value = _as_float(adapter_payload.get(metric))
        if value is not None:
            return value
    if label == "base":
        value = _as_float(row.get(f"base_{metric}"))
        if value is not None:
            return value
    value = _as_float(row.get(f"{label}__{metric}"))
    if value is not None:
        return value
    return _cer(reference, prediction) if metric == "cer" else _wer(reference, prediction)


def _audio_path(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for payload in (metadata, row):
        value = _first_present(
            payload,
            ("local_audio_path", "audio_path", "audio_filepath", "path", "file", "wav"),
        )
        if value is not None:
            return value
    return None


def _speaker_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for payload in (metadata, row):
        for key in ("speaker_key", "client_id", "speaker_id", "Speaker_ID", "speaker", "user_id", "utt_spk"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return f"{key}:{str(value).strip()}"
    return ""


def _duration_seconds(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for payload in (metadata, row):
        for key in ("duration_seconds", "audio_seconds", "duration", "duration_sec"):
            value = _as_float(payload.get(key))
            if value is not None:
                return value
    return None


def _text_key(text: str) -> str:
    return re.sub(r"\s+", " ", " ".join(_tokens(text))).strip()


def _is_format_sensitive(text: str) -> bool:
    return (
        bool(DIGIT_RE.search(text))
        or bool(DATE_RE.search(text))
        or bool(CURRENCY_RE.search(text))
        or bool(MARKUP_RE.search(text))
    )


def _is_content_token(token: str) -> bool:
    plain = token.lower().replace("'", "")
    return bool(plain) and plain not in STOPWORDS and len(plain) > 1 and not plain.isdigit()


def _content_error_count(reference: str, prediction: str) -> int:
    ref_tokens = _tokens(reference)
    pred_tokens = _tokens(prediction)
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=pred_tokens, autojunk=False)
    count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        ref_span = ref_tokens[i1:i2]
        pred_span = pred_tokens[j1:j2]
        if tag == "delete":
            count += sum(1 for token in ref_span if _is_content_token(token))
        elif tag == "insert":
            count += sum(1 for token in pred_span if _is_content_token(token))
        else:
            count += sum(
                1
                for token in ref_span + pred_span
                if _is_content_token(token)
            )
    return count


def _target_text(row: dict[str, Any], *, teacher_prediction: str, config: HardMineConfig) -> str:
    if config.target_text_source == "teacher":
        return teacher_prediction
    if config.target_text_source == "reference":
        return str(row.get("normalized_reference") or row.get("reference") or "")
    raise ValueError("target_text_source must be one of: teacher, reference")


def _normalize_pairwise_row(
    row: dict[str, Any],
    *,
    source: SourceSpec,
    config: HardMineConfig,
) -> tuple[dict[str, Any] | None, str]:
    reference = str(row.get("normalized_reference") or row.get("reference") or "")
    if not reference or not _tokens(reference):
        return None, "missing_reference"
    audio_path = _audio_path(row)
    if not audio_path:
        return None, "missing_audio"
    base_prediction = _prediction_for_label(row, config.base_label)
    teacher_prediction = _prediction_for_label(row, config.teacher_label)
    if not base_prediction:
        return None, "missing_base_prediction"
    if not teacher_prediction:
        return None, "missing_teacher_prediction"

    base_cer = _metric_for_label(row, config.base_label, "cer", reference, base_prediction)
    base_wer = _metric_for_label(row, config.base_label, "wer", reference, base_prediction)
    teacher_cer = _metric_for_label(row, config.teacher_label, "cer", reference, teacher_prediction)
    teacher_wer = _metric_for_label(row, config.teacher_label, "wer", reference, teacher_prediction)
    if base_cer < config.min_base_cer:
        return None, "base_not_hard"
    if base_cer > config.max_base_cer:
        return None, "base_too_wrong"
    if teacher_cer > config.max_teacher_cer:
        return None, "teacher_not_clean"

    target_text = _target_text(row, teacher_prediction=teacher_prediction, config=config)
    words = len(_tokens(target_text))
    if words < config.min_word_count:
        return None, "too_few_words"
    if words > config.max_word_count:
        return None, "too_many_words"
    duration = _duration_seconds(row)
    if duration is not None and duration < config.min_duration_seconds:
        return None, "too_short"
    if duration is not None and duration > config.max_duration_seconds:
        return None, "too_long"
    if config.reject_format_sensitive and _is_format_sensitive(target_text):
        return None, "format_sensitive"
    content_errors = _content_error_count(reference, base_prediction)
    if config.require_content_error and content_errors <= 0:
        return None, "no_content_error"

    selection_score = (base_cer - teacher_cer) + 0.25 * (base_wer - teacher_wer)
    if selection_score < config.min_selection_score:
        return None, "selection_score_too_low"

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return (
        {
            "audio": audio_path,
            "text": target_text,
            "reference_text": reference,
            "teacher_text": teacher_prediction,
            "base_prediction": base_prediction,
            "source_run_id": source.run_id,
            "mix_source": source.label,
            "pairwise_index": row.get("index"),
            "duration_seconds": duration,
            "base_cer": base_cer,
            "base_wer": base_wer,
            "teacher_cer": teacher_cer,
            "teacher_wer": teacher_wer,
            "selection_score": selection_score,
            "content_error_count": content_errors,
            "speaker_key": _speaker_key(row),
            **{f"metadata_{key}": value for key, value in metadata.items() if isinstance(key, str)},
        },
        "selected",
    )


def _load_pairwise(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _numeric_summary(values: list[float]) -> dict[str, Any]:
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


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "rows_by_source": dict(Counter(str(row.get("mix_source") or "") for row in rows).most_common()),
        "duration_seconds": _numeric_summary(
            [float(row["duration_seconds"]) for row in rows if row.get("duration_seconds") is not None]
        ),
        "word_count": _numeric_summary([float(len(_tokens(row.get("text")))) for row in rows]),
        "selection_score": _numeric_summary([float(row.get("selection_score") or 0.0) for row in rows]),
        "base_cer": _numeric_summary([float(row.get("base_cer") or 0.0) for row in rows]),
        "teacher_cer": _numeric_summary([float(row.get("teacher_cer") or 0.0) for row in rows]),
    }


@app.function(
    image=image,
    timeout=60 * 20,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def pairwise_hard_mine_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = HardMineConfig(
        **{
            **config_payload,
            "sources": [SourceSpec(**item) for item in config_payload["sources"]],
        }
    )
    rng = random.Random(config.seed)
    run_id = f"{config.output_name}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    source_reports = []
    reject_counts_total = Counter()
    for source in config.sources or []:
        path = ARTIFACTS_DIR / source.run_id / config.manifest_name
        source_rows = _load_pairwise(path) if path.exists() else []
        candidates = []
        reject_counts = Counter()
        for row in source_rows:
            normalized, reason = _normalize_pairwise_row(row, source=source, config=config)
            if normalized is None:
                reject_counts[reason] += 1
                reject_counts_total[reason] += 1
                continue
            candidates.append(normalized)
        candidates.sort(
            key=lambda item: (
                float(item.get("selection_score") or 0.0),
                float(item.get("content_error_count") or 0.0),
                float(item.get("base_cer") or 0.0),
            ),
            reverse=True,
        )
        if source.limit > 0:
            candidates = candidates[: source.limit]
        selected_rows.extend(candidates)
        source_reports.append(
            {
                "label": source.label,
                "run_id": source.run_id,
                "path": str(path),
                "exists": path.exists(),
                "rows": len(source_rows),
                "selected_rows": len(candidates),
                "limit": source.limit,
                "reject_counts": reject_counts.most_common(20),
            }
        )

    rng.shuffle(selected_rows)
    filtered_rows = []
    seen_texts = set()
    speaker_counts: dict[str, int] = {}
    post_filter_counts = Counter()
    for row in selected_rows:
        if config.dedupe_text:
            key = _text_key(str(row.get("text") or ""))
            if key and key in seen_texts:
                post_filter_counts["duplicate_text"] += 1
                continue
            if key:
                seen_texts.add(key)
        if config.max_samples_per_speaker > 0:
            speaker = str(row.get("speaker_key") or "")
            if speaker:
                count = speaker_counts.get(speaker, 0)
                if count >= config.max_samples_per_speaker:
                    post_filter_counts["speaker_cap"] += 1
                    continue
                speaker_counts[speaker] = count + 1
        filtered_rows.append(row)

    train_path = run_dir / "train.jsonl"
    manifest_path = run_dir / "manifest.jsonl"
    with train_path.open("w", encoding="utf-8") as train_handle, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_handle:
        for row in filtered_rows:
            manifest_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            manifest_handle.write("\n")
            train_row = dict(row)
            train_handle.write(json.dumps(train_row, ensure_ascii=False, sort_keys=True))
            train_handle.write("\n")

    report = {
        "run_id": run_id,
        "created_at_utc": _now_iso(),
        "config": asdict(config),
        "sources": source_reports,
        "reject_counts": reject_counts_total.most_common(30),
        "post_filter_counts": post_filter_counts.most_common(),
        "summary": _summarize(filtered_rows),
        "artifacts": {
            "run_dir": str(run_dir),
            "train_jsonl": str(train_path),
            "manifest_jsonl": str(manifest_path),
            "report_path": str(run_dir / "report.json"),
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return report


@app.local_entrypoint()
def main(
    sources: str,
    output_name: str = "cohere-hardmine",
    manifest_name: str = "pairwise_predictions.jsonl",
    base_label: str = "cohere_base",
    teacher_label: str = "whisper_large_v3",
    target_text_source: str = "teacher",
    seed: int = 42,
    min_base_cer: float = 0.04,
    max_base_cer: float = 0.35,
    max_teacher_cer: float = 0.02,
    min_selection_score: float = 0.03,
    min_word_count: int = 4,
    max_word_count: int = 18,
    min_duration_seconds: float = 1.5,
    max_duration_seconds: float = 8.0,
    max_samples_per_speaker: int = 8,
    dedupe_text: bool = True,
    reject_format_sensitive: bool = True,
    require_content_error: bool = True,
) -> None:
    report = pairwise_hard_mine_remote.remote(
        {
            "sources": _parse_sources(sources),
            "output_name": output_name,
            "manifest_name": manifest_name,
            "base_label": base_label,
            "teacher_label": teacher_label,
            "target_text_source": target_text_source,
            "seed": seed,
            "min_base_cer": min_base_cer,
            "max_base_cer": max_base_cer,
            "max_teacher_cer": max_teacher_cer,
            "min_selection_score": min_selection_score,
            "min_word_count": min_word_count,
            "max_word_count": max_word_count,
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "max_samples_per_speaker": max_samples_per_speaker,
            "dedupe_text": dedupe_text,
            "reject_format_sensitive": reject_format_sensitive,
            "require_content_error": require_content_error,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
