#!/usr/bin/env python3
"""Build a targeted non-Svarah hard-negative training manifest on Modal.

The input is one or more hard-mined manifests where turbo and large-v3 were
already run over the same non-Svarah source rows. This script filters for rows
that look like acoustic/content failures, drops format-only disagreements, and
balances the selected rows by source before writing train.jsonl.
"""

from __future__ import annotations

import difflib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App("localwispr-targeted-hard-negative-manifest")
image = modal.Image.debian_slim(python_version="3.11")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
CURRENCY_RE = re.compile(r"\b(rs|rupees|inr|usd|dollars?|refund|amount|balance|account|bank)\b|₹", re.IGNORECASE)

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
CONTRACTION_SUFFIXES = {"s", "ll", "re", "ve", "d", "m", "t"}

# Terms came from Svarah disagreement analysis, but this script never reads
# Svarah rows for training. It only uses the list as a source-side prioritizer.
INDIAN_CONTEXT_TERMS = {
    "aam",
    "bhature",
    "chhattisgarh",
    "chole",
    "dogri",
    "fresho",
    "gauhati",
    "guwahati",
    "jeera",
    "karnataka",
    "kollam",
    "lucknow",
    "malappuram",
    "manure",
    "nipah",
    "phonepe",
    "pooja",
    "puja",
    "saree",
    "satyagraha",
    "shillong",
    "shri",
    "sri",
    "veg",
    "yono",
    "zomato",
}


@dataclass(frozen=True)
class SourceSpec:
    path: str
    limit: int
    label: str


@dataclass(frozen=True)
class TargetedConfig:
    mix_name: str
    sources: list[SourceSpec]
    seed: int = 42
    teacher_max_cer: float = 0.02
    turbo_min_cer: float = 0.04
    turbo_max_cer: float = 0.35
    min_selection_score: float = 0.03
    min_word_count: int = 4
    max_word_count: int = 18
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 10.0
    max_samples_per_speaker: int = 40
    dedupe_text: bool = True
    reject_format_sensitive: bool = True
    use_indian_context_terms: bool = False


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _tokens(text: Any) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(str(text or ""))]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _canonical_token(token: str) -> str:
    lowered = token.lower().strip().replace("'", "")
    lowered = NUMBER_CANONICAL.get(lowered, lowered)
    lowered = SPELLING_CANONICAL.get(lowered, lowered)
    return lowered


def _is_format_pair(turbo_token: str, ref_token: str) -> bool:
    turbo = turbo_token.lower()
    ref = ref_token.lower()
    if _canonical_token(turbo) == _canonical_token(ref):
        return True
    if ORDINAL_RE.match(turbo) or ORDINAL_RE.match(ref):
        return True
    if turbo in CONTRACTION_SUFFIXES or ref in CONTRACTION_SUFFIXES:
        return True
    turbo_plain = turbo.replace("'", "")
    ref_plain = ref.replace("'", "")
    if "'" in turbo or "'" in ref:
        if turbo_plain.startswith(ref_plain) or ref_plain.startswith(turbo_plain):
            return True
        if turbo_plain.rstrip("s") == ref_plain.rstrip("s"):
            return True
    if turbo_plain.startswith(ref_plain) and ref_plain in CONTRACTION_FRAGMENTS:
        return True
    if ref_plain.startswith(turbo_plain) and turbo_plain in CONTRACTION_FRAGMENTS:
        return True
    return False


def _is_content_token(token: str) -> bool:
    plain = token.lower().replace("'", "")
    return (
        bool(token)
        and token not in STOPWORDS
        and token not in CONTRACTION_SUFFIXES
        and plain not in CONTRACTION_SUFFIXES
        and not token.isdigit()
        and not ORDINAL_RE.match(token)
        and len(token) > 1
    )


def _text_key(text: str) -> str:
    return re.sub(r"\s+", " ", " ".join(_tokens(text))).strip()


def _speaker_key(row: dict[str, Any]) -> str:
    for key in ("speaker_key", "client_id", "speaker_id", "Speaker_ID", "speaker", "user_id", "utt_spk"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{str(value).strip()}"
    return ""


def _is_format_sensitive(row: dict[str, Any], text: str) -> bool:
    return (
        _as_bool(row.get("contains_digit"))
        or _as_bool(row.get("contains_date_like"))
        or _as_bool(row.get("contains_currency_or_amount"))
        or _as_bool(row.get("contains_markup"))
        or any(character.isdigit() for character in text)
        or bool(DATE_RE.search(text))
        or bool(CURRENCY_RE.search(text))
    )


def _diff_stats(reference: str, turbo: str) -> dict[str, Any]:
    ref_tokens = _tokens(reference)
    turbo_tokens = _tokens(turbo)
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=turbo_tokens, autojunk=False)
    counts = Counter()
    content_pairs = Counter()
    tags: set[str] = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        ref_span = ref_tokens[i1:i2]
        turbo_span = turbo_tokens[j1:j2]
        counts[tag] += max(len(ref_span), len(turbo_span))

        if tag == "delete":
            content_deleted = [token for token in ref_span if _is_content_token(token)]
            counts["content_deletions"] += len(content_deleted)
            if content_deleted:
                tags.add("content_deletion")
            continue
        if tag == "insert":
            content_inserted = [token for token in turbo_span if _is_content_token(token)]
            counts["content_insertions"] += len(content_inserted)
            if content_inserted:
                tags.add("content_insertion")
            continue

        for ref_token, turbo_token in zip(ref_span, turbo_span):
            if _is_format_pair(turbo_token, ref_token):
                counts["format_pairs"] += 1
                continue
            if _is_content_token(ref_token) or _is_content_token(turbo_token):
                counts["content_substitutions"] += 1
                content_pairs[f"{turbo_token} -> {ref_token}"] += 1
                tags.add("content_substitution")
            else:
                counts["function_word_substitutions"] += 1
        if len(ref_span) != len(turbo_span):
            ref_extra = ref_span[min(len(ref_span), len(turbo_span)) :]
            turbo_extra = turbo_span[min(len(ref_span), len(turbo_span)) :]
            counts["content_deletions"] += sum(1 for token in ref_extra if _is_content_token(token))
            counts["content_insertions"] += sum(1 for token in turbo_extra if _is_content_token(token))

    if counts["delete"] > counts["replace"] + counts["insert"]:
        tags.add("mostly_deletion")
    if counts["insert"] > counts["replace"] + counts["delete"]:
        tags.add("mostly_insertion")

    content_error_count = (
        counts["content_substitutions"]
        + counts["content_deletions"]
        + counts["content_insertions"]
    )
    format_heavy = counts["format_pairs"] > 0 and counts["format_pairs"] >= content_error_count
    if format_heavy:
        tags.add("format_heavy")

    return {
        "counts": dict(counts),
        "tags": sorted(tags),
        "content_error_count": int(content_error_count),
        "format_heavy": bool(format_heavy),
        "top_content_pairs": content_pairs.most_common(5),
    }


def _row_score(row: dict[str, Any], diff: dict[str, Any], source_label: str) -> tuple[bool, float, list[str]]:
    text = str(row.get("text") or row.get("raw_transcript") or "")
    turbo_prediction = str(row.get("turbo_prediction") or "")
    teacher_prediction = str(row.get("teacher_prediction") or "")
    if not text.strip() or not turbo_prediction.strip() or not teacher_prediction.strip():
        return False, 0.0, ["missing_text_or_prediction"]

    teacher_cer = _as_float(row.get("teacher_cer"))
    turbo_cer = _as_float(row.get("turbo_cer"))
    if teacher_cer is None or turbo_cer is None:
        return False, 0.0, ["missing_metrics"]
    if teacher_cer > CONFIG.teacher_max_cer:
        return False, 0.0, ["teacher_not_clean"]
    if turbo_cer < CONFIG.turbo_min_cer:
        return False, 0.0, ["turbo_not_hard"]
    if turbo_cer > CONFIG.turbo_max_cer:
        return False, 0.0, ["turbo_too_wrong"]

    word_count = len(_tokens(text))
    if word_count < CONFIG.min_word_count:
        return False, 0.0, ["too_few_words"]
    if word_count > CONFIG.max_word_count:
        return False, 0.0, ["too_many_words"]

    duration_seconds = _as_float(row.get("duration_seconds"))
    if duration_seconds is not None:
        if duration_seconds < CONFIG.min_duration_seconds:
            return False, 0.0, ["too_short"]
        if duration_seconds > CONFIG.max_duration_seconds:
            return False, 0.0, ["too_long"]

    if CONFIG.reject_format_sensitive and _is_format_sensitive(row, text):
        return False, 0.0, ["format_sensitive_reference"]

    diff_counts = diff["counts"]
    content_error_count = int(diff.get("content_error_count") or 0)
    if content_error_count <= 0:
        return False, 0.0, ["no_content_error"]
    if diff.get("format_heavy"):
        return False, 0.0, ["format_heavy"]

    selection_score = _as_float(row.get("selection_score"))
    if selection_score is None:
        turbo_wer = _as_float(row.get("turbo_wer")) or 0.0
        teacher_wer = _as_float(row.get("teacher_wer")) or 0.0
        selection_score = (turbo_cer - teacher_cer) + 0.25 * (turbo_wer - teacher_wer)
    if selection_score < CONFIG.min_selection_score:
        return False, 0.0, ["selection_score_too_low"]

    text_tokens = set(_tokens(text))
    has_indian_context = bool(CONFIG.use_indian_context_terms and text_tokens & INDIAN_CONTEXT_TERMS)
    tags = list(diff["tags"])
    if has_indian_context:
        tags.append("indian_context_term")
    if word_count <= 8:
        tags.append("short_utterance")
    if duration_seconds is not None and duration_seconds <= 6.0:
        tags.append("short_audio")
    if "commonvoice" in source_label.lower() or "cv" in source_label.lower():
        source_bonus = 0.0
    elif "vaani" in source_label.lower():
        source_bonus = 0.025
    elif "indic" in source_label.lower():
        source_bonus = 0.02
    else:
        source_bonus = 0.015

    score = float(selection_score)
    score += min(0.12, 0.03 * content_error_count)
    score += 0.06 if has_indian_context else 0.0
    score += 0.025 if word_count <= 8 else 0.0
    score += 0.015 if duration_seconds is not None and duration_seconds <= 6.0 else 0.0
    score += source_bonus
    score -= min(0.08, 0.02 * int(diff_counts.get("format_pairs", 0)))
    return True, score, sorted(set(tags))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p50": ordered[round((len(ordered) - 1) * 0.50)],
        "p90": ordered[round((len(ordered) - 1) * 0.90)],
        "max": ordered[-1],
        "mean": mean(values),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [_as_float(row.get("duration_seconds")) for row in rows]
    durations = [value for value in durations if value is not None]
    word_counts = [float(len(_tokens(row.get("text") or ""))) for row in rows]
    turbo_cers = [_as_float(row.get("turbo_cer")) for row in rows]
    turbo_cers = [value for value in turbo_cers if value is not None]
    teacher_cers = [_as_float(row.get("teacher_cer")) for row in rows]
    teacher_cers = [value for value in teacher_cers if value is not None]
    return {
        "rows": len(rows),
        "rows_by_source": dict(Counter(str(row.get("mix_source") or "") for row in rows).most_common()),
        "tag_counts": Counter(
            tag
            for row in rows
            for tag in str(row.get("targeted_tags") or "").split(";")
            if tag
        ).most_common(30),
        "duration_seconds": _numeric_summary(durations),
        "word_count": _numeric_summary(word_counts),
        "turbo_cer": _numeric_summary(turbo_cers),
        "teacher_cer": _numeric_summary(teacher_cers),
    }


def _source_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.suffix:
        return ARTIFACTS_DIR / path_value.lstrip("/")
    return ARTIFACTS_DIR / path_value.strip("/") / "manifest.jsonl"


CONFIG: TargetedConfig


@app.function(
    image=image,
    timeout=60 * 20,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def targeted_manifest_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    global CONFIG
    CONFIG = TargetedConfig(
        mix_name=str(config_payload["mix_name"]),
        sources=[SourceSpec(**item) for item in config_payload["sources"]],
        seed=int(config_payload.get("seed", 42)),
        teacher_max_cer=float(config_payload.get("teacher_max_cer", 0.02)),
        turbo_min_cer=float(config_payload.get("turbo_min_cer", 0.04)),
        turbo_max_cer=float(config_payload.get("turbo_max_cer", 0.35)),
        min_selection_score=float(config_payload.get("min_selection_score", 0.03)),
        min_word_count=int(config_payload.get("min_word_count", 4)),
        max_word_count=int(config_payload.get("max_word_count", 18)),
        min_duration_seconds=float(config_payload.get("min_duration_seconds", 1.0)),
        max_duration_seconds=float(config_payload.get("max_duration_seconds", 10.0)),
        max_samples_per_speaker=int(config_payload.get("max_samples_per_speaker", 40)),
        dedupe_text=bool(config_payload.get("dedupe_text", True)),
        reject_format_sensitive=bool(config_payload.get("reject_format_sensitive", True)),
        use_indian_context_terms=bool(config_payload.get("use_indian_context_terms", False)),
    )
    rng = random.Random(CONFIG.seed)
    run_id = f"{CONFIG.mix_name}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    source_reports = []
    reject_counts = Counter()
    global_pair_counts = Counter()

    for source_index, source in enumerate(CONFIG.sources):
        path = _source_path(source.path)
        rows = _load_jsonl(path)
        candidates = []
        source_rejects = Counter()
        for row_index, row in enumerate(rows):
            reference = str(row.get("text") or row.get("raw_transcript") or "")
            diff = _diff_stats(reference, str(row.get("turbo_prediction") or ""))
            keep, score, tags_or_reasons = _row_score(row, diff, source.label)
            if not keep:
                for reason in tags_or_reasons:
                    reject_counts[reason] += 1
                    source_rejects[reason] += 1
                continue
            for pair, count in diff["top_content_pairs"]:
                global_pair_counts[pair] += count
            mixed_row = dict(row)
            mixed_row["mix_source"] = source.label
            mixed_row["mix_source_manifest"] = str(path)
            mixed_row["mix_source_index"] = source_index
            mixed_row["mix_source_row_index"] = row_index
            mixed_row["targeted_score"] = score
            mixed_row["targeted_tags"] = ";".join(tags_or_reasons)
            mixed_row["targeted_diff_counts"] = json.dumps(diff["counts"], sort_keys=True)
            candidates.append(mixed_row)

        candidates.sort(
            key=lambda item: (
                -float(item.get("targeted_score") or 0.0),
                str(item.get("text") or ""),
            )
        )
        if source.limit > 0:
            candidates = candidates[: source.limit]
        selected_rows.extend(candidates)
        source_reports.append(
            {
                "label": source.label,
                "path": str(path),
                "available_rows": len(rows),
                "eligible_rows": len(candidates),
                "limit": source.limit,
                "reject_counts": source_rejects.most_common(20),
            }
        )

    rng.shuffle(selected_rows)
    filtered_rows: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    speaker_counts: Counter[str] = Counter()
    post_filter_counts = Counter()
    for row in selected_rows:
        if CONFIG.dedupe_text:
            key = _text_key(str(row.get("text") or ""))
            if key and key in seen_texts:
                post_filter_counts["duplicate_text"] += 1
                continue
            if key:
                seen_texts.add(key)
        if CONFIG.max_samples_per_speaker > 0:
            speaker = _speaker_key(row)
            if speaker:
                if speaker_counts[speaker] >= CONFIG.max_samples_per_speaker:
                    post_filter_counts["speaker_cap"] += 1
                    continue
                speaker_counts[speaker] += 1
        filtered_rows.append(row)

    filtered_rows.sort(
        key=lambda item: (
            str(item.get("mix_source") or ""),
            -float(item.get("targeted_score") or 0.0),
        )
    )

    manifest_path = run_dir / "manifest.jsonl"
    train_path = run_dir / "train.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_handle, train_path.open(
        "w", encoding="utf-8"
    ) as train_handle:
        for row in filtered_rows:
            manifest_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            manifest_handle.write("\n")
            train_handle.write(
                json.dumps(
                    {
                        "audio": row["audio"],
                        "text": row["text"],
                        "source_dataset": row.get("source_dataset"),
                        "mix_source": row.get("mix_source"),
                        "dataset_index": row.get("dataset_index"),
                        "selection_score": row.get("selection_score"),
                        "targeted_score": row.get("targeted_score"),
                        "targeted_tags": row.get("targeted_tags"),
                        "turbo_cer": row.get("turbo_cer"),
                        "teacher_cer": row.get("teacher_cer"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            train_handle.write("\n")

    report = {
        "mix_run_id": run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "config": {
            "seed": CONFIG.seed,
            "teacher_max_cer": CONFIG.teacher_max_cer,
            "turbo_min_cer": CONFIG.turbo_min_cer,
            "turbo_max_cer": CONFIG.turbo_max_cer,
            "min_selection_score": CONFIG.min_selection_score,
            "min_word_count": CONFIG.min_word_count,
            "max_word_count": CONFIG.max_word_count,
            "min_duration_seconds": CONFIG.min_duration_seconds,
            "max_duration_seconds": CONFIG.max_duration_seconds,
            "max_samples_per_speaker": CONFIG.max_samples_per_speaker,
            "dedupe_text": CONFIG.dedupe_text,
            "reject_format_sensitive": CONFIG.reject_format_sensitive,
            "use_indian_context_terms": CONFIG.use_indian_context_terms,
        },
        "sources": source_reports,
        "filters": {
            "pre_balance_reject_counts": reject_counts.most_common(30),
            "post_balance_filter_counts": post_filter_counts.most_common(30),
        },
        "summary": _summarize(filtered_rows),
        "top_content_pairs": global_pair_counts.most_common(30),
        "artifacts": {
            "mix_dir": str(run_dir),
            "manifest_jsonl": str(manifest_path),
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
    teacher_max_cer: float = 0.02,
    turbo_min_cer: float = 0.04,
    turbo_max_cer: float = 0.35,
    min_selection_score: float = 0.03,
    min_word_count: int = 4,
    max_word_count: int = 18,
    min_duration_seconds: float = 1.0,
    max_duration_seconds: float = 10.0,
    max_samples_per_speaker: int = 40,
    dedupe_text: bool = True,
    reject_format_sensitive: bool = True,
    use_indian_context_terms: bool = False,
) -> None:
    report = targeted_manifest_remote.remote(
        {
            "mix_name": mix_name,
            "sources": _parse_sources(sources),
            "seed": seed,
            "teacher_max_cer": teacher_max_cer,
            "turbo_min_cer": turbo_min_cer,
            "turbo_max_cer": turbo_max_cer,
            "min_selection_score": min_selection_score,
            "min_word_count": min_word_count,
            "max_word_count": max_word_count,
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "max_samples_per_speaker": max_samples_per_speaker,
            "dedupe_text": dedupe_text,
            "reject_format_sensitive": reject_format_sensitive,
            "use_indian_context_terms": use_indian_context_terms,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
