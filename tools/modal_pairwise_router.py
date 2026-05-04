#!/usr/bin/env python3
"""Fit and evaluate a simple pairwise ASR router on Modal artifacts.

The router chooses between the base prediction and one adapter prediction using
only prediction-text features. It trains on non-Svarah pairwise artifacts and
can then evaluate the frozen rule on held-out artifacts such as Svarah.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App("localwispr-pairwise-router")
image = modal.Image.debian_slim(python_version="3.11")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
NUM_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "lakh",
    "crore",
}
DEFAULT_FEATURE_SETS = ["edit_len", "edit_len_words", "edit_len_digit", "boundary", "fine"]


@dataclass
class RouterConfig:
    train_run_ids: list[str]
    eval_run_ids: list[str]
    candidate_label: str
    router_name: str = "pairwise-router"
    manifest_name: str = "pairwise_predictions.jsonl"
    optimize_metric: str = "wer"
    min_counts: list[int] | None = None
    min_mean_gains: list[float] | None = None
    feature_sets: list[str] | None = None


@dataclass
class PairwiseRow:
    run_id: str
    index: int
    reference: str
    base_prediction: str
    candidate_prediction: str
    ref_words: int
    ref_chars: int
    base_wer_ops: int
    candidate_wer_ops: int
    base_cer_ops: int
    candidate_cer_ops: int
    base_wer_rate: float
    candidate_wer_rate: float
    base_cer_rate: float
    candidate_cer_rate: float
    feature_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


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


def _bucket_count(value: int, cuts: tuple[int, ...]) -> str:
    for cut in cuts:
        if value <= cut:
            return f"<= {cut}"
    return f"> {cuts[-1]}"


def _bucket_float(value: float, cuts: tuple[float, ...]) -> str:
    for cut in cuts:
        if value <= cut:
            return f"<= {cut:g}"
    return f"> {cuts[-1]:g}"


def _digit_state(base_prediction: str, candidate_prediction: str) -> str:
    return f"bd{int(any(ch.isdigit() for ch in base_prediction))}_cd{int(any(ch.isdigit() for ch in candidate_prediction))}"


def _numword_state(base_tokens: list[str], candidate_tokens: list[str]) -> str:
    return f"bn{int(any(token in NUM_WORDS for token in base_tokens))}_cn{int(any(token in NUM_WORDS for token in candidate_tokens))}"


def _feature_key(row: PairwiseRow, feature_set: str) -> tuple[str, ...]:
    cached_key = row.feature_keys.get(feature_set)
    if cached_key is not None:
        return cached_key

    base_tokens = _tokens(row.base_prediction)
    candidate_tokens = _tokens(row.candidate_prediction)
    base_chars = row.base_prediction or ""
    candidate_chars = row.candidate_prediction or ""
    pred_cer = _levenshtein_distance(base_chars, candidate_chars) / max(len(base_chars), len(candidate_chars), 1)
    pred_wer = _levenshtein_distance(base_tokens, candidate_tokens) / max(len(base_tokens), len(candidate_tokens), 1)
    word_delta = len(candidate_tokens) - len(base_tokens)
    char_delta = len(candidate_chars) - len(base_chars)
    char_delta_ratio = char_delta / max(len(base_chars), 1)
    first_same = bool(base_tokens and candidate_tokens and base_tokens[0] == candidate_tokens[0])
    last_same = bool(base_tokens and candidate_tokens and base_tokens[-1] == candidate_tokens[-1])

    common = (
        f"pred_cer:{_bucket_float(pred_cer, (0, 0.05, 0.12, 0.25, 0.5))}",
        f"pred_wer:{_bucket_float(pred_wer, (0, 0.12, 0.25, 0.5, 0.8))}",
        f"word_delta:{_bucket_count(word_delta, (-3, -2, -1, 0, 1, 2, 3))}",
        f"char_delta_ratio:{_bucket_float(char_delta_ratio, (-0.25, -0.12, -0.04, 0.04, 0.12, 0.25))}",
    )
    if feature_set == "edit_len":
        return common
    if feature_set == "edit_len_words":
        return common + (f"base_words:{_bucket_count(len(base_tokens), (3, 6, 10, 16, 24))}",)
    if feature_set == "edit_len_digit":
        return common + (
            f"digit:{_digit_state(row.base_prediction, row.candidate_prediction)}",
            f"numword:{_numword_state(base_tokens, candidate_tokens)}",
        )
    if feature_set == "boundary":
        return common + (
            f"first_same:{int(first_same)}",
            f"last_same:{int(last_same)}",
            f"base_words:{_bucket_count(len(base_tokens), (3, 6, 10, 16, 24))}",
        )
    if feature_set == "fine":
        return common + (
            f"first_same:{int(first_same)}",
            f"last_same:{int(last_same)}",
            f"base_words:{_bucket_count(len(base_tokens), (3, 6, 10, 16, 24))}",
            f"digit:{_digit_state(row.base_prediction, row.candidate_prediction)}",
            f"numword:{_numword_state(base_tokens, candidate_tokens)}",
        )
    raise ValueError(f"Unknown feature set: {feature_set}")


def _precompute_feature_keys(rows: list[PairwiseRow], feature_sets: list[str]) -> None:
    for row in rows:
        for feature_set in feature_sets:
            row.feature_keys[feature_set] = _feature_key(row, feature_set)


def _load_rows(run_ids: list[str], manifest_name: str, candidate_label: str) -> tuple[list[PairwiseRow], dict[str, Any]]:
    rows: list[PairwiseRow] = []
    per_run: dict[str, Any] = {}
    for run_id in run_ids:
        path = ARTIFACTS_DIR / run_id / manifest_name
        run_payload = {"path": str(path), "exists": path.exists(), "rows": 0, "usable_rows": 0, "missing_candidate": 0}
        per_run[run_id] = run_payload
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                run_payload["rows"] += 1
                adapters = payload.get("adapters") if isinstance(payload.get("adapters"), dict) else {}
                candidate_payload = adapters.get(candidate_label)
                if not isinstance(candidate_payload, dict):
                    run_payload["missing_candidate"] += 1
                    continue
                reference = str(payload.get("normalized_reference") or payload.get("reference") or "")
                base_prediction = str(payload.get("base_normalized_prediction") or payload.get("base_prediction") or "")
                candidate_prediction = str(
                    candidate_payload.get("normalized_prediction") or candidate_payload.get("prediction") or ""
                )
                ref_tokens = _tokens(reference)
                ref_words = len(ref_tokens)
                ref_chars = len(reference)
                if ref_words == 0 or ref_chars == 0:
                    continue
                base_wer_ops = _levenshtein_distance(ref_tokens, _tokens(base_prediction))
                candidate_wer_ops = _levenshtein_distance(ref_tokens, _tokens(candidate_prediction))
                base_cer_ops = _levenshtein_distance(reference, base_prediction)
                candidate_cer_ops = _levenshtein_distance(reference, candidate_prediction)
                rows.append(
                    PairwiseRow(
                        run_id=run_id,
                        index=int(payload.get("index") or run_payload["rows"] - 1),
                        reference=reference,
                        base_prediction=base_prediction,
                        candidate_prediction=candidate_prediction,
                        ref_words=ref_words,
                        ref_chars=ref_chars,
                        base_wer_ops=base_wer_ops,
                        candidate_wer_ops=candidate_wer_ops,
                        base_cer_ops=base_cer_ops,
                        candidate_cer_ops=candidate_cer_ops,
                        base_wer_rate=base_wer_ops / ref_words,
                        candidate_wer_rate=candidate_wer_ops / ref_words,
                        base_cer_rate=base_cer_ops / ref_chars,
                        candidate_cer_rate=candidate_cer_ops / ref_chars,
                    )
                )
                run_payload["usable_rows"] += 1
    return rows, per_run


def _compute_metrics(rows: list[PairwiseRow], chooser: Any) -> dict[str, Any]:
    totals = Counter()
    choice_counts = Counter()
    row_counts = Counter()
    for row in rows:
        choose_candidate = bool(chooser(row))
        choice_counts["candidate" if choose_candidate else "base"] += 1
        row_counts["improved"] += int(row.candidate_cer_ops < row.base_cer_ops)
        row_counts["worsened"] += int(row.candidate_cer_ops > row.base_cer_ops)
        row_counts["unchanged"] += int(row.candidate_cer_ops == row.base_cer_ops)
        totals["wer_ops"] += row.candidate_wer_ops if choose_candidate else row.base_wer_ops
        totals["cer_ops"] += row.candidate_cer_ops if choose_candidate else row.base_cer_ops
        totals["ref_words"] += row.ref_words
        totals["ref_chars"] += row.ref_chars
    samples = len(rows)
    return {
        "samples": samples,
        "wer": totals["wer_ops"] / max(1, totals["ref_words"]),
        "cer": totals["cer_ops"] / max(1, totals["ref_chars"]),
        "choice_counts": dict(choice_counts),
        "candidate_pick_rate": choice_counts["candidate"] / max(1, samples),
        "pairwise_candidate_vs_base_counts": dict(row_counts),
    }


def _metric_delta(row: PairwiseRow, metric: str) -> float:
    if metric == "cer":
        return row.candidate_cer_rate - row.base_cer_rate
    if metric == "wer":
        return row.candidate_wer_rate - row.base_wer_rate
    raise ValueError("optimize_metric must be wer or cer")


def _fit_bins(
    rows: list[PairwiseRow],
    *,
    feature_set: str,
    min_count: int,
    min_mean_gain: float,
    optimize_metric: str,
) -> set[tuple[str, ...]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.base_prediction == row.candidate_prediction:
            continue
        grouped[_feature_key(row, feature_set)].append(_metric_delta(row, optimize_metric))
    selected = set()
    for key, deltas in grouped.items():
        if len(deltas) < min_count:
            continue
        if mean(deltas) <= -min_mean_gain:
            selected.add(key)
    return selected


def _chooser_for_policy(policy: dict[str, Any]) -> Any:
    kind = policy["kind"]
    if kind == "never_adapter":
        return lambda row: False
    if kind == "all_adapter":
        return lambda row: True
    if kind == "bin_router":
        feature_set = str(policy["feature_set"])
        selected_bins = {tuple(item) for item in policy["selected_bins"]}
        return lambda row: _feature_key(row, feature_set) in selected_bins
    raise ValueError(f"Unknown policy kind: {kind}")


def _fit_policy(rows: list[PairwiseRow], config: dict[str, Any]) -> dict[str, Any]:
    kind = config["kind"]
    if kind in {"never_adapter", "all_adapter"}:
        return dict(config)
    selected_bins = _fit_bins(
        rows,
        feature_set=str(config["feature_set"]),
        min_count=int(config["min_count"]),
        min_mean_gain=float(config["min_mean_gain"]),
        optimize_metric=str(config["optimize_metric"]),
    )
    return {
        **config,
        "selected_bins": [list(item) for item in sorted(selected_bins)],
        "selected_bin_count": len(selected_bins),
    }


def _candidate_policy_configs(config: RouterConfig) -> list[dict[str, Any]]:
    min_counts = config.min_counts or [2, 3, 5, 8, 12, 20]
    min_mean_gains = config.min_mean_gains or [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    feature_sets = config.feature_sets or DEFAULT_FEATURE_SETS
    policies = [
        {"kind": "never_adapter", "optimize_metric": config.optimize_metric},
        {"kind": "all_adapter", "optimize_metric": config.optimize_metric},
    ]
    for feature_set in feature_sets:
        for min_count in min_counts:
            for min_mean_gain in min_mean_gains:
                policies.append(
                    {
                        "kind": "bin_router",
                        "feature_set": feature_set,
                        "min_count": min_count,
                        "min_mean_gain": min_mean_gain,
                        "optimize_metric": config.optimize_metric,
                    }
                )
    return policies


def _evaluate_leave_one_run_out(rows: list[PairwiseRow], policy_config: dict[str, Any]) -> dict[str, Any]:
    run_ids = sorted({row.run_id for row in rows})
    choices: dict[tuple[str, int], bool] = {}
    per_run = {}
    for heldout_run_id in run_ids:
        train_rows = [row for row in rows if row.run_id != heldout_run_id]
        heldout_rows = [row for row in rows if row.run_id == heldout_run_id]
        policy = _fit_policy(train_rows, policy_config)
        chooser = _chooser_for_policy(policy)
        for row in heldout_rows:
            choices[(row.run_id, row.index)] = bool(chooser(row))
        per_run[heldout_run_id] = _compute_metrics(heldout_rows, chooser)

    chooser = lambda row: choices.get((row.run_id, row.index), False)
    combined = _compute_metrics(rows, chooser)
    return {"combined": combined, "per_run": per_run}


def _policy_score(metrics: dict[str, Any], optimize_metric: str) -> tuple[float, float, float]:
    other_metric = "cer" if optimize_metric == "wer" else "wer"
    return (
        float(metrics[optimize_metric]),
        float(metrics[other_metric]),
        float(metrics.get("candidate_pick_rate") or 0.0),
    )


def _summarize_policy(policy: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in policy.items() if key != "selected_bins"}
    selected_bins = policy.get("selected_bins")
    if isinstance(selected_bins, list):
        payload["selected_bin_count"] = len(selected_bins)
        payload["selected_bins_preview"] = selected_bins[:20]
    return payload


@app.function(
    image=image,
    timeout=60 * 10,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def router_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = RouterConfig(**config_payload)
    if config.optimize_metric not in {"wer", "cer"}:
        raise ValueError("optimize_metric must be wer or cer")

    train_rows, train_per_run = _load_rows(config.train_run_ids, config.manifest_name, config.candidate_label)
    eval_rows, eval_per_run = _load_rows(config.eval_run_ids, config.manifest_name, config.candidate_label)
    if not train_rows:
        raise ValueError("No usable train rows loaded")
    if not eval_rows:
        raise ValueError("No usable eval rows loaded")
    _precompute_feature_keys(train_rows + eval_rows, config.feature_sets or DEFAULT_FEATURE_SETS)

    base_train = _compute_metrics(train_rows, lambda row: False)
    candidate_train = _compute_metrics(train_rows, lambda row: True)
    policy_results = []
    for policy_config in _candidate_policy_configs(config):
        oof = _evaluate_leave_one_run_out(train_rows, policy_config)
        policy_results.append(
            {
                "policy_config": policy_config,
                "oof": oof,
                "score": _policy_score(oof["combined"], config.optimize_metric),
            }
        )
    policy_results.sort(key=lambda item: item["score"])
    best_policy_config = dict(policy_results[0]["policy_config"])
    final_policy = _fit_policy(train_rows, best_policy_config)
    final_chooser = _chooser_for_policy(final_policy)

    train_router = _compute_metrics(train_rows, final_chooser)
    eval_router = _compute_metrics(eval_rows, final_chooser)
    eval_base = _compute_metrics(eval_rows, lambda row: False)
    eval_candidate = _compute_metrics(eval_rows, lambda row: True)

    per_eval_run = {}
    for run_id in config.eval_run_ids:
        run_rows = [row for row in eval_rows if row.run_id == run_id]
        if not run_rows:
            continue
        per_eval_run[run_id] = {
            "base": _compute_metrics(run_rows, lambda row: False),
            "candidate": _compute_metrics(run_rows, lambda row: True),
            "router": _compute_metrics(run_rows, final_chooser),
        }

    report = {
        "run_id": f"{config.router_name}-{_now_utc()}",
        "config": {
            "train_run_ids": config.train_run_ids,
            "eval_run_ids": config.eval_run_ids,
            "candidate_label": config.candidate_label,
            "manifest_name": config.manifest_name,
            "optimize_metric": config.optimize_metric,
        },
        "loaded": {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_per_run": train_per_run,
            "eval_per_run": eval_per_run,
        },
        "selected_policy": _summarize_policy(final_policy),
        "train_oof": {
            "base": base_train,
            "candidate": candidate_train,
            "router": policy_results[0]["oof"]["combined"],
            "router_delta_vs_base": {
                "wer": policy_results[0]["oof"]["combined"]["wer"] - base_train["wer"],
                "cer": policy_results[0]["oof"]["combined"]["cer"] - base_train["cer"],
            },
            "router_delta_vs_candidate": {
                "wer": policy_results[0]["oof"]["combined"]["wer"] - candidate_train["wer"],
                "cer": policy_results[0]["oof"]["combined"]["cer"] - candidate_train["cer"],
            },
            "per_run": policy_results[0]["oof"]["per_run"],
        },
        "train_refit": {
            "router": train_router,
            "router_delta_vs_base": {
                "wer": train_router["wer"] - base_train["wer"],
                "cer": train_router["cer"] - base_train["cer"],
            },
            "router_delta_vs_candidate": {
                "wer": train_router["wer"] - candidate_train["wer"],
                "cer": train_router["cer"] - candidate_train["cer"],
            },
        },
        "eval": {
            "base": eval_base,
            "candidate": eval_candidate,
            "router": eval_router,
            "router_delta_vs_base": {
                "wer": eval_router["wer"] - eval_base["wer"],
                "cer": eval_router["cer"] - eval_base["cer"],
            },
            "router_delta_vs_candidate": {
                "wer": eval_router["wer"] - eval_candidate["wer"],
                "cer": eval_router["cer"] - eval_candidate["cer"],
            },
            "per_run": per_eval_run,
        },
        "top_policy_candidates": [
            {
                "policy": _summarize_policy(_fit_policy(train_rows, item["policy_config"])),
                "oof": item["oof"]["combined"],
            }
            for item in policy_results[:10]
        ],
    }

    run_dir = ARTIFACTS_DIR / report["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report["artifacts"] = {"report_path": str(report_path)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return report


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_csv(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


@app.local_entrypoint()
def main(
    train_run_ids: str,
    eval_run_ids: str,
    candidate_label: str,
    router_name: str = "pairwise-router",
    manifest_name: str = "pairwise_predictions.jsonl",
    optimize_metric: str = "wer",
    min_counts: str | None = None,
    min_mean_gains: str | None = None,
    feature_sets: str | None = None,
) -> None:
    report = router_remote.remote(
        {
            "train_run_ids": _parse_csv(train_run_ids),
            "eval_run_ids": _parse_csv(eval_run_ids),
            "candidate_label": candidate_label,
            "router_name": router_name,
            "manifest_name": manifest_name,
            "optimize_metric": optimize_metric,
            "min_counts": _parse_int_csv(min_counts),
            "min_mean_gains": _parse_float_csv(min_mean_gains),
            "feature_sets": _parse_csv(feature_sets) if feature_sets else None,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
