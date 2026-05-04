#!/usr/bin/env python3
"""Fit a multi-candidate ASR router from pairwise prediction artifacts."""

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

app = modal.App("localwispr-pairwise-multi-router")
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
    "hundred",
    "thousand",
    "lakh",
    "crore",
}
DEFAULT_FEATURE_SETS = ["edit_len", "edit_len_words", "edit_len_digit", "boundary", "fine"]


@dataclass
class MultiRouterConfig:
    train_run_ids: list[str]
    candidate_labels: list[str]
    eval_run_ids: list[str] | None = None
    router_name: str = "pairwise-multi-router"
    manifest_name: str = "pairwise_predictions.jsonl"
    optimize_metric: str = "wer"
    min_counts: list[int] | None = None
    min_mean_gains: list[float] | None = None
    feature_sets: list[str] | None = None


@dataclass
class CandidatePrediction:
    label: str
    prediction: str
    wer_ops: int
    cer_ops: int
    wer_rate: float
    cer_rate: float
    feature_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class MultiRow:
    run_id: str
    index: int
    reference: str
    base_prediction: str
    ref_words: int
    ref_chars: int
    base_wer_ops: int
    base_cer_ops: int
    base_wer_rate: float
    base_cer_rate: float
    candidates: dict[str, CandidatePrediction]


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


def _candidate_feature_key(row: MultiRow, candidate: CandidatePrediction, feature_set: str) -> tuple[str, ...]:
    cached = candidate.feature_keys.get(feature_set)
    if cached is not None:
        return cached

    base_tokens = _tokens(row.base_prediction)
    candidate_tokens = _tokens(candidate.prediction)
    pred_cer = _levenshtein_distance(row.base_prediction, candidate.prediction) / max(
        len(row.base_prediction), len(candidate.prediction), 1
    )
    pred_wer = _levenshtein_distance(base_tokens, candidate_tokens) / max(
        len(base_tokens), len(candidate_tokens), 1
    )
    word_delta = len(candidate_tokens) - len(base_tokens)
    char_delta_ratio = (len(candidate.prediction) - len(row.base_prediction)) / max(
        len(row.base_prediction), 1
    )
    first_same = bool(base_tokens and candidate_tokens and base_tokens[0] == candidate_tokens[0])
    last_same = bool(base_tokens and candidate_tokens and base_tokens[-1] == candidate_tokens[-1])
    digit_state = f"bd{int(any(ch.isdigit() for ch in row.base_prediction))}_cd{int(any(ch.isdigit() for ch in candidate.prediction))}"
    numword_state = (
        f"bn{int(any(token in NUM_WORDS for token in base_tokens))}_"
        f"cn{int(any(token in NUM_WORDS for token in candidate_tokens))}"
    )

    common = (
        f"pred_cer:{_bucket_float(pred_cer, (0, 0.05, 0.12, 0.25, 0.5))}",
        f"pred_wer:{_bucket_float(pred_wer, (0, 0.12, 0.25, 0.5, 0.8))}",
        f"word_delta:{_bucket_count(word_delta, (-3, -2, -1, 0, 1, 2, 3))}",
        f"char_delta_ratio:{_bucket_float(char_delta_ratio, (-0.25, -0.12, -0.04, 0.04, 0.12, 0.25))}",
    )
    if feature_set == "edit_len":
        result = common
    elif feature_set == "edit_len_words":
        result = common + (f"base_words:{_bucket_count(len(base_tokens), (3, 6, 10, 16, 24))}",)
    elif feature_set == "edit_len_digit":
        result = common + (f"digit:{digit_state}", f"numword:{numword_state}")
    elif feature_set == "boundary":
        result = common + (
            f"first_same:{int(first_same)}",
            f"last_same:{int(last_same)}",
            f"base_words:{_bucket_count(len(base_tokens), (3, 6, 10, 16, 24))}",
        )
    elif feature_set == "fine":
        result = common + (
            f"first_same:{int(first_same)}",
            f"last_same:{int(last_same)}",
            f"base_words:{_bucket_count(len(base_tokens), (3, 6, 10, 16, 24))}",
            f"digit:{digit_state}",
            f"numword:{numword_state}",
        )
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    candidate.feature_keys[feature_set] = result
    return result


def _load_rows(
    run_ids: list[str],
    manifest_name: str,
    candidate_labels: list[str],
) -> tuple[list[MultiRow], dict[str, Any]]:
    rows: list[MultiRow] = []
    per_run: dict[str, Any] = {}
    for run_id in run_ids:
        path = ARTIFACTS_DIR / run_id / manifest_name
        run_payload = {
            "path": str(path),
            "exists": path.exists(),
            "rows": 0,
            "usable_rows": 0,
            "missing_by_label": Counter(),
        }
        per_run[run_id] = run_payload
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                run_payload["rows"] += 1
                reference = str(payload.get("normalized_reference") or payload.get("reference") or "")
                base_prediction = str(payload.get("base_normalized_prediction") or payload.get("base_prediction") or "")
                ref_tokens = _tokens(reference)
                ref_words = len(ref_tokens)
                ref_chars = len(reference)
                if not ref_words or not ref_chars:
                    continue
                base_wer_ops = _levenshtein_distance(ref_tokens, _tokens(base_prediction))
                base_cer_ops = _levenshtein_distance(reference, base_prediction)
                adapters = payload.get("adapters") if isinstance(payload.get("adapters"), dict) else {}
                candidates = {}
                for label in candidate_labels:
                    candidate_payload = adapters.get(label)
                    if not isinstance(candidate_payload, dict):
                        run_payload["missing_by_label"][label] += 1
                        continue
                    prediction = str(
                        candidate_payload.get("normalized_prediction")
                        or candidate_payload.get("prediction")
                        or ""
                    )
                    wer_ops = _levenshtein_distance(ref_tokens, _tokens(prediction))
                    cer_ops = _levenshtein_distance(reference, prediction)
                    candidates[label] = CandidatePrediction(
                        label=label,
                        prediction=prediction,
                        wer_ops=wer_ops,
                        cer_ops=cer_ops,
                        wer_rate=wer_ops / ref_words,
                        cer_rate=cer_ops / ref_chars,
                    )
                if not candidates:
                    continue
                rows.append(
                    MultiRow(
                        run_id=run_id,
                        index=int(payload.get("index") or run_payload["rows"] - 1),
                        reference=reference,
                        base_prediction=base_prediction,
                        ref_words=ref_words,
                        ref_chars=ref_chars,
                        base_wer_ops=base_wer_ops,
                        base_cer_ops=base_cer_ops,
                        base_wer_rate=base_wer_ops / ref_words,
                        base_cer_rate=base_cer_ops / ref_chars,
                        candidates=candidates,
                    )
                )
                run_payload["usable_rows"] += 1
        run_payload["missing_by_label"] = dict(run_payload["missing_by_label"])
    return rows, per_run


def _precompute(rows: list[MultiRow], feature_sets: list[str]) -> None:
    for row in rows:
        for candidate in row.candidates.values():
            for feature_set in feature_sets:
                _candidate_feature_key(row, candidate, feature_set)


def _compute_metrics(rows: list[MultiRow], chooser: Any) -> dict[str, Any]:
    totals = Counter()
    choice_counts = Counter()
    for row in rows:
        label = chooser(row)
        candidate = row.candidates.get(label) if label else None
        if candidate is None:
            choice_counts["base"] += 1
            totals["wer_ops"] += row.base_wer_ops
            totals["cer_ops"] += row.base_cer_ops
        else:
            choice_counts[candidate.label] += 1
            totals["wer_ops"] += candidate.wer_ops
            totals["cer_ops"] += candidate.cer_ops
        totals["ref_words"] += row.ref_words
        totals["ref_chars"] += row.ref_chars
    samples = len(rows)
    return {
        "samples": samples,
        "wer": totals["wer_ops"] / max(1, totals["ref_words"]),
        "cer": totals["cer_ops"] / max(1, totals["ref_chars"]),
        "choice_counts": dict(choice_counts),
        "adapter_pick_rate": (samples - choice_counts["base"]) / max(1, samples),
    }


def _candidate_delta(row: MultiRow, candidate: CandidatePrediction, metric: str) -> float:
    if metric == "wer":
        return candidate.wer_rate - row.base_wer_rate
    if metric == "cer":
        return candidate.cer_rate - row.base_cer_rate
    raise ValueError("optimize_metric must be wer or cer")


def _fit_policy(rows: list[MultiRow], policy_config: dict[str, Any]) -> dict[str, Any]:
    kind = policy_config["kind"]
    if kind in {"base", "fixed"}:
        return dict(policy_config)

    feature_set = str(policy_config["feature_set"])
    min_count = int(policy_config["min_count"])
    min_mean_gain = float(policy_config["min_mean_gain"])
    metric = str(policy_config["optimize_metric"])
    grouped: dict[tuple[str, tuple[str, ...]], list[float]] = defaultdict(list)
    for row in rows:
        for label, candidate in row.candidates.items():
            if candidate.prediction == row.base_prediction:
                continue
            grouped[(label, _candidate_feature_key(row, candidate, feature_set))].append(
                _candidate_delta(row, candidate, metric)
            )

    selected_scores = {}
    for key, deltas in grouped.items():
        if len(deltas) < min_count:
            continue
        score = mean(deltas)
        if score <= -min_mean_gain:
            selected_scores[key] = score

    return {
        **policy_config,
        "selected_scores": [
            {"label": label, "feature_key": list(feature_key), "mean_delta": score}
            for (label, feature_key), score in sorted(selected_scores.items(), key=lambda item: item[1])
        ],
    }


def _chooser(policy: dict[str, Any]) -> Any:
    kind = policy["kind"]
    if kind == "base":
        return lambda row: None
    if kind == "fixed":
        label = str(policy["label"])
        return lambda row: label if label in row.candidates else None
    if kind == "bin_multirouter":
        feature_set = str(policy["feature_set"])
        scores = {
            (str(item["label"]), tuple(item["feature_key"])): float(item["mean_delta"])
            for item in policy.get("selected_scores", [])
            if isinstance(item, dict)
        }

        def choose(row: MultiRow) -> str | None:
            best_label = None
            best_score = 0.0
            for label, candidate in row.candidates.items():
                score = scores.get((label, _candidate_feature_key(row, candidate, feature_set)))
                if score is not None and score < best_score:
                    best_score = score
                    best_label = label
            return best_label

        return choose
    raise ValueError(f"Unknown policy kind: {kind}")


def _policy_configs(config: MultiRouterConfig) -> list[dict[str, Any]]:
    min_counts = config.min_counts or [2, 3, 5, 8, 12, 20]
    min_mean_gains = config.min_mean_gains or [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    feature_sets = config.feature_sets or DEFAULT_FEATURE_SETS
    policies = [{"kind": "base", "optimize_metric": config.optimize_metric}]
    policies.extend(
        {"kind": "fixed", "label": label, "optimize_metric": config.optimize_metric}
        for label in config.candidate_labels
    )
    for feature_set in feature_sets:
        for min_count in min_counts:
            for min_mean_gain in min_mean_gains:
                policies.append(
                    {
                        "kind": "bin_multirouter",
                        "feature_set": feature_set,
                        "min_count": min_count,
                        "min_mean_gain": min_mean_gain,
                        "optimize_metric": config.optimize_metric,
                    }
                )
    return policies


def _loo(rows: list[MultiRow], policy_config: dict[str, Any]) -> dict[str, Any]:
    choices: dict[tuple[str, int], str | None] = {}
    per_run = {}
    for heldout_run_id in sorted({row.run_id for row in rows}):
        train_rows = [row for row in rows if row.run_id != heldout_run_id]
        heldout_rows = [row for row in rows if row.run_id == heldout_run_id]
        policy = _fit_policy(train_rows, policy_config)
        choose = _chooser(policy)
        for row in heldout_rows:
            choices[(row.run_id, row.index)] = choose(row)
        per_run[heldout_run_id] = _compute_metrics(heldout_rows, choose)
    combined = _compute_metrics(rows, lambda row: choices.get((row.run_id, row.index)))
    return {"combined": combined, "per_run": per_run}


def _score(metrics: dict[str, Any], optimize_metric: str) -> tuple[float, float, float]:
    other = "cer" if optimize_metric == "wer" else "wer"
    return (
        float(metrics[optimize_metric]),
        float(metrics[other]),
        float(metrics.get("adapter_pick_rate") or 0.0),
    )


def _policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in policy.items() if key != "selected_scores"}
    selected_scores = policy.get("selected_scores")
    if isinstance(selected_scores, list):
        payload["selected_score_count"] = len(selected_scores)
        payload["selected_scores_preview"] = selected_scores[:20]
    return payload


@app.function(
    image=image,
    timeout=60 * 15,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def multi_router_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = MultiRouterConfig(**config_payload)
    if config.optimize_metric not in {"wer", "cer"}:
        raise ValueError("optimize_metric must be wer or cer")
    feature_sets = config.feature_sets or DEFAULT_FEATURE_SETS
    train_rows, train_per_run = _load_rows(config.train_run_ids, config.manifest_name, config.candidate_labels)
    eval_rows, eval_per_run = _load_rows(config.eval_run_ids or [], config.manifest_name, config.candidate_labels)
    if not train_rows:
        raise ValueError("No usable train rows loaded")
    _precompute(train_rows + eval_rows, feature_sets)

    base_train = _compute_metrics(train_rows, lambda row: None)
    policy_results = []
    for policy_config in _policy_configs(config):
        oof = _loo(train_rows, policy_config)
        policy_results.append({"policy_config": policy_config, "oof": oof, "score": _score(oof["combined"], config.optimize_metric)})
    policy_results.sort(key=lambda item: item["score"])
    final_policy = _fit_policy(train_rows, policy_results[0]["policy_config"])
    final_choose = _chooser(final_policy)

    fixed_train = {
        label: _compute_metrics(train_rows, lambda row, label=label: label if label in row.candidates else None)
        for label in config.candidate_labels
    }
    report: dict[str, Any] = {
        "run_id": f"{config.router_name}-{_now_utc()}",
        "config": {
            "train_run_ids": config.train_run_ids,
            "eval_run_ids": config.eval_run_ids or [],
            "candidate_labels": config.candidate_labels,
            "manifest_name": config.manifest_name,
            "optimize_metric": config.optimize_metric,
        },
        "loaded": {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_per_run": train_per_run,
            "eval_per_run": eval_per_run,
        },
        "selected_policy": _policy_summary(final_policy),
        "train_oof": {
            "base": base_train,
            "fixed_candidates": fixed_train,
            "router": policy_results[0]["oof"]["combined"],
            "router_delta_vs_base": {
                "wer": policy_results[0]["oof"]["combined"]["wer"] - base_train["wer"],
                "cer": policy_results[0]["oof"]["combined"]["cer"] - base_train["cer"],
            },
            "per_run": policy_results[0]["oof"]["per_run"],
        },
        "train_refit": {
            "router": _compute_metrics(train_rows, final_choose),
        },
        "top_policy_candidates": [
            {
                "policy": _policy_summary(_fit_policy(train_rows, item["policy_config"])),
                "oof": item["oof"]["combined"],
            }
            for item in policy_results[:10]
        ],
    }

    if eval_rows:
        eval_base = _compute_metrics(eval_rows, lambda row: None)
        eval_router = _compute_metrics(eval_rows, final_choose)
        report["eval"] = {
            "base": eval_base,
            "fixed_candidates": {
                label: _compute_metrics(eval_rows, lambda row, label=label: label if label in row.candidates else None)
                for label in config.candidate_labels
            },
            "router": eval_router,
            "router_delta_vs_base": {
                "wer": eval_router["wer"] - eval_base["wer"],
                "cer": eval_router["cer"] - eval_base["cer"],
            },
        }

    run_dir = ARTIFACTS_DIR / report["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report["artifacts"] = {"report_path": str(report_path)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return report


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
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
    candidate_labels: str,
    eval_run_ids: str | None = None,
    router_name: str = "pairwise-multi-router",
    manifest_name: str = "pairwise_predictions.jsonl",
    optimize_metric: str = "wer",
    min_counts: str | None = None,
    min_mean_gains: str | None = None,
    feature_sets: str | None = None,
) -> None:
    report = multi_router_remote.remote(
        {
            "train_run_ids": _parse_csv(train_run_ids),
            "candidate_labels": _parse_csv(candidate_labels),
            "eval_run_ids": _parse_csv(eval_run_ids),
            "router_name": router_name,
            "manifest_name": manifest_name,
            "optimize_metric": optimize_metric,
            "min_counts": _parse_int_csv(min_counts),
            "min_mean_gains": _parse_float_csv(min_mean_gains),
            "feature_sets": _parse_csv(feature_sets) if feature_sets else None,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
