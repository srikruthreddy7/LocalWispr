#!/usr/bin/env python3
"""Modal-side safe summaries for artifact reports.

This keeps gated report artifacts inside the Modal artifacts volume and only
prints aggregate metrics locally. It intentionally omits references,
predictions, pairwise examples, and group rows that may include transcript text.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal


ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
ARTIFACTS_DIR = Path("/artifacts")

app = modal.App("localwispr-artifact-report-summary")
image = modal.Image.debian_slim(python_version="3.11")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)


@dataclass
class ReportSummaryConfig:
    run_id: str
    report_name: str = "report.json"
    include_progress: bool = True


def _safe_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": candidate.get("label"),
        "model_role": candidate.get("model_role"),
        "base_model": candidate.get("base_model"),
        "adapter_run_id": candidate.get("adapter_run_id"),
        "adapter_scale": candidate.get("adapter_scale"),
    }


def _safe_pairwise_counts(report: dict[str, Any]) -> dict[str, Any]:
    pairwise = report.get("pairwise_vs_baseline") or report.get("pairwise_vs_base")
    if not isinstance(pairwise, dict):
        return {}
    return {
        str(label): payload.get("counts_vs_base")
        for label, payload in pairwise.items()
        if isinstance(payload, dict) and isinstance(payload.get("counts_vs_base"), dict)
    }


def _safe_oracle(report: dict[str, Any]) -> dict[str, Any]:
    oracle = report.get("oracle")
    if not isinstance(oracle, dict):
        return {}
    return {
        "metrics": oracle.get("metrics"),
        "delta_vs_baseline": oracle.get("delta_vs_baseline"),
        "choice_counts": oracle.get("choice_counts"),
    }


def _safe_progress(progress: dict[str, Any]) -> dict[str, Any]:
    eval_progress = progress.get("eval")
    if not isinstance(eval_progress, dict):
        return progress

    phases = {}
    raw_phases = eval_progress.get("phases")
    if isinstance(raw_phases, dict):
        for phase_key, phase in raw_phases.items():
            if not isinstance(phase, dict):
                continue
            candidate = phase.get("candidate")
            phases[str(phase_key)] = {
                "phase_name": phase.get("phase_name") or phase.get("phase"),
                "candidate": _safe_candidate(candidate) if isinstance(candidate, dict) else None,
                "status": phase.get("status"),
                "gpu_index": phase.get("gpu_index"),
                "samples_done": phase.get("samples_done"),
                "samples_total": phase.get("samples_total"),
                "batches_done": phase.get("batches_done"),
                "batches_total": phase.get("batches_total"),
                "updated_at_utc": phase.get("updated_at_utc"),
            }

    return {
        "stage": progress.get("stage"),
        "status": progress.get("status"),
        "updated_at_utc": progress.get("updated_at_utc"),
        "eval": {
            "samples_done": eval_progress.get("samples_done"),
            "samples_total": eval_progress.get("samples_total"),
            "percent_complete": eval_progress.get("percent_complete"),
            "phases": phases,
        },
    }


def _safe_phase_progress(phase: dict[str, Any]) -> dict[str, Any]:
    candidate = phase.get("candidate")
    return {
        "phase": phase.get("phase") or phase.get("phase_name"),
        "candidate": _safe_candidate(candidate) if isinstance(candidate, dict) else None,
        "status": phase.get("status"),
        "gpu_index": phase.get("gpu_index"),
        "samples_done": phase.get("samples_done"),
        "samples_total": phase.get("samples_total"),
        "batches_done": phase.get("batches_done"),
        "batches_total": phase.get("batches_total"),
        "updated_at_utc": phase.get("updated_at_utc"),
    }


def _safe_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    candidates = report.get("candidates")
    return {
        "created_at_utc": report.get("created_at_utc"),
        "run_id": report.get("sweep_run_id") or report.get("eval_run_id") or report.get("run_id"),
        "source_run_id": report.get("source_run_id"),
        "runtime": report.get("runtime"),
        "dataset": report.get("dataset"),
        "baseline_label": report.get("baseline_label"),
        "candidates": [
            _safe_candidate(candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        if isinstance(candidates, list)
        else None,
        "overall_metrics": report.get("overall_metrics"),
        "deltas_vs_baseline": report.get("deltas_vs_baseline") or report.get("deltas_vs_base"),
        "pairwise_counts_vs_baseline": _safe_pairwise_counts(report),
        "oracle": _safe_oracle(report),
        "artifacts": {
            "report_path": (report.get("artifacts") or {}).get("report_path")
            if isinstance(report.get("artifacts"), dict)
            else None,
            "progress_path": (report.get("artifacts") or {}).get("progress_path")
            if isinstance(report.get("artifacts"), dict)
            else None,
        },
    }


@app.function(
    image=image,
    timeout=60 * 5,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def report_summary_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = ReportSummaryConfig(**config_payload)
    run_dir = ARTIFACTS_DIR / config.run_id
    report_path = run_dir / config.report_name
    progress_path = run_dir / "progress.json"

    payload: dict[str, Any] = {
        "requested_run_id": config.run_id,
        "report_path": str(report_path),
        "report_exists": report_path.exists(),
    }
    if report_path.exists():
        payload["report"] = _safe_report_summary(
            json.loads(report_path.read_text(encoding="utf-8"))
        )
    if config.include_progress:
        payload["progress_path"] = str(progress_path)
        payload["progress_exists"] = progress_path.exists()
        if progress_path.exists():
            payload["progress"] = _safe_progress(
                json.loads(progress_path.read_text(encoding="utf-8"))
            )
        progress_dir = run_dir / "progress"
        if progress_dir.exists():
            phase_progress = {}
            for phase_path in sorted(progress_dir.glob("*.json")):
                if phase_path.name.endswith(".worker.json") or phase_path.name.endswith(".result.json"):
                    continue
                try:
                    phase_progress[phase_path.stem] = _safe_phase_progress(
                        json.loads(phase_path.read_text(encoding="utf-8"))
                    )
                except json.JSONDecodeError:
                    continue
            if phase_progress:
                payload["phase_progress"] = phase_progress
    return payload


@app.local_entrypoint()
def main(
    run_id: str,
    report_name: str = "report.json",
    include_progress: bool = True,
) -> None:
    summary = report_summary_remote.remote(
        {
            "run_id": run_id,
            "report_name": report_name,
            "include_progress": include_progress,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
