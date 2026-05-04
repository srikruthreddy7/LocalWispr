#!/usr/bin/env python3
"""Modal-side Hugging Face dataset metadata probe.

Use this for gated datasets when row loading is too expensive. It lists Hub repo
files through the Hugging Face API using the Modal `huggingface-secret`, without
downloading parquet shards.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import modal


HF_SECRET_NAME = os.environ.get("LOCALWISPR_MODAL_LORA_HF_SECRET_NAME", "huggingface-secret")

app = modal.App("localwispr-hf-dataset-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")


@dataclass
class ProbeConfig:
    dataset: str
    pattern: str = ""
    limit: int = 500


@app.function(image=image, timeout=60 * 10, secrets=[modal.Secret.from_name(HF_SECRET_NAME)])
def probe_dataset_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    config = ProbeConfig(**config_payload)
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    dataset_info = api.dataset_info(config.dataset, token=token, files_metadata=True)
    pattern = re.compile(config.pattern, flags=re.IGNORECASE) if config.pattern else None

    siblings = []
    total_size = 0
    matched_size = 0
    for sibling in dataset_info.siblings:
        path = sibling.rfilename
        size = int(getattr(sibling, "size", 0) or 0)
        total_size += size
        if pattern is not None and not pattern.search(path):
            continue
        if len(siblings) < config.limit:
            siblings.append(
                {
                    "path": path,
                    "size": size,
                    "blob_id": getattr(sibling, "blob_id", None),
                    "lfs": bool(getattr(sibling, "lfs", None)),
                }
            )
        matched_size += size

    return {
        "dataset": config.dataset,
        "sha": dataset_info.sha,
        "private": dataset_info.private,
        "gated": getattr(dataset_info, "gated", None),
        "downloads": getattr(dataset_info, "downloads", None),
        "likes": getattr(dataset_info, "likes", None),
        "tags": dataset_info.tags,
        "siblings_total": len(dataset_info.siblings),
        "total_size": total_size,
        "pattern": config.pattern,
        "matched_count_returned": len(siblings),
        "matched_size": matched_size,
        "matched_siblings": siblings,
    }


@app.local_entrypoint()
def main(dataset: str, pattern: str = "", limit: int = 500) -> None:
    report = probe_dataset_remote.remote(
        {
            "dataset": dataset,
            "pattern": pattern,
            "limit": limit,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
