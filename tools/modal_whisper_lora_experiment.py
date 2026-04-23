"""Modal-native Whisper LoRA training + Svarah evaluation workflow.

This script is intended for one narrow experiment:

1. LoRA fine-tune `openai/whisper-large-v3-turbo` on Indian-accent English.
2. Evaluate the resulting adapter against AI4Bharat Svarah.

The default training dataset is `WillHeld/india_accent_cv` because it is public
and explicitly targets Indian-accent English. Svarah is used as the external
evaluation set and requires Hugging Face access approval.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LOCALWISPR_MODAL_LORA_APP_NAME", "localwispr-whisper-lora")
BASE_MODEL = os.environ.get("LOCALWISPR_MODAL_LORA_BASE_MODEL", "openai/whisper-large-v3-turbo")
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
HF_CACHE_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_HF_CACHE_VOLUME", "localwispr-hf-cache"
)
HF_SECRET_NAME = os.environ.get("LOCALWISPR_MODAL_LORA_HF_SECRET_NAME", "huggingface-secret")
TRAIN_GPU = os.environ.get("LOCALWISPR_MODAL_LORA_TRAIN_GPU", "H100!")
FOUR_GPU_TRAIN_GPU = f"{TRAIN_GPU}:4"
FIVE_GPU_TRAIN_GPU = f"{TRAIN_GPU}:5"
H100_BENCHMARK_GPU = os.environ.get("LOCALWISPR_MODAL_LORA_H100_GPU", "H100!")
DEFAULT_ATTN_IMPLEMENTATION = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ATTN_IMPLEMENTATION", "sdpa"
)
LOCAL_PREPARED_DATASET_ROOT = Path("/tmp/localwispr-whisper-lora")

ARTIFACTS_DIR = Path("/artifacts")
HF_CACHE_DIR = Path("/cache/huggingface")

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "wget")
    .pip_install(
        "accelerate==1.2.1",
        "datasets[audio]==3.3.2",
        "evaluate==0.4.3",
        "jiwer==3.0.5",
        "librosa==0.10.2",
        "peft==0.14.0",
        "soundfile==0.12.1",
        "torch==2.5.1",
        "transformers==4.49.0",
    )
    .env(
        {
            "HF_HOME": str(HF_CACHE_DIR),
            "HF_DATASETS_CACHE": str(HF_CACHE_DIR / "datasets"),
            "TRANSFORMERS_CACHE": str(HF_CACHE_DIR / "transformers"),
            "HF_HUB_ETAG_TIMEOUT": "30",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME, image=image)


@dataclass
class DatasetConfig:
    name: str
    config: str | None = None
    config_names: list[str] = field(default_factory=list)
    split: str | None = None
    audio_column: str | None = None
    text_column: str | None = None
    max_samples: int | None = None
    max_word_count: int | None = None
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    trust_remote_code: bool = False
    require_text: bool = False
    metadata_filters: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainConfig:
    experiment_name: str
    recipe: str = "baseline"
    base_model: str = BASE_MODEL
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    target_module_set: str = "full"
    lora_scope: str = "all"
    train_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name="WillHeld/india_accent_cv",
            split="train",
        )
    )
    eval_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name="ai4bharat/Svarah",
            split="test",
        )
    )
    validation_dataset: DatasetConfig | None = None
    anchor_dataset: DatasetConfig | None = None
    language: str = "english"
    task: str = "transcribe"
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-4
    warmup_steps: int = 0
    lr_scheduler_type: str = "constant"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_accumulation_steps: int = 2
    rank: int = 64
    alpha: int = 32
    dropout: float = 0.05
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    preprocess_num_workers: int = 0
    preprocess_batch_size: int = 8
    distributed_gpu_count: int = 1
    ddp_find_unused_parameters: bool = False
    optim: str = "adamw_torch_fused"
    skip_validation_eval: bool = False
    seed: int = 42
    train_validation_split: float = 0.1
    max_new_tokens: int = 256
    logging_steps: int = 10
    save_total_limit: int = 2
    train_max_samples: int | None = None
    anchor_max_samples: int | None = None
    focus_max_samples: int | None = None
    focus_oversample_repeats: int = 0
    focus_short_word_threshold: int = 5
    validation_max_samples: int | None = None
    svarah_max_samples: int | None = None
    normalize_transcripts: bool = False
    push_to_hub: bool = False


@dataclass
class AnalysisConfig:
    analysis_name: str = "svarah-analysis"
    base_model: str = BASE_MODEL
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    eval_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name="ai4bharat/Svarah",
            split="test",
        )
    )
    adapter_runs: list[dict[str, str]] = field(default_factory=list)
    language: str = "english"
    task: str = "transcribe"
    max_new_tokens: int = 256
    per_device_eval_batch_size: int = 4
    min_group_samples: int = 100
    max_groups_per_field: int = 12
    top_examples: int = 5
    row_filters: dict[str, str] = field(default_factory=dict)
    group_fields: list[str] = field(
        default_factory=lambda: [
            "duration_bucket",
            "word_count_bucket",
            "contains_digit",
            "contains_date_like",
            "contains_currency_or_amount",
            "gender",
            "age-group",
            "primary_language",
            "native_place_state",
            "occupation_domain",
        ]
    )


@dataclass
class DatasetProfileConfig:
    dataset: DatasetConfig
    sample_limit: int | None = None
    seed: int = 42
    short_word_threshold: int = 5


@dataclass
class AuditConfig:
    audit_name: str = "dataset-audit"
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name="WillHeld/india_accent_cv",
            split="train",
        )
    )
    sample_limit: int = 500
    seed: int = 42
    stratify_fields: list[str] = field(
        default_factory=lambda: [
            "primary_language",
            "native_place_state",
            "gender",
            "language",
            "state",
            "district",
        ]
    )
    max_samples_per_group: int | None = 50
    export_audio: bool = True
    metadata_fields: list[str] = field(default_factory=list)
    normalize_transcripts: bool = False


@dataclass
class TrainingManifestConfig:
    manifest_name: str = "training-manifest"
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name="WillHeld/india_accent_cv",
            split="train",
        )
    )
    sample_limit: int = 50_000
    output_limit: int = 16_384
    seed: int = 42
    quality_preset: str = "accent_safe"
    selection_strategy: str = "score"
    max_samples_per_speaker: int = 50
    export_audio: bool = True
    normalize_transcripts: bool = True
    metadata_fields: list[str] = field(default_factory=list)


@dataclass
class DatasetConfigSurveyConfig:
    survey_name: str = "dataset-config-survey"
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name="ARTPARK-IISc/Vaani",
            split="train",
        )
    )
    config_name_regex: str = ""
    config_names: list[str] = field(default_factory=list)
    max_configs: int | None = None
    max_workers: int = 6
    sample_transcripts_per_config: int = 3
    top_k: int = 25


@dataclass
class AudioManifestVerifyConfig:
    verify_name: str = "audio-manifest-verify"
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            name="/artifacts/training-manifest/train.jsonl",
            split="train",
            audio_column="audio",
            text_column="text",
        )
    )
    sample_limit: int | None = None
    seed: int = 42


@dataclass
class ExternalArchive:
    filename: str
    url: str


@dataclass
class ExternalArchiveDownloadConfig:
    download_name: str = "external-archives"
    target_subdir: str = "downloads"
    overwrite: bool = False
    timeout_seconds: int = 120
    archive: ExternalArchive = field(
        default_factory=lambda: ExternalArchive(
            filename="archive.tar.gz",
            url="",
        )
    )


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_progress_path(run_dir: Path) -> Path:
    return run_dir / "progress.json"


def _run_progress_dir(run_dir: Path) -> Path:
    return run_dir / "progress"


def _write_run_progress(run_dir: Path, payload: dict[str, Any], *, commit: bool = False) -> None:
    _ensure_dir(run_dir)
    _write_json(_run_progress_path(run_dir), payload)
    if commit:
        artifacts_volume.commit()


def _write_phase_progress(path: Path, payload: dict[str, Any], *, commit: bool = False) -> None:
    _ensure_dir(path.parent)
    _write_json(path, payload)
    if commit:
        artifacts_volume.commit()


def _phase_progress_path(run_dir: Path, phase_key: str) -> Path:
    return _run_progress_dir(run_dir) / f"{phase_key}.json"


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _effective_gradient_accumulation_steps(config: TrainConfig) -> int:
    if config.distributed_gpu_count <= 1:
        return config.gradient_accumulation_steps
    return max(1, config.gradient_accumulation_steps // config.distributed_gpu_count)


def _effective_dataloader_num_workers(config: TrainConfig) -> int:
    if config.dataloader_num_workers > 0:
        return config.dataloader_num_workers
    return 8 if config.distributed_gpu_count > 1 else 2


def _effective_preprocess_num_workers(config: TrainConfig) -> int:
    if config.preprocess_num_workers > 0:
        return config.preprocess_num_workers
    if config.distributed_gpu_count <= 1:
        return 0
    cpu_count = os.cpu_count() or 8
    return max(4, min(16, cpu_count // 2))


def _effective_preprocess_batch_size(config: TrainConfig) -> int:
    return max(1, config.preprocess_batch_size)


def _sanitize_artifact_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-._") or "artifact"


def _normalize_dataset_config(value: DatasetConfig | dict[str, Any]) -> DatasetConfig:
    if isinstance(value, DatasetConfig):
        return value
    payload = dict(value)
    payload["config_names"] = list(payload.get("config_names", []))
    payload["metadata_filters"] = dict(payload.get("metadata_filters", {}))
    return DatasetConfig(**payload)


def _parse_metadata_filters(raw_value: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid metadata filter '{item}'. Expected comma-separated key=value pairs."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(
                f"Invalid metadata filter '{item}'. Filter keys must be non-empty."
            )
        filters[key] = value
    return filters


def _parse_config_names(raw_value: str) -> list[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _normalize_train_config(value: TrainConfig | dict[str, Any]) -> TrainConfig:
    if isinstance(value, TrainConfig):
        config = _apply_recipe_defaults(value)
    else:
        payload = dict(value)
        payload["train_dataset"] = _normalize_dataset_config(payload["train_dataset"])
        payload["eval_dataset"] = _normalize_dataset_config(payload["eval_dataset"])
        if payload.get("validation_dataset") is not None:
            payload["validation_dataset"] = _normalize_dataset_config(payload["validation_dataset"])
        if payload.get("anchor_dataset") is not None:
            payload["anchor_dataset"] = _normalize_dataset_config(payload["anchor_dataset"])
        config = _apply_recipe_defaults(TrainConfig(**payload))

    if config.distributed_gpu_count > 1 and config.gradient_checkpointing:
        config = replace(
            config,
            gradient_checkpointing=False,
            ddp_find_unused_parameters=False,
        )
    return config


def _normalize_analysis_config(value: AnalysisConfig | dict[str, Any]) -> AnalysisConfig:
    if isinstance(value, AnalysisConfig):
        return value

    payload = dict(value)
    payload["eval_dataset"] = _normalize_dataset_config(payload["eval_dataset"])
    payload["adapter_runs"] = [dict(item) for item in payload.get("adapter_runs", [])]
    payload["group_fields"] = list(payload.get("group_fields", []))
    payload["row_filters"] = dict(payload.get("row_filters", {}))
    return AnalysisConfig(**payload)


def _normalize_dataset_profile_config(value: DatasetProfileConfig | dict[str, Any]) -> DatasetProfileConfig:
    if isinstance(value, DatasetProfileConfig):
        return value

    payload = dict(value)
    payload["dataset"] = _normalize_dataset_config(payload["dataset"])
    return DatasetProfileConfig(**payload)


def _normalize_external_archive_download_config(
    value: ExternalArchiveDownloadConfig | dict[str, Any],
) -> ExternalArchiveDownloadConfig:
    if isinstance(value, ExternalArchiveDownloadConfig):
        return value

    payload = dict(value)
    payload["archive"] = ExternalArchive(**dict(payload["archive"]))
    return ExternalArchiveDownloadConfig(**payload)


def _normalize_audit_config(value: AuditConfig | dict[str, Any]) -> AuditConfig:
    if isinstance(value, AuditConfig):
        return value

    payload = dict(value)
    payload["dataset"] = _normalize_dataset_config(payload["dataset"])
    payload["stratify_fields"] = list(payload.get("stratify_fields", []))
    payload["metadata_fields"] = list(payload.get("metadata_fields", []))
    return AuditConfig(**payload)


def _normalize_training_manifest_config(
    value: TrainingManifestConfig | dict[str, Any],
) -> TrainingManifestConfig:
    if isinstance(value, TrainingManifestConfig):
        return value

    payload = dict(value)
    payload["dataset"] = _normalize_dataset_config(payload["dataset"])
    payload["metadata_fields"] = list(payload.get("metadata_fields", []))
    payload.setdefault("selection_strategy", "score")
    return TrainingManifestConfig(**payload)


def _normalize_dataset_survey_config(
    value: DatasetConfigSurveyConfig | dict[str, Any],
) -> DatasetConfigSurveyConfig:
    if isinstance(value, DatasetConfigSurveyConfig):
        return value

    payload = dict(value)
    payload["dataset"] = _normalize_dataset_config(payload["dataset"])
    payload["config_names"] = list(payload.get("config_names", []))
    return DatasetConfigSurveyConfig(**payload)


def _normalize_audio_manifest_verify_config(
    value: AudioManifestVerifyConfig | dict[str, Any],
) -> AudioManifestVerifyConfig:
    if isinstance(value, AudioManifestVerifyConfig):
        return value

    payload = dict(value)
    payload["dataset"] = _normalize_dataset_config(payload["dataset"])
    return AudioManifestVerifyConfig(**payload)


def _apply_recipe_defaults(config: TrainConfig) -> TrainConfig:
    recipe = config.recipe.strip().lower()
    if recipe in ("", "baseline"):
        return config

    if recipe not in {"mixed-anchor-v1", "mixed-format-v1"}:
        raise ValueError(
            f"Unsupported recipe '{config.recipe}'. Expected one of: baseline, mixed-anchor-v1, mixed-format-v1"
        )

    updated = config
    if updated.anchor_dataset is None:
        updated = replace(
            updated,
            anchor_dataset=DatasetConfig(
                name="openslr/librispeech_asr",
                config="clean",
                split="train.100",
                audio_column="audio",
                text_column="text",
            ),
        )
    if updated.train_max_samples is None:
        updated = replace(updated, train_max_samples=40_000)
    if updated.anchor_max_samples is None:
        updated = replace(updated, anchor_max_samples=20_000)
    if updated.validation_max_samples is None:
        updated = replace(updated, validation_max_samples=2_000)
    if updated.num_train_epochs == 3.0:
        updated = replace(updated, num_train_epochs=1.0)
    if updated.learning_rate == 1e-4:
        updated = replace(updated, learning_rate=5e-5)
    if not updated.normalize_transcripts:
        updated = replace(updated, normalize_transcripts=True)
    if recipe == "mixed-format-v1":
        if updated.target_module_set == "full":
            updated = replace(updated, target_module_set="attention")
        if updated.rank == 64:
            updated = replace(updated, rank=16)
        if updated.alpha == 32:
            updated = replace(updated, alpha=32)
        if updated.num_train_epochs == 1.0:
            updated = replace(updated, num_train_epochs=0.5)
        if updated.learning_rate == 5e-5:
            updated = replace(updated, learning_rate=2e-5)
        if updated.focus_max_samples is None:
            updated = replace(updated, focus_max_samples=4_000)
        if updated.focus_oversample_repeats == 0:
            updated = replace(updated, focus_oversample_repeats=2)
    return updated


def _infer_audio_column(features: Any) -> str:
    from datasets import Audio

    for column_name, feature in features.items():
        if isinstance(feature, Audio):
            return column_name

    for candidate in ("audio", "speech", "input_audio"):
        if candidate in features:
            return candidate

    raise ValueError(f"Unable to infer audio column from dataset features: {list(features.keys())}")


def _infer_text_column(features: Any) -> str:
    candidates = (
        "sentence",
        "transcript",
        "transcription",
        "text",
        "normalized_text",
        "raw_text",
    )
    for candidate in candidates:
        if candidate in features:
            return candidate

    raise ValueError(f"Unable to infer text column from dataset features: {list(features.keys())}")


def _resolve_split(dataset_dict: Any, preferred_split: str | None) -> str:
    candidates = [preferred_split, "test", "validation", "valid", "train"]
    for candidate in candidates:
        if candidate and candidate in dataset_dict:
            return candidate

    available = list(dataset_dict.keys())
    if not available:
        raise ValueError("Dataset has no available splits")
    return available[0]


def _resolved_config_names(config: DatasetConfig) -> list[str | None]:
    if config.config and config.config_names:
        raise ValueError("Specify either config or config_names, not both")
    if config.config_names:
        return list(config.config_names)
    return [config.config]


def _finalize_loaded_dataset(dataset, *, config: DatasetConfig):
    from datasets import Audio

    audio_column = config.audio_column or _infer_audio_column(dataset.features)
    text_column = config.text_column or _infer_text_column(dataset.features)
    if config.metadata_filters:
        dataset = _filter_dataset_metadata(dataset, metadata_filters=config.metadata_filters)
    if config.require_text:
        dataset = _filter_dataset_require_text(dataset, text_column=text_column)
    if config.max_word_count is not None:
        dataset = _filter_dataset_max_word_count(
            dataset,
            text_column=text_column,
            max_word_count=config.max_word_count,
        )
    audio_feature = dataset.features.get(audio_column)
    if not (isinstance(audio_feature, Audio) and audio_feature.sampling_rate == 16_000):
        try:
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16_000))
        except Exception:
            sample_audio = dataset[0][audio_column] if len(dataset) else None
            if not (
                isinstance(sample_audio, dict)
                and "array" in sample_audio
                and "sampling_rate" in sample_audio
            ):
                raise
    if config.min_duration_seconds is not None or config.max_duration_seconds is not None:
        dataset = _filter_dataset_duration(
            dataset,
            audio_column=audio_column,
            min_duration_seconds=config.min_duration_seconds,
            max_duration_seconds=config.max_duration_seconds,
        )

    if config.max_samples is not None:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    return dataset, audio_column, text_column


def _load_local_jsonl_manifest(
    path: Path,
    *,
    audio_column: str | None,
    sample_max_hint: int | None,
    sample_seed: int,
    progress_run_dir: Path | None = None,
    progress_payload: dict[str, Any] | None = None,
):
    from datasets import Dataset

    rng = random.Random(sample_seed)
    total_bytes = path.stat().st_size
    rows: list[dict[str, Any]] = []
    rows_seen = 0
    bytes_read = 0
    last_reported_rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            bytes_read += len(line.encode("utf-8"))
            stripped = line.strip()
            if not stripped:
                continue

            row = json.loads(stripped)
            rows_seen += 1

            if audio_column and audio_column in row and isinstance(row[audio_column], str):
                audio_path = row[audio_column].strip()
                if audio_path and not Path(audio_path).is_absolute() and not audio_path.startswith(f"{ARTIFACTS_DIR}/"):
                    row[audio_column] = str(ARTIFACTS_DIR / audio_path.lstrip("/"))

            if sample_max_hint is None or sample_max_hint <= 0:
                rows.append(row)
            elif len(rows) < sample_max_hint:
                rows.append(row)
            else:
                replacement_index = rng.randint(0, rows_seen - 1)
                if replacement_index < sample_max_hint:
                    rows[replacement_index] = row

            if progress_run_dir is not None and (
                rows_seen == 1
                or rows_seen - last_reported_rows >= 2000
                or bytes_read >= total_bytes
            ):
                last_reported_rows = rows_seen
                payload = dict(progress_payload or {})
                payload.update(
                    {
                        "updated_at_utc": _now_iso(),
                        "manifest_rows_seen": rows_seen,
                        "manifest_sample_rows_kept": len(rows),
                        "manifest_bytes_read": bytes_read,
                        "manifest_bytes_total": total_bytes,
                        "manifest_read_fraction": (bytes_read / total_bytes) if total_bytes else None,
                    }
                )
                _write_run_progress(progress_run_dir, payload, commit=True)

    return Dataset.from_list(rows)


def _load_dataset_split(
    config: DatasetConfig,
    *,
    token: str | None,
    sample_max_hint: int | None = None,
    sample_seed: int = 42,
    progress_run_dir: Path | None = None,
    progress_payload: dict[str, Any] | None = None,
):
    from datasets import DatasetDict, concatenate_datasets, load_dataset

    local_path = Path(config.name)
    if local_path.exists() and local_path.is_file() and local_path.suffix.lower() in {".jsonl", ".json", ".csv"}:
        if config.config or config.config_names:
            raise ValueError("Local manifest datasets do not support config/config_names")
        if local_path.suffix.lower() == ".jsonl":
            dataset = _load_local_jsonl_manifest(
                local_path,
                audio_column=config.audio_column,
                sample_max_hint=sample_max_hint,
                sample_seed=sample_seed,
                progress_run_dir=progress_run_dir,
                progress_payload=progress_payload,
            )
        else:
            builder_name = "json" if local_path.suffix.lower() in {".json"} else "csv"
            split_name = config.split or "train"
            dataset = load_dataset(
                builder_name,
                data_files={split_name: str(local_path)},
                split=split_name,
            )
            if sample_max_hint is not None:
                dataset = _sample_dataset_rows(dataset, max_samples=sample_max_hint, seed=sample_seed)
        return _finalize_loaded_dataset(dataset, config=config)

    loaded_datasets = []
    for config_name in _resolved_config_names(config):
        if config.split:
            dataset_or_dict = load_dataset(
                config.name,
                config_name,
                split=config.split,
                token=token,
                trust_remote_code=config.trust_remote_code,
            )
        else:
            dataset_or_dict = load_dataset(
                config.name,
                config_name,
                token=token,
                trust_remote_code=config.trust_remote_code,
            )
        if isinstance(dataset_or_dict, DatasetDict):
            split = _resolve_split(dataset_or_dict, config.split)
            loaded_datasets.append(dataset_or_dict[split])
        else:
            loaded_datasets.append(dataset_or_dict)

    dataset = loaded_datasets[0] if len(loaded_datasets) == 1 else concatenate_datasets(loaded_datasets)
    return _finalize_loaded_dataset(dataset, config=config)


def _sample_dataset_rows(dataset, *, max_samples: int | None, seed: int):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(max_samples))


def _normalized_filter_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def _filter_dataset_require_text(dataset, *, text_column: str):
    selected_indexes = [
        index for index in range(len(dataset)) if str(dataset[index][text_column] or "").strip()
    ]
    if len(selected_indexes) == len(dataset):
        return dataset
    return dataset.select(selected_indexes)


def _filter_dataset_metadata(dataset, *, metadata_filters: dict[str, str]):
    if not metadata_filters:
        return dataset

    normalized_filters = {
        key: _normalized_filter_value(value) for key, value in metadata_filters.items()
    }
    selected_indexes = []
    for index in range(len(dataset)):
        row = dataset[index]
        if all(
            _normalized_filter_value(row.get(field_name)) == expected_value
            for field_name, expected_value in normalized_filters.items()
        ):
            selected_indexes.append(index)

    if len(selected_indexes) == len(dataset):
        return dataset
    return dataset.select(selected_indexes)


def _filter_dataset_max_word_count(dataset, *, text_column: str, max_word_count: int):
    selected_indexes = [
        index
        for index in range(len(dataset))
        if _word_count(str(dataset[index][text_column])) <= max_word_count
    ]
    if len(selected_indexes) == len(dataset):
        return dataset
    return dataset.select(selected_indexes)


def _row_duration_seconds(row: dict[str, Any], *, audio_column: str) -> float | None:
    for candidate in ("duration", "duration_seconds", "Speech_Duration_seconds"):
        value = row.get(candidate)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    value = row.get("duration_ms")
    if value is not None:
        try:
            return float(value) / 1000.0
        except (TypeError, ValueError):
            pass

    audio = row.get(audio_column)
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        sampling_rate = float(audio["sampling_rate"])
        if sampling_rate > 0:
            return float(len(_normalize_audio_array(audio["array"]))) / sampling_rate
    return None


def _download_external_archive_impl(config: ExternalArchiveDownloadConfig) -> dict[str, Any]:
    archive = config.archive
    if not archive.url:
        raise ValueError("archive.url is required")

    print(f"[download] starting {archive.filename}")
    download_dir = (
        ARTIFACTS_DIR
        / _sanitize_artifact_component(config.target_subdir)
        / _sanitize_artifact_component(config.download_name)
    )
    _ensure_dir(download_dir)

    destination = download_dir / archive.filename
    temporary_destination = download_dir / f"{archive.filename}.part"

    metadata: dict[str, Any] = {
        "filename": archive.filename,
        "artifact_path": str(destination),
        "download_name": config.download_name,
        "target_subdir": config.target_subdir,
        "overwrite": config.overwrite,
        "started_at": _now_utc(),
    }

    request = urllib.request.Request(
        archive.url,
        headers={"User-Agent": "LocalWispr Modal Downloader/1.0"},
    )

    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        content_length_header = response.headers.get("Content-Length")
        expected_bytes = int(content_length_header) if content_length_header else None
        metadata["content_length_bytes"] = expected_bytes

    if (
        destination.exists()
        and not config.overwrite
        and expected_bytes is not None
        and destination.stat().st_size == expected_bytes
    ):
        print(f"[download] skipping {archive.filename}; existing file matches expected size")
        metadata["status"] = "skipped"
        metadata["skipped_reason"] = "existing_file_matches_content_length"
        metadata["bytes_written"] = destination.stat().st_size
        metadata["completed_at"] = _now_utc()
        artifacts_volume.commit()
        return metadata

    if destination.exists():
        destination.unlink()
    if temporary_destination.exists():
        temporary_destination.unlink()

    command = [
        "wget",
        "--tries=4",
        f"--timeout={config.timeout_seconds}",
        "--waitretry=5",
        "-O",
        str(temporary_destination),
        archive.url,
    ]
    subprocess.run(command, check=True)

    bytes_written = temporary_destination.stat().st_size
    if expected_bytes is not None and bytes_written != expected_bytes:
        temporary_destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download for {archive.filename} was truncated: expected {expected_bytes} bytes, got {bytes_written}"
        )

    temporary_destination.replace(destination)

    metadata["status"] = "downloaded"
    metadata["bytes_written"] = bytes_written
    metadata["completed_at"] = _now_utc()
    _write_json(download_dir / f"{archive.filename}.download.json", metadata)
    print(f"[download] completed {archive.filename} ({bytes_written} bytes)")
    artifacts_volume.commit()
    return metadata


def _filter_dataset_duration(
    dataset,
    *,
    audio_column: str,
    min_duration_seconds: float | None,
    max_duration_seconds: float | None,
):
    selected_indexes = []
    for index in range(len(dataset)):
        duration_seconds = _row_duration_seconds(dataset[index], audio_column=audio_column)
        if duration_seconds is None:
            continue
        if min_duration_seconds is not None and duration_seconds < min_duration_seconds:
            continue
        if max_duration_seconds is not None and duration_seconds > max_duration_seconds:
            continue
        selected_indexes.append(index)

    if len(selected_indexes) == len(dataset):
        return dataset
    return dataset.select(selected_indexes)


def _load_dataset_slice(config: DatasetConfig, *, token: str | None, rows: int):
    from datasets import concatenate_datasets, load_dataset

    split = config.split or "train"
    config_names = _resolved_config_names(config)
    per_config_rows = max(1, (rows + len(config_names) - 1) // len(config_names))
    loaded_datasets = []
    for config_name in config_names:
        sliced_split = f"{split}[:{per_config_rows}]"
        loaded_datasets.append(
            load_dataset(
                config.name,
                config_name,
                split=sliced_split,
                token=token,
                trust_remote_code=config.trust_remote_code,
            )
        )
    dataset = loaded_datasets[0] if len(loaded_datasets) == 1 else concatenate_datasets(loaded_datasets)
    dataset, audio_column, text_column = _finalize_loaded_dataset(dataset, config=config)
    if len(dataset) > rows:
        dataset = dataset.select(range(rows))
    return dataset, audio_column, text_column


def _load_dataset_streaming_split(config: DatasetConfig, *, token: str | None):
    from datasets import DatasetDict, IterableDatasetDict, load_dataset

    if config.config_names:
        raise ValueError("Streaming mode does not support config_names; survey one config at a time")

    if config.split:
        dataset_or_dict = load_dataset(
            config.name,
            config.config,
            split=config.split,
            token=token,
            trust_remote_code=config.trust_remote_code,
            streaming=True,
        )
    else:
        dataset_or_dict = load_dataset(
            config.name,
            config.config,
            token=token,
            trust_remote_code=config.trust_remote_code,
            streaming=True,
        )

    if isinstance(dataset_or_dict, (DatasetDict, IterableDatasetDict)):
        split = _resolve_split(dataset_or_dict, config.split)
        dataset = dataset_or_dict[split]
    else:
        dataset = dataset_or_dict

    audio_column = config.audio_column or _infer_audio_column(dataset.features)
    text_column = config.text_column or _infer_text_column(dataset.features)
    return dataset, audio_column, text_column


def _strip_transcript_markup(text: str) -> str:
    text = re.sub(r"</?[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\{([^}]+)\}", r" \1 ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_transcript_text(text: str, *, normalizer: Any | None = None) -> str:
    cleaned = _strip_transcript_markup(str(text or ""))
    if normalizer is not None:
        cleaned = normalizer(cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def _row_matches_dataset_config(
    row: dict[str, Any],
    *,
    config: DatasetConfig,
    text_column: str,
    audio_column: str,
) -> tuple[bool, str, float | None]:
    if config.metadata_filters:
        for field_name, expected_value in config.metadata_filters.items():
            if _normalized_filter_value(row.get(field_name)) != _normalized_filter_value(expected_value):
                return False, "", None

    text = str(row.get(text_column) or "")
    if config.require_text and not text.strip():
        return False, text, None
    if config.max_word_count is not None and _word_count(text) > config.max_word_count:
        return False, text, None

    duration_seconds = _row_duration_seconds(row, audio_column=audio_column)
    if config.min_duration_seconds is not None:
        if duration_seconds is None or duration_seconds < config.min_duration_seconds:
            return False, text, duration_seconds
    if config.max_duration_seconds is not None:
        if duration_seconds is None or duration_seconds > config.max_duration_seconds:
            return False, text, duration_seconds

    return True, text, duration_seconds


def _survey_single_dataset_config(
    *,
    base_config: DatasetConfig,
    config_name: str,
    token: str | None,
    sample_transcripts_per_config: int,
) -> dict[str, Any]:
    config = replace(base_config, config=config_name)
    try:
        dataset, audio_column, text_column = _load_dataset_streaming_split(config, token=token)
        row_count = 0
        duration_sum_seconds = 0.0
        duration_count = 0
        word_sum = 0
        sample_transcripts: list[str] = []
        speaker_ids: set[str] = set()
        gender_counts: dict[str, int] = {}
        state_value = "unknown"
        district_value = "unknown"

        for row in dataset:
            matches, text, duration_seconds = _row_matches_dataset_config(
                row,
                config=config,
                text_column=text_column,
                audio_column=audio_column,
            )
            if not matches:
                continue

            row_count += 1
            cleaned_text = _strip_transcript_markup(text)
            word_sum += _word_count(cleaned_text)
            if duration_seconds is not None:
                duration_sum_seconds += duration_seconds
                duration_count += 1
            if len(sample_transcripts) < sample_transcripts_per_config and cleaned_text:
                sample_transcripts.append(cleaned_text)

            speaker_value = row.get("speakerID") or row.get("speaker_id") or row.get("speaker")
            if speaker_value is not None:
                normalized_speaker = str(speaker_value).strip()
                if normalized_speaker:
                    speaker_ids.add(normalized_speaker)

            gender_key = str(row.get("gender") or "").strip() or "unknown"
            gender_counts[gender_key] = gender_counts.get(gender_key, 0) + 1

            if state_value == "unknown":
                state_value = str(row.get("state") or "").strip() or "unknown"
            if district_value == "unknown":
                district_value = str(row.get("district") or "").strip() or "unknown"

        return {
            "config": config_name,
            "state": state_value,
            "district": district_value,
            "rows": row_count,
            "hours": duration_sum_seconds / 3600,
            "mean_duration_seconds": (duration_sum_seconds / duration_count) if duration_count else 0.0,
            "mean_word_count": (word_sum / row_count) if row_count else 0.0,
            "speaker_count": len(speaker_ids),
            "gender_counts": gender_counts,
            "audio_column": audio_column,
            "text_column": text_column,
            "sample_transcripts": sample_transcripts,
        }
    except Exception as exc:
        return {
            "config": config_name,
            "error": str(exc),
        }


def _survey_dataset_configs_impl(config: DatasetConfigSurveyConfig) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import get_dataset_config_names

    token = _get_hf_token()
    config_names = get_dataset_config_names(
        config.dataset.name,
        token=token,
        trust_remote_code=config.dataset.trust_remote_code,
    )
    if config.config_names:
        requested = set(config.config_names)
        config_names = [name for name in config_names if name in requested]
    if config.config_name_regex:
        pattern = re.compile(config.config_name_regex)
        config_names = [name for name in config_names if pattern.search(name)]
    if config.max_configs is not None:
        config_names = config_names[: config.max_configs]

    survey_run_id = f"{config.survey_name}-{_now_utc()}"
    survey_dir = ARTIFACTS_DIR / survey_run_id
    _ensure_dir(survey_dir)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as executor:
        futures = {
            executor.submit(
                _survey_single_dataset_config,
                base_config=replace(config.dataset, config=None),
                config_name=config_name,
                token=token,
                sample_transcripts_per_config=config.sample_transcripts_per_config,
            ): config_name
            for config_name in config_names
        }
        for future in as_completed(futures):
            results.append(future.result())

    successful = [item for item in results if "error" not in item]
    failed = [item for item in results if "error" in item]
    successful.sort(key=lambda item: (item["hours"], item["rows"], item["speaker_count"]), reverse=True)

    summary_rows = []
    csv_columns = [
        "config",
        "state",
        "district",
        "rows",
        "hours",
        "mean_duration_seconds",
        "mean_word_count",
        "speaker_count",
        "gender_counts",
        "sample_transcripts",
        "error",
    ]
    for item in successful + failed:
        summary_rows.append(
            {
                "config": item.get("config"),
                "state": item.get("state"),
                "district": item.get("district"),
                "rows": item.get("rows"),
                "hours": item.get("hours"),
                "mean_duration_seconds": item.get("mean_duration_seconds"),
                "mean_word_count": item.get("mean_word_count"),
                "speaker_count": item.get("speaker_count"),
                "gender_counts": _csv_safe_value(item.get("gender_counts")),
                "sample_transcripts": _csv_safe_value(item.get("sample_transcripts")),
                "error": item.get("error", ""),
            }
        )

    summary_csv_path = survey_dir / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    report = {
        "survey_run_id": survey_run_id,
        "dataset": {
            "name": config.dataset.name,
            "split": config.dataset.split,
            "audio_column": config.dataset.audio_column,
            "text_column": config.dataset.text_column,
            "max_word_count": config.dataset.max_word_count,
            "min_duration_seconds": config.dataset.min_duration_seconds,
            "max_duration_seconds": config.dataset.max_duration_seconds,
            "require_text": config.dataset.require_text,
            "metadata_filters": config.dataset.metadata_filters,
        },
        "config_selection": {
            "config_name_regex": config.config_name_regex,
            "config_names": config.config_names,
            "max_configs": config.max_configs,
            "config_count": len(config_names),
        },
        "summary": {
            "successful_configs": len(successful),
            "failed_configs": len(failed),
            "configs_with_rows": sum(1 for item in successful if item["rows"] > 0),
            "total_rows": sum(item["rows"] for item in successful),
            "total_hours": sum(item["hours"] for item in successful),
            "top_configs_by_hours": successful[: config.top_k],
            "failed": failed,
        },
        "artifacts": {
            "survey_dir": str(survey_dir),
            "summary_csv": str(summary_csv_path),
        },
    }
    _write_json(survey_dir / "report.json", report)
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


def _build_processor(model_name: str, *, language: str, task: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(language=language, task=task)
    return processor


def _normalize_audio_array(array: Any):
    import numpy as np

    audio_array = np.asarray(array, dtype=np.float32)
    if audio_array.ndim <= 1:
        return audio_array.reshape(-1)

    sample_axis = int(np.argmax(audio_array.shape))
    reduce_axes = tuple(index for index in range(audio_array.ndim) if index != sample_axis)
    if reduce_axes:
        audio_array = audio_array.mean(axis=reduce_axes)
    return np.asarray(audio_array).reshape(-1)


def _default_audit_metadata_fields(column_names: list[str]) -> list[str]:
    preferred = [
        "id",
        "speakerID",
        "speaker_id",
        "speaker",
        "gender",
        "age",
        "age-group",
        "language",
        "lang",
        "languagesKnown",
        "primary_language",
        "state",
        "district",
        "isTranscriptionAvailable",
        "native_place_state",
        "native_place_district",
        "occupation_domain",
        "highest_qualification",
        "job_category",
        "UtteranceSequenceID",
    ]
    return [field for field in preferred if field in column_names]


def _csv_safe_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _round_robin_group_indexes(
    grouped_indexes: dict[str, list[int]],
    *,
    sample_limit: int,
    max_samples_per_group: int | None,
) -> list[int]:
    selected_indexes: list[int] = []
    group_keys = sorted(grouped_indexes, key=lambda key: (-len(grouped_indexes[key]), key))
    group_positions = {key: 0 for key in group_keys}
    samples_per_group = {key: 0 for key in group_keys}

    while len(selected_indexes) < sample_limit:
        made_progress = False
        for group_key in group_keys:
            if max_samples_per_group is not None and samples_per_group[group_key] >= max_samples_per_group:
                continue
            position = group_positions[group_key]
            if position >= len(grouped_indexes[group_key]):
                continue

            selected_indexes.append(grouped_indexes[group_key][position])
            group_positions[group_key] += 1
            samples_per_group[group_key] += 1
            made_progress = True

            if len(selected_indexes) >= sample_limit:
                break

        if not made_progress:
            break

    return selected_indexes


def _select_audit_indexes(
    metadata_rows: list[dict[str, Any]],
    *,
    sample_limit: int,
    seed: int,
    stratify_fields: list[str],
    max_samples_per_group: int | None,
) -> tuple[list[int], str | None]:
    shuffled_indexes = list(range(len(metadata_rows)))
    random.Random(seed).shuffle(shuffled_indexes)
    if sample_limit <= 0:
        return shuffled_indexes, None

    for field_name in stratify_fields:
        grouped_indexes: dict[str, list[int]] = {}
        for index in shuffled_indexes:
            group_key = _group_value(metadata_rows[index], field_name)
            grouped_indexes.setdefault(group_key, []).append(index)

        if len(grouped_indexes) < 2:
            continue

        selected_indexes = _round_robin_group_indexes(
            grouped_indexes,
            sample_limit=sample_limit,
            max_samples_per_group=max_samples_per_group,
        )
        if selected_indexes:
            return selected_indexes[:sample_limit], field_name

    return shuffled_indexes[:sample_limit], None


def _build_audit_manifest_impl(config: AuditConfig) -> dict[str, Any]:
    import soundfile as sf
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    token = _get_hf_token()
    dataset, audio_column, text_column = _load_dataset_split(config.dataset, token=token)
    metadata_dataset = dataset.remove_columns([audio_column])
    metadata_rows = [metadata_dataset[index] for index in range(len(metadata_dataset))]

    selected_indexes, stratify_field_used = _select_audit_indexes(
        metadata_rows,
        sample_limit=min(config.sample_limit, len(dataset)),
        seed=config.seed,
        stratify_fields=[field for field in config.stratify_fields if field.strip()],
        max_samples_per_group=config.max_samples_per_group,
    )

    audit_run_id = f"{config.audit_name}-{_now_utc()}"
    audit_dir = ARTIFACTS_DIR / audit_run_id
    audio_dir = audit_dir / "audio"
    _ensure_dir(audit_dir)
    if config.export_audio:
        _ensure_dir(audio_dir)

    transcript_normalizer = BasicTextNormalizer() if config.normalize_transcripts else None
    metadata_fields = config.metadata_fields or _default_audit_metadata_fields(metadata_dataset.column_names)

    manifest_rows = []
    for audit_index, dataset_index in enumerate(selected_indexes, start=1):
        row = dataset[dataset_index]
        transcript = str(row[text_column] or "")
        normalized_transcript = (
            _clean_transcript_text(transcript, normalizer=transcript_normalizer)
            if transcript_normalizer is not None
            else _strip_transcript_markup(transcript)
        )
        duration_seconds = _row_duration_seconds(row, audio_column=audio_column)
        audio_relative_path = ""

        if config.export_audio:
            audio = row[audio_column]
            audio_relative_path = f"audio/{audit_index:04d}.wav"
            sf.write(
                str(audit_dir / audio_relative_path),
                _normalize_audio_array(audio["array"]),
                int(audio["sampling_rate"]),
            )

        manifest_row = {
            "audit_index": audit_index,
            "dataset_index": dataset_index,
            "transcript": transcript,
            "normalized_transcript": normalized_transcript,
            "duration_seconds": duration_seconds,
            "word_count": _word_count(transcript),
            "contains_digit": "yes" if any(character.isdigit() for character in transcript) else "no",
            "contains_date_like": "yes" if _contains_date_like(transcript) else "no",
            "contains_currency_or_amount": "yes" if _contains_currency_or_amount(transcript) else "no",
            "audio_file": audio_relative_path,
            "artifact_audio_file": f"{audit_run_id}/{audio_relative_path}" if audio_relative_path else "",
        }
        score_payload = _score_training_row(
            row,
            text_column=text_column,
            audio_column=audio_column,
            quality_preset="lenient",
        )
        manifest_row.update(
            {
                "quality_score": score_payload["score"],
                "quality_warnings": ";".join(score_payload["warnings"]),
                "quality_reject_reasons": ";".join(score_payload["reject_reasons"]),
                "contains_markup": score_payload["contains_markup"],
                "non_ascii_ratio": score_payload["non_ascii_ratio"],
            }
        )
        for field_name in metadata_fields:
            manifest_row[field_name] = row.get(field_name)
        manifest_rows.append(manifest_row)

    manifest_columns = list(manifest_rows[0].keys()) if manifest_rows else [
        "audit_index",
        "dataset_index",
        "transcript",
        "normalized_transcript",
        "duration_seconds",
        "word_count",
        "contains_digit",
        "contains_date_like",
        "contains_currency_or_amount",
        "audio_file",
    ]
    manifest_csv_path = audit_dir / "manifest.csv"
    with manifest_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_columns)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: _csv_safe_value(value) for key, value in row.items()})

    manifest_jsonl_path = audit_dir / "manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    group_counts = {}
    if stratify_field_used is not None:
        for row in manifest_rows:
            group_key = _group_value(row, stratify_field_used)
            group_counts[group_key] = group_counts.get(group_key, 0) + 1

    report = {
        "audit_run_id": audit_run_id,
        "dataset": {
            "name": config.dataset.name,
            "config": config.dataset.config,
            "split": config.dataset.split,
            "rows": len(dataset),
            "audio_column": audio_column,
            "text_column": text_column,
            "metadata_fields": metadata_fields,
        },
        "sampling": {
            "seed": config.seed,
            "sample_limit": config.sample_limit,
            "selected_rows": len(manifest_rows),
            "stratify_fields": config.stratify_fields,
            "stratify_field_used": stratify_field_used,
            "max_samples_per_group": config.max_samples_per_group,
            "group_counts": group_counts,
        },
        "artifacts": {
            "audit_dir": str(audit_dir),
            "manifest_csv": str(manifest_csv_path),
            "manifest_jsonl": str(manifest_jsonl_path),
            "audio_dir": str(audio_dir) if config.export_audio else None,
        },
    }
    _write_json(audit_dir / "report.json", report)
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


def _manifest_candidate_transfer_score(candidate: dict[str, Any]) -> float:
    score_payload = candidate["score_payload"]
    score = float(score_payload["score"])
    duration_seconds = score_payload["duration_seconds"]
    word_count = int(score_payload["word_count"])

    if duration_seconds is not None:
        duration = float(duration_seconds)
        if 6.0 <= duration <= 10.0:
            score += 24.0
        elif 3.0 <= duration < 6.0:
            score += 10.0
        elif duration < 3.0:
            score -= 10.0
        elif duration > 10.0:
            score -= 20.0

    if word_count >= 11:
        score += 18.0
    elif 6 <= word_count <= 10:
        score += 8.0
    elif word_count <= 5:
        score -= 8.0

    if (
        score_payload["contains_digit"] == "yes"
        or score_payload["contains_date_like"] == "yes"
        or score_payload["contains_currency_or_amount"] == "yes"
    ):
        score -= 2.0

    return score


def _manifest_candidate_bucket(candidate: dict[str, Any]) -> str:
    score_payload = candidate["score_payload"]
    duration_seconds = score_payload["duration_seconds"]
    word_count = int(score_payload["word_count"])
    is_format_sensitive = (
        score_payload["contains_digit"] == "yes"
        or score_payload["contains_date_like"] == "yes"
        or score_payload["contains_currency_or_amount"] == "yes"
    )
    if is_format_sensitive:
        return "format_sensitive"
    if (duration_seconds is not None and float(duration_seconds) >= 6.0) or word_count >= 11:
        return "long_context"
    if (duration_seconds is not None and float(duration_seconds) >= 3.0) and word_count >= 6:
        return "medium_context"
    return "short_context"


def _select_manifest_candidates_by_score(
    candidates: list[dict[str, Any]],
    *,
    output_limit: int,
    max_samples_per_speaker: int,
    rejection_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = []
    speaker_counts: dict[str, int] = {}
    for candidate in candidates:
        speaker_key = str(candidate["speaker_key"])
        if max_samples_per_speaker > 0 and speaker_counts.get(speaker_key, 0) >= max_samples_per_speaker:
            rejection_counts["speaker_cap"] = rejection_counts.get("speaker_cap", 0) + 1
            continue
        selected.append(candidate)
        speaker_counts[speaker_key] = speaker_counts.get(speaker_key, 0) + 1
        if len(selected) >= output_limit:
            break
    return selected, speaker_counts


def _select_manifest_candidates_bucketed_transfer(
    candidates: list[dict[str, Any]],
    *,
    output_limit: int,
    max_samples_per_speaker: int,
    rejection_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            -_manifest_candidate_transfer_score(item),
            item["tie_breaker"],
        ),
    )
    quotas = {
        "long_context": int(round(output_limit * 0.55)),
        "medium_context": int(round(output_limit * 0.35)),
        "short_context": int(round(output_limit * 0.08)),
        "format_sensitive": max(1, output_limit - int(round(output_limit * 0.55)) - int(round(output_limit * 0.35)) - int(round(output_limit * 0.08))),
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    speaker_counts: dict[str, int] = {}

    def try_add(candidate: dict[str, Any]) -> bool:
        candidate_id = int(candidate["dataset_index"])
        if candidate_id in selected_ids:
            return False
        speaker_key = str(candidate["speaker_key"])
        if max_samples_per_speaker > 0 and speaker_counts.get(speaker_key, 0) >= max_samples_per_speaker:
            rejection_counts["speaker_cap"] = rejection_counts.get("speaker_cap", 0) + 1
            return False
        selected.append(candidate)
        selected_ids.add(candidate_id)
        speaker_counts[speaker_key] = speaker_counts.get(speaker_key, 0) + 1
        return True

    for bucket_name, quota in quotas.items():
        if quota <= 0:
            continue
        added_for_bucket = 0
        for candidate in ordered_candidates:
            if len(selected) >= output_limit or added_for_bucket >= quota:
                break
            if _manifest_candidate_bucket(candidate) != bucket_name:
                continue
            if try_add(candidate):
                added_for_bucket += 1

    for candidate in ordered_candidates:
        if len(selected) >= output_limit:
            break
        try_add(candidate)

    return selected, speaker_counts


def _build_training_manifest_impl(config: TrainingManifestConfig) -> dict[str, Any]:
    import soundfile as sf
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    token = _get_hf_token()
    dataset, audio_column, text_column = _load_dataset_split(
        config.dataset,
        token=token,
        sample_max_hint=config.sample_limit,
        sample_seed=config.seed,
    )
    if len(dataset) > config.sample_limit:
        dataset = _sample_dataset_rows(dataset, max_samples=config.sample_limit, seed=config.seed)

    manifest_run_id = f"{config.manifest_name}-{_now_utc()}"
    manifest_dir = ARTIFACTS_DIR / manifest_run_id
    audio_dir = manifest_dir / "audio"
    _ensure_dir(manifest_dir)
    if config.export_audio:
        _ensure_dir(audio_dir)
    _write_run_progress(
        manifest_dir,
        {
            "stage": "build_training_manifest",
            "status": "scoring",
            "updated_at_utc": _now_iso(),
            "rows_loaded": len(dataset),
            "rows_scored": 0,
            "selected_rows": 0,
            "exported_rows": 0,
            "output_limit": config.output_limit,
            "export_audio": config.export_audio,
            "selection_strategy": config.selection_strategy,
        },
        commit=True,
    )

    transcript_normalizer = BasicTextNormalizer() if config.normalize_transcripts else None
    metadata_dataset = dataset.remove_columns([audio_column])
    metadata_fields = config.metadata_fields or _default_audit_metadata_fields(metadata_dataset.column_names)

    rng = random.Random(config.seed)
    candidates = []
    rejection_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    score_values: list[float] = []

    for dataset_index in range(len(dataset)):
        row = metadata_dataset[dataset_index]
        score_payload = _score_training_row(
            row,
            text_column=text_column,
            audio_column=audio_column,
            quality_preset=config.quality_preset,
        )
        score_values.append(float(score_payload["score"]))
        for reason in score_payload["reject_reasons"]:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        for warning in score_payload["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        if (dataset_index + 1) % 5000 == 0 or dataset_index + 1 == len(dataset):
            print(
                f"[manifest:{manifest_run_id}] scored {dataset_index + 1}/{len(dataset)} "
                f"candidate_rows={len(candidates)}"
            )
            _write_run_progress(
                manifest_dir,
                {
                    "stage": "build_training_manifest",
                    "status": "scoring",
                    "updated_at_utc": _now_iso(),
                    "rows_loaded": len(dataset),
                    "rows_scored": dataset_index + 1,
                    "candidate_rows": len(candidates),
                    "selected_rows": 0,
                    "exported_rows": 0,
                    "output_limit": config.output_limit,
                    "export_audio": config.export_audio,
                    "selection_strategy": config.selection_strategy,
                },
                commit=True,
            )
        if score_payload["reject"]:
            continue
        candidates.append(
            {
                "dataset_index": dataset_index,
                "speaker_key": _speaker_key(row),
                "score_payload": score_payload,
                "tie_breaker": rng.random(),
            }
        )

    candidates.sort(key=lambda item: (-float(item["score_payload"]["score"]), item["tie_breaker"]))

    selection_strategy = config.selection_strategy.strip().lower()
    if selection_strategy == "score":
        selected, speaker_counts = _select_manifest_candidates_by_score(
            candidates,
            output_limit=config.output_limit,
            max_samples_per_speaker=config.max_samples_per_speaker,
            rejection_counts=rejection_counts,
        )
    elif selection_strategy == "bucketed_transfer":
        selected, speaker_counts = _select_manifest_candidates_bucketed_transfer(
            candidates,
            output_limit=config.output_limit,
            max_samples_per_speaker=config.max_samples_per_speaker,
            rejection_counts=rejection_counts,
        )
    else:
        raise ValueError("Unsupported manifest selection_strategy. Expected one of: score, bucketed_transfer")
    _write_run_progress(
        manifest_dir,
        {
            "stage": "build_training_manifest",
            "status": "exporting",
            "updated_at_utc": _now_iso(),
            "rows_loaded": len(dataset),
            "rows_scored": len(dataset),
            "candidate_rows": len(candidates),
            "selected_rows": len(selected),
            "exported_rows": 0,
            "output_limit": config.output_limit,
            "export_audio": config.export_audio,
            "selection_strategy": selection_strategy,
        },
        commit=True,
    )

    manifest_rows = []
    transcript_values = []
    duration_values = []
    for selected_index, candidate in enumerate(selected, start=1):
        dataset_index = int(candidate["dataset_index"])
        metadata_row = metadata_dataset[dataset_index]
        raw_transcript = str(metadata_row[text_column] or "")
        transcript = (
            _clean_transcript_text(raw_transcript, normalizer=transcript_normalizer)
            if transcript_normalizer is not None
            else _strip_transcript_markup(raw_transcript)
        )
        audio_relative_path = ""
        artifact_audio_file = ""
        if config.export_audio:
            row = dataset[dataset_index]
            audio = row[audio_column]
            audio_relative_path = f"audio/{selected_index:06d}.wav"
            artifact_audio_file = f"{manifest_run_id}/{audio_relative_path}"
            sf.write(
                str(manifest_dir / audio_relative_path),
                _normalize_audio_array(audio["array"]),
                int(audio["sampling_rate"]),
            )

        score_payload = candidate["score_payload"]
        transcript_values.append(transcript.lower())
        if score_payload["duration_seconds"] is not None:
            duration_values.append(float(score_payload["duration_seconds"]))

        manifest_row = {
            "manifest_index": selected_index,
            "dataset_index": dataset_index,
            "audio": artifact_audio_file,
            "text": transcript,
            "raw_transcript": raw_transcript,
            "duration_seconds": score_payload["duration_seconds"],
            "word_count": score_payload["word_count"],
            "quality_score": score_payload["score"],
            "quality_warnings": ";".join(score_payload["warnings"]),
            "contains_digit": score_payload["contains_digit"],
            "contains_date_like": score_payload["contains_date_like"],
            "contains_currency_or_amount": score_payload["contains_currency_or_amount"],
            "contains_markup": score_payload["contains_markup"],
            "non_ascii_ratio": score_payload["non_ascii_ratio"],
            "source_dataset": config.dataset.name,
            "source_config": config.dataset.config,
            "source_split": config.dataset.split,
            "speaker_key": candidate["speaker_key"],
            "audio_file": audio_relative_path,
        }
        for field_name in metadata_fields:
            manifest_row[field_name] = metadata_row.get(field_name)
        manifest_rows.append(manifest_row)
        if selected_index % 500 == 0 or selected_index == len(selected):
            print(
                f"[manifest:{manifest_run_id}] exported {selected_index}/{len(selected)} "
                f"export_audio={config.export_audio}"
            )
            _write_run_progress(
                manifest_dir,
                {
                    "stage": "build_training_manifest",
                    "status": "exporting",
                    "updated_at_utc": _now_iso(),
                    "rows_loaded": len(dataset),
                    "rows_scored": len(dataset),
                    "candidate_rows": len(candidates),
                    "selected_rows": len(selected),
                    "exported_rows": selected_index,
                    "output_limit": config.output_limit,
                    "export_audio": config.export_audio,
                },
                commit=True,
            )

    manifest_columns = list(manifest_rows[0].keys()) if manifest_rows else []
    manifest_csv_path = manifest_dir / "manifest.csv"
    with manifest_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_columns)
        if manifest_columns:
            writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: _csv_safe_value(value) for key, value in row.items()})

    manifest_jsonl_path = manifest_dir / "manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    training_jsonl_path = manifest_dir / "train.jsonl"
    with training_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(
                json.dumps(
                    {
                        "audio": row["audio"],
                        "text": row["text"],
                        "source_dataset": row["source_dataset"],
                        "dataset_index": row["dataset_index"],
                        "quality_score": row["quality_score"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")

    report = {
        "manifest_run_id": manifest_run_id,
        "dataset": {
            "name": config.dataset.name,
            "config": config.dataset.config,
            "split": config.dataset.split,
            "rows_loaded": len(dataset),
            "audio_column": audio_column,
            "text_column": text_column,
            "metadata_fields": metadata_fields,
        },
        "selection": {
            "sample_limit": config.sample_limit,
            "output_limit": config.output_limit,
            "selected_rows": len(manifest_rows),
            "candidate_rows": len(candidates),
            "quality_preset": config.quality_preset,
            "selection_strategy": selection_strategy,
            "max_samples_per_speaker": config.max_samples_per_speaker,
            "unique_speakers": len(speaker_counts),
            "score_distribution_all_loaded": _summarize_numeric_values(score_values),
            "selected_duration_seconds": _summarize_numeric_values(duration_values),
            "selected_duplicate_texts": [
                item for item in _top_counts(transcript_values, limit=20) if item["count"] > 1
            ],
            "rejection_counts": dict(sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))),
            "warning_counts": dict(sorted(warning_counts.items(), key=lambda item: (-item[1], item[0]))),
        },
        "artifacts": {
            "manifest_dir": str(manifest_dir),
            "manifest_csv": str(manifest_csv_path),
            "manifest_jsonl": str(manifest_jsonl_path),
            "training_jsonl": str(training_jsonl_path),
            "audio_dir": str(audio_dir) if config.export_audio else None,
        },
    }
    _write_json(manifest_dir / "report.json", report)
    _write_run_progress(
        manifest_dir,
        {
            "stage": "build_training_manifest",
            "status": "complete",
            "updated_at_utc": _now_iso(),
            "rows_loaded": len(dataset),
            "rows_scored": len(dataset),
            "candidate_rows": len(candidates),
            "selected_rows": len(manifest_rows),
            "exported_rows": len(manifest_rows),
            "output_limit": config.output_limit,
            "export_audio": config.export_audio,
            "selection_strategy": selection_strategy,
            "report_path": str(manifest_dir / "report.json"),
        },
        commit=True,
    )
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


def _prepare_split(
    dataset,
    *,
    processor: Any,
    audio_column: str,
    text_column: str,
    normalize_transcripts: bool = False,
    map_batch_size: int = 8,
    num_proc: int = 0,
    desc: str | None = None,
    progress_run_dir: Path | None = None,
    progress_phase_key: str | None = None,
    progress_source: str | None = None,
    progress_source_index: int | None = None,
    progress_sources_total: int | None = None,
):
    import math

    transcript_normalizer = None
    if normalize_transcripts:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        transcript_normalizer = BasicTextNormalizer()

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        audios = batch[audio_column]
        transcripts = [
            _clean_transcript_text(value, normalizer=transcript_normalizer) for value in batch[text_column]
        ]

        audio_arrays = [_normalize_audio_array(audio["array"]) for audio in audios]
        sampling_rates = [int(audio["sampling_rate"]) for audio in audios]
        if len(set(sampling_rates)) == 1:
            input_features = processor.feature_extractor(
                audio_arrays,
                sampling_rate=sampling_rates[0],
            ).input_features
        else:
            input_features = [
                processor.feature_extractor(array, sampling_rate=sampling_rate).input_features[0]
                for array, sampling_rate in zip(audio_arrays, sampling_rates)
            ]

        return {
            "input_features": input_features,
            "labels": processor.tokenizer(transcripts).input_ids,
        }

    map_kwargs = {
        "batched": True,
        "batch_size": max(1, map_batch_size),
        "remove_columns": dataset.column_names,
    }
    requested_num_proc = max(0, num_proc)
    used_num_proc = 0
    total_rows = len(dataset)

    def map_dataset(input_dataset):
        nonlocal used_num_proc
        if requested_num_proc > 1:
            try:
                prepared = input_dataset.map(
                    prepare_batch,
                    num_proc=requested_num_proc,
                    desc=desc,
                    **map_kwargs,
                )
                used_num_proc = max(used_num_proc, requested_num_proc)
                return prepared
            except Exception as exc:
                print(
                    f"[prepare_split] num_proc={requested_num_proc} failed with {type(exc).__name__}: {exc}. "
                    "Retrying single-process."
                )
        return input_dataset.map(
            prepare_batch,
            desc=desc,
            **map_kwargs,
        )

    if progress_run_dir is not None and total_rows > 0:
        from datasets import concatenate_datasets

        chunk_rows = max(
            map_kwargs["batch_size"],
            min(
                total_rows,
                max(
                    1024,
                    map_kwargs["batch_size"] * max(1, requested_num_proc if requested_num_proc > 0 else 1) * 4,
                ),
            ),
        )
        total_chunks = int(math.ceil(total_rows / chunk_rows))
        started_at = datetime.now(tz=UTC)
        prepared_chunks = []
        prepared_rows_total = 0

        for chunk_index, start in enumerate(range(0, total_rows, chunk_rows), start=1):
            stop = min(start + chunk_rows, total_rows)
            chunk_dataset = dataset.select(range(start, stop))
            prepared_chunk = map_dataset(chunk_dataset)
            prepared_chunks.append(prepared_chunk)
            prepared_rows_total += len(prepared_chunk)

            elapsed_seconds = max(0.001, (datetime.now(tz=UTC) - started_at).total_seconds())
            rows_per_second = prepared_rows_total / elapsed_seconds
            remaining_rows = max(0, total_rows - prepared_rows_total)
            eta_seconds = remaining_rows / rows_per_second if rows_per_second > 0 else None
            payload = {
                "stage": "preprocess",
                "updated_at_utc": _now_iso(),
                "source": progress_source,
                "source_index": progress_source_index,
                "sources_total": progress_sources_total,
                "source_rows_done": prepared_rows_total,
                "source_rows_total": total_rows,
                "source_chunks_done": chunk_index,
                "source_chunks_total": total_chunks,
                "rows_per_second": rows_per_second,
                "eta_seconds": eta_seconds,
            }
            if progress_phase_key is not None:
                _write_phase_progress(
                    _phase_progress_path(progress_run_dir, progress_phase_key),
                    payload,
                )
            _write_run_progress(progress_run_dir, payload, commit=True)
            print(
                f"[preprocess:{progress_source or progress_phase_key or 'dataset'}] "
                f"chunks {chunk_index}/{total_chunks} rows {prepared_rows_total}/{total_rows}"
            )

        prepared_dataset = prepared_chunks[0] if len(prepared_chunks) == 1 else concatenate_datasets(prepared_chunks)
    else:
        prepared_dataset = map_dataset(dataset)

    return prepared_dataset, {
        "requested_num_proc": requested_num_proc,
        "used_num_proc": used_num_proc,
        "batch_size": max(1, map_batch_size),
        "progress_enabled": progress_run_dir is not None,
        "rows": len(prepared_dataset),
    }


def _contains_date_like(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        r"\bdate of birth\b",
        r"\bdob\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _contains_currency_or_amount(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        r"\b(rs|rupees|inr|usd|dollars?)\b",
        r"₹",
        r"\brefund\b",
        r"\bamount\b",
        r"\bbalance\b",
        r"\baccount\b",
        r"\bbank\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _word_count_bucket(text: str) -> str:
    word_count = _word_count(text)
    if word_count <= 2:
        return "1-2 words"
    if word_count <= 5:
        return "3-5 words"
    if word_count <= 10:
        return "6-10 words"
    return "11+ words"


def _word_count(text: str) -> int:
    cleaned = _strip_transcript_markup(str(text or ""))
    return len([token for token in cleaned.split() if token.strip()])


def _is_format_focus_text(text: str, *, short_word_threshold: int) -> bool:
    text = str(text or "")
    words = [token for token in text.split() if token.strip()]
    return (
        any(character.isdigit() for character in text)
        or _contains_date_like(text)
        or _contains_currency_or_amount(text)
        or len(words) <= short_word_threshold
    )


def _select_format_focus_subset(
    dataset,
    *,
    text_column: str,
    max_samples: int | None,
    seed: int,
    short_word_threshold: int,
):
    focus_indexes = [
        index
        for index in range(len(dataset))
        if _is_format_focus_text(dataset[index][text_column], short_word_threshold=short_word_threshold)
    ]
    if not focus_indexes:
        return dataset.select([])

    focus_dataset = dataset.select(focus_indexes)
    return _sample_dataset_rows(focus_dataset, max_samples=max_samples, seed=seed)


def _build_compute_metrics(processor: Any):
    import evaluate
    import numpy as np
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [_clean_transcript_text(text, normalizer=normalizer) for text in pred_str]
        label_str = [_clean_transcript_text(text, normalizer=normalizer) for text in label_str]
        scored_indexes = [index for index, reference in enumerate(label_str) if reference]
        scored_predictions = [pred_str[index] for index in scored_indexes]
        scored_references = [label_str[index] for index in scored_indexes]

        return {
            "wer": wer_metric.compute(predictions=scored_predictions, references=scored_references)
            if scored_indexes
            else 0.0,
            "cer": cer_metric.compute(predictions=scored_predictions, references=scored_references)
            if scored_indexes
            else 0.0,
            "samples": len(pred_str),
            "scored_samples": len(scored_indexes),
            "skipped_empty_references": len(pred_str) - len(scored_indexes),
        }

    return compute_metrics


def _target_modules(target_module_set: str) -> list[str]:
    normalized = target_module_set.strip().lower()
    if normalized == "attention":
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    if normalized == "full":
        # Following the LoRA guidance in the Thinking Machines post, we cover both
        # attention and MLP projections rather than attention-only adapters.
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    raise ValueError(
        f"Unsupported target_module_set '{target_module_set}'. Expected one of: full, attention"
    )


def _load_base_model(model_name: str, *, attn_implementation: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    return model


def _build_lora_model(model_name: str, config: TrainConfig):
    from peft import LoraConfig, get_peft_model

    model = _load_base_model(model_name, attn_implementation=config.attn_implementation)
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        target_modules=_target_modules(config.target_module_set),
    )
    model = get_peft_model(model, lora_config)
    normalized_scope = config.lora_scope.strip().lower()
    if normalized_scope not in {"all", "encoder"}:
        raise ValueError("Unsupported lora_scope. Expected one of: all, encoder")
    if normalized_scope == "encoder":
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_" in name and ".encoder." not in name:
                param.requires_grad = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def _describe_trainable_parameters(model: Any) -> dict[str, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "trainable": trainable,
        "total": total,
    }


def _resolve_train_validation_splits(
    config: TrainConfig,
    *,
    token: str | None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if run_dir is not None:
        _write_run_progress(
            run_dir,
            {
                "stage": "load_train_split",
                "updated_at_utc": _now_iso(),
                "dataset": config.train_dataset.name,
                "split": config.train_dataset.split or "train",
            },
            commit=True,
        )
    train_dataset, train_audio_column, train_text_column = _load_dataset_split(
        config.train_dataset,
        token=token,
        sample_max_hint=(config.train_max_samples * 2 if config.train_max_samples else None),
        sample_seed=config.seed,
        progress_run_dir=run_dir,
        progress_payload={
            "stage": "load_train_split",
            "dataset": config.train_dataset.name,
            "split": config.train_dataset.split or "train",
        }
        if run_dir is not None
        else None,
    )
    validation_source_summary: dict[str, Any]
    if config.validation_dataset is not None:
        if run_dir is not None:
            _write_run_progress(
                run_dir,
                {
                    "stage": "load_validation_split",
                    "updated_at_utc": _now_iso(),
                    "dataset": config.validation_dataset.name,
                    "split": config.validation_dataset.split or "validation",
                },
                commit=True,
            )
        train_split = train_dataset
        validation_split, validation_audio_column, validation_text_column = _load_dataset_split(
            config.validation_dataset,
            token=token,
            sample_max_hint=(config.validation_max_samples * 2 if config.validation_max_samples else None),
            sample_seed=config.seed + 1,
            progress_run_dir=run_dir,
            progress_payload={
                "stage": "load_validation_split",
                "dataset": config.validation_dataset.name,
                "split": config.validation_dataset.split or "validation",
            }
            if run_dir is not None
            else None,
        )
        validation_source_summary = {
            "role": "explicit_validation",
            "name": config.validation_dataset.name,
            "config": config.validation_dataset.config,
            "config_names": config.validation_dataset.config_names,
            "split": config.validation_dataset.split,
            "audio_column": validation_audio_column,
            "text_column": validation_text_column,
        }
    else:
        if run_dir is not None:
            _write_run_progress(
                run_dir,
                {
                    "stage": "split_validation",
                    "updated_at_utc": _now_iso(),
                    "test_size": config.train_validation_split,
                },
                commit=True,
            )
        train_validation = train_dataset.train_test_split(
            test_size=config.train_validation_split,
            seed=config.seed,
        )
        train_split = train_validation["train"]
        validation_split = train_validation["test"]
        validation_audio_column = train_audio_column
        validation_text_column = train_text_column
        validation_source_summary = {
            "role": "internal_random_split",
            "name": config.train_dataset.name,
            "config": config.train_dataset.config,
            "config_names": config.train_dataset.config_names,
            "split": config.train_dataset.split,
            "audio_column": train_audio_column,
            "text_column": train_text_column,
            "test_size": config.train_validation_split,
        }

    train_split = _sample_dataset_rows(train_split, max_samples=config.train_max_samples, seed=config.seed)
    validation_split = _sample_dataset_rows(
        validation_split,
        max_samples=config.validation_max_samples,
        seed=config.seed + 1,
    )
    validation_source_summary["samples"] = len(validation_split)

    return {
        "train_split": train_split,
        "train_audio_column": train_audio_column,
        "train_text_column": train_text_column,
        "validation_split": validation_split,
        "validation_audio_column": validation_audio_column,
        "validation_text_column": validation_text_column,
        "validation_source_summary": validation_source_summary,
    }


def _build_train_features_and_summaries(
    config: TrainConfig,
    *,
    processor: Any,
    token: str | None,
    train_split,
    train_audio_column: str,
    train_text_column: str,
    run_dir: Path | None = None,
):
    from datasets import concatenate_datasets

    preprocess_num_workers = _effective_preprocess_num_workers(config)
    preprocess_batch_size = _effective_preprocess_batch_size(config)
    preprocess_sources_total = 1 + int(config.anchor_dataset is not None)
    preprocess_source_index = 1

    train_sources = []
    train_source_summaries = [
        {
            "role": "primary",
            "name": config.train_dataset.name,
            "config": config.train_dataset.config,
            "config_names": config.train_dataset.config_names,
            "split": config.train_dataset.split,
            "audio_column": train_audio_column,
            "text_column": train_text_column,
            "samples": len(train_split),
        }
    ]

    if run_dir is not None:
        _write_run_progress(
            run_dir,
            {
                "stage": "preprocess",
                "updated_at_utc": _now_iso(),
                "source": "primary",
                "source_index": preprocess_source_index,
                "sources_total": preprocess_sources_total,
                "source_rows_done": 0,
                "source_rows_total": len(train_split),
                "source_chunks_done": 0,
                "normalize_transcripts": config.normalize_transcripts,
                "distributed_gpu_count": config.distributed_gpu_count,
            },
            commit=True,
        )

    primary_train_features, primary_prepare_summary = _prepare_split(
        train_split,
        processor=processor,
        audio_column=train_audio_column,
        text_column=train_text_column,
        normalize_transcripts=config.normalize_transcripts,
        map_batch_size=preprocess_batch_size,
        num_proc=preprocess_num_workers,
        desc="prepare primary train",
        progress_run_dir=run_dir,
        progress_phase_key="preprocess_primary",
        progress_source="primary",
        progress_source_index=preprocess_source_index,
        progress_sources_total=preprocess_sources_total,
    )
    train_source_summaries[0]["prepare"] = primary_prepare_summary
    train_sources.append(primary_train_features)
    preprocess_source_index += 1

    if config.focus_oversample_repeats > 0:
        primary_focus_dataset = _select_format_focus_subset(
            train_split,
            text_column=train_text_column,
            max_samples=config.focus_max_samples,
            seed=config.seed + 11,
            short_word_threshold=config.focus_short_word_threshold,
        )
        if len(primary_focus_dataset):
            primary_focus_features, primary_focus_prepare_summary = _prepare_split(
                primary_focus_dataset,
                processor=processor,
                audio_column=train_audio_column,
                text_column=train_text_column,
                normalize_transcripts=config.normalize_transcripts,
                map_batch_size=preprocess_batch_size,
                num_proc=preprocess_num_workers,
                desc="prepare primary focus",
            )
            for _ in range(config.focus_oversample_repeats):
                train_sources.append(primary_focus_features)
            train_source_summaries.append(
                {
                    "role": "primary_format_focus",
                    "name": config.train_dataset.name,
                    "config": config.train_dataset.config,
                    "config_names": config.train_dataset.config_names,
                    "split": config.train_dataset.split,
                    "audio_column": train_audio_column,
                    "text_column": train_text_column,
                    "samples": len(primary_focus_dataset),
                    "repeats": config.focus_oversample_repeats,
                    "prepare": primary_focus_prepare_summary,
                }
            )

    if config.anchor_dataset is not None:
        if run_dir is not None:
            _write_run_progress(
                run_dir,
                {
                    "stage": "load_anchor_split",
                    "updated_at_utc": _now_iso(),
                    "source": "anchor",
                    "source_index": preprocess_source_index,
                    "sources_total": preprocess_sources_total,
                    "dataset": config.anchor_dataset.name,
                    "split": config.anchor_dataset.split or "train",
                },
                commit=True,
            )
        anchor_dataset, anchor_audio_column, anchor_text_column = _load_dataset_split(
            config.anchor_dataset,
            token=token,
            sample_max_hint=(config.anchor_max_samples * 4 if config.anchor_max_samples else None),
            sample_seed=config.seed + 2,
            progress_run_dir=run_dir,
            progress_payload={
                "stage": "load_anchor_split",
                "source": "anchor",
                "source_index": preprocess_source_index,
                "sources_total": preprocess_sources_total,
                "dataset": config.anchor_dataset.name,
                "split": config.anchor_dataset.split or "train",
            }
            if run_dir is not None
            else None,
        )
        anchor_dataset = _sample_dataset_rows(
            anchor_dataset,
            max_samples=config.anchor_max_samples,
            seed=config.seed + 2,
        )
        anchor_features, anchor_prepare_summary = _prepare_split(
            anchor_dataset,
            processor=processor,
            audio_column=anchor_audio_column,
            text_column=anchor_text_column,
            normalize_transcripts=config.normalize_transcripts,
            map_batch_size=preprocess_batch_size,
            num_proc=preprocess_num_workers,
            desc="prepare anchor train",
            progress_run_dir=run_dir,
            progress_phase_key="preprocess_anchor",
            progress_source="anchor",
            progress_source_index=preprocess_source_index,
            progress_sources_total=preprocess_sources_total,
        )
        train_sources.append(anchor_features)
        train_source_summaries.append(
            {
                "role": "anchor",
                "name": config.anchor_dataset.name,
                "config": config.anchor_dataset.config,
                "config_names": config.anchor_dataset.config_names,
                "split": config.anchor_dataset.split,
                "audio_column": anchor_audio_column,
                "text_column": anchor_text_column,
                "samples": len(anchor_dataset),
                "prepare": anchor_prepare_summary,
            }
        )
        if config.focus_oversample_repeats > 0:
            anchor_focus_dataset = _select_format_focus_subset(
                anchor_dataset,
                text_column=anchor_text_column,
                max_samples=config.focus_max_samples,
                seed=config.seed + 12,
                short_word_threshold=config.focus_short_word_threshold,
            )
            if len(anchor_focus_dataset):
                anchor_focus_features, anchor_focus_prepare_summary = _prepare_split(
                    anchor_focus_dataset,
                    processor=processor,
                    audio_column=anchor_audio_column,
                    text_column=anchor_text_column,
                    normalize_transcripts=config.normalize_transcripts,
                    map_batch_size=preprocess_batch_size,
                    num_proc=preprocess_num_workers,
                    desc="prepare anchor focus",
                )
                for _ in range(config.focus_oversample_repeats):
                    train_sources.append(anchor_focus_features)
                train_source_summaries.append(
                    {
                        "role": "anchor_format_focus",
                        "name": config.anchor_dataset.name,
                        "config": config.anchor_dataset.config,
                        "config_names": config.anchor_dataset.config_names,
                        "split": config.anchor_dataset.split,
                        "audio_column": anchor_audio_column,
                        "text_column": anchor_text_column,
                        "samples": len(anchor_focus_dataset),
                        "repeats": config.focus_oversample_repeats,
                        "prepare": anchor_focus_prepare_summary,
                    }
                )

    if len(train_sources) == 1:
        train_features = train_sources[0].shuffle(seed=config.seed)
    else:
        train_features = concatenate_datasets(train_sources).shuffle(seed=config.seed)

    if run_dir is not None:
        _write_run_progress(
            run_dir,
            {
                "stage": "preprocess_complete",
                "updated_at_utc": _now_iso(),
                "prepared_rows": len(train_features),
                "train_sources": train_source_summaries,
            },
            commit=True,
        )

    return train_features, train_source_summaries


def _stage_prepared_train_features(train_features, *, run_id: str) -> Path:
    staged_dir = LOCAL_PREPARED_DATASET_ROOT / run_id / "train_features"
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    _ensure_dir(staged_dir.parent)
    train_features.save_to_disk(str(staged_dir))
    return staged_dir


def _build_seq2seq_training_arguments(
    config: TrainConfig,
    *,
    output_dir: Path,
    local_rank: int = -1,
):
    import torch
    from transformers import Seq2SeqTrainingArguments

    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=_effective_gradient_accumulation_steps(config),
        eval_accumulation_steps=config.eval_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        evaluation_strategy="no",
        save_strategy="no",
        save_total_limit=config.save_total_limit,
        predict_with_generate=False,
        generation_max_length=config.max_new_tokens,
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=[],
        load_best_model_at_end=False,
        seed=config.seed,
        dataloader_num_workers=_effective_dataloader_num_workers(config),
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters if config.distributed_gpu_count > 1 else None,
        optim=config.optim,
        local_rank=local_rank,
    )


class _VolumeProgressCallback:
    def __init__(self, *, run_dir: Path, commit_every_steps: int = 50):
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            def __init__(self, parent: "_VolumeProgressCallback"):
                self.parent = parent

            def on_train_begin(self, args, state, control, **kwargs):
                self.parent._write(
                    {
                        "stage": "train",
                        "updated_at_utc": _now_iso(),
                        "train": {
                            "global_step": int(state.global_step),
                            "max_steps": int(state.max_steps),
                            "epoch": float(state.epoch or 0.0),
                            "status": "running",
                        },
                    },
                    commit=True,
                )

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not state.is_world_process_zero:
                    return
                payload = {
                    "stage": "train",
                    "updated_at_utc": _now_iso(),
                    "train": {
                        "global_step": int(state.global_step),
                        "max_steps": int(state.max_steps),
                        "epoch": float(state.epoch or 0.0),
                        "status": "running",
                        "logs": dict(logs or {}),
                    },
                }
                should_commit = (state.global_step - self.parent.last_committed_step) >= self.parent.commit_every_steps
                if should_commit:
                    self.parent.last_committed_step = int(state.global_step)
                self.parent._write(payload, commit=should_commit)

            def on_train_end(self, args, state, control, **kwargs):
                if not state.is_world_process_zero:
                    return
                self.parent._write(
                    {
                        "stage": "train_complete",
                        "updated_at_utc": _now_iso(),
                        "train": {
                            "global_step": int(state.global_step),
                            "max_steps": int(state.max_steps),
                            "epoch": float(state.epoch or 0.0),
                            "status": "complete",
                        },
                    },
                    commit=True,
                )

        self.impl = _Impl(self)
        self.run_dir = run_dir
        self.commit_every_steps = commit_every_steps
        self.last_committed_step = 0

    def _write(self, payload: dict[str, Any], *, commit: bool) -> None:
        _write_run_progress(self.run_dir, payload, commit=commit)


def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _run_optimizer_step(
    *,
    model: Any,
    optimizer: Any,
    collator: Any,
    feature_rows: list[dict[str, Any]],
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    device: str,
) -> dict[str, Any]:
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    consumed_rows = 0
    allowed_keys = {
        "input_features",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
    }
    model_dtype = next(model.parameters()).dtype

    torch.cuda.synchronize()
    started = datetime.now(tz=UTC)

    for micro_step in range(gradient_accumulation_steps):
        batch_start = micro_step * micro_batch_size
        batch_end = batch_start + micro_batch_size
        batch_rows = feature_rows[batch_start:batch_end]
        if len(batch_rows) != micro_batch_size:
            raise ValueError(
                f"Expected {micro_batch_size} rows for micro step {micro_step}, got {len(batch_rows)}"
            )

        batch = collator(batch_rows)
        batch = {key: value for key, value in batch.items() if key in allowed_keys}
        batch = _move_batch_to_device(batch, device)
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(device=device, dtype=model_dtype)
        outputs = model(**batch)
        raw_loss = outputs.loss.detach().float().item()
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        losses.append(raw_loss)
        consumed_rows += len(batch_rows)

    optimizer.step()
    torch.cuda.synchronize()
    ended = datetime.now(tz=UTC)

    return {
        "step_ms": int((ended - started).total_seconds() * 1000),
        "losses": losses,
        "consumed_rows": consumed_rows,
    }


def _evaluate_model(
    *,
    model: Any,
    processor: Any,
    dataset,
    audio_column: str,
    text_column: str,
    language: str,
    task: str,
    max_new_tokens: int,
    batch_size: int,
    phase_name: str | None = None,
    progress_path: Path | None = None,
    progress_commit_interval_batches: int = 100,
    progress_log_interval_batches: int = 25,
) -> dict[str, Any]:
    import evaluate

    prediction_payload = _predict_text_dataset(
        model=model,
        processor=processor,
        dataset=dataset,
        audio_column=audio_column,
        text_column=text_column,
        language=language,
        task=task,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        phase_name=phase_name,
        progress_path=progress_path,
        progress_commit_interval_batches=progress_commit_interval_batches,
        progress_log_interval_batches=progress_log_interval_batches,
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    predictions = prediction_payload["predictions"]
    references = prediction_payload["references"]
    normalized_predictions = prediction_payload["normalized_predictions"]
    normalized_references = prediction_payload["normalized_references"]

    scored_indexes = [
        index for index, reference in enumerate(normalized_references) if str(reference).strip()
    ]
    scored_predictions = [normalized_predictions[index] for index in scored_indexes]
    scored_references = [normalized_references[index] for index in scored_indexes]

    return {
        "samples": len(predictions),
        "scored_samples": len(scored_indexes),
        "skipped_empty_references": len(predictions) - len(scored_indexes),
        "wer": wer_metric.compute(predictions=scored_predictions, references=scored_references)
        if scored_indexes
        else 0.0,
        "cer": cer_metric.compute(predictions=scored_predictions, references=scored_references)
        if scored_indexes
        else 0.0,
        "preview": [
            {
                "reference": references[index],
                "prediction": predictions[index],
            }
            for index in range(min(5, len(predictions)))
        ],
    }


def _predict_text_dataset(
    *,
    model: Any,
    processor: Any,
    dataset,
    audio_column: str,
    text_column: str,
    language: str,
    task: str,
    max_new_tokens: int,
    batch_size: int,
    phase_name: str | None = None,
    progress_path: Path | None = None,
    progress_commit_interval_batches: int = 100,
    progress_log_interval_batches: int = 25,
) -> dict[str, Any]:
    import math
    import torch
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    normalizer = BasicTextNormalizer()

    model = model.to("cuda")
    model.eval()

    predictions: list[str] = []
    references: list[str] = []
    total_samples = len(dataset)
    total_batches = int(math.ceil(total_samples / batch_size)) if total_samples else 0
    last_committed_batch = 0

    for batch_index, start in enumerate(range(0, len(dataset), batch_size), start=1):
        stop = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, stop))
        audios = [row[audio_column] for row in batch]
        refs = [str(row[text_column]) for row in batch]
        inputs = processor.feature_extractor(
            [_normalize_audio_array(audio["array"]) for audio in audios],
            sampling_rate=16_000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to("cuda", dtype=torch.bfloat16)
        attention_mask = None
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                language=language,
                task=task,
                max_new_tokens=max_new_tokens,
            )

        preds = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(preds)
        references.extend(refs)

        if phase_name and (
            batch_index == 1
            or batch_index % progress_log_interval_batches == 0
            or stop >= total_samples
        ):
            payload = {
                "phase": phase_name,
                "updated_at_utc": _now_iso(),
                "samples_done": stop,
                "samples_total": total_samples,
                "batches_done": batch_index,
                "batches_total": total_batches,
            }
            print(
                f"[eval:{phase_name}] batches {batch_index}/{total_batches} "
                f"samples {stop}/{total_samples}"
            )
            if progress_path is not None:
                should_commit = (batch_index - last_committed_batch) >= progress_commit_interval_batches or stop >= total_samples
                if should_commit:
                    last_committed_batch = batch_index
                _write_phase_progress(progress_path, payload, commit=should_commit)

    normalized_predictions = [normalizer(text).strip() for text in predictions]
    normalized_references = [normalizer(text).strip() for text in references]
    return {
        "predictions": predictions,
        "references": references,
        "normalized_predictions": normalized_predictions,
        "normalized_references": normalized_references,
    }


def _predict_dataset(
    *,
    model: Any,
    processor: Any,
    dataset,
    audio_column: str,
    text_column: str,
    language: str,
    task: str,
    max_new_tokens: int,
    batch_size: int,
    phase_name: str | None = None,
    progress_path: Path | None = None,
    progress_commit_interval_batches: int = 100,
    progress_log_interval_batches: int = 25,
) -> dict[str, Any]:
    metadata_dataset = dataset.remove_columns([audio_column])
    metadata_rows = [metadata_dataset[index] for index in range(len(metadata_dataset))]
    prediction_payload = _predict_text_dataset(
        model=model,
        processor=processor,
        dataset=dataset,
        audio_column=audio_column,
        text_column=text_column,
        language=language,
        task=task,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        phase_name=phase_name,
        progress_path=progress_path,
        progress_commit_interval_batches=progress_commit_interval_batches,
        progress_log_interval_batches=progress_log_interval_batches,
    )
    prediction_payload["metadata_rows"] = metadata_rows
    return prediction_payload


def _duration_bucket(duration_seconds: float) -> str:
    if duration_seconds < 3:
        return "<3s"
    if duration_seconds < 6:
        return "3-6s"
    if duration_seconds < 10:
        return "6-10s"
    return "10s+"


def _group_value(row: dict[str, Any], field_name: str) -> str:
    raw_text = str(row.get("text") or "")
    if field_name == "duration_bucket":
        duration = float(row.get("duration") or 0.0)
        return _duration_bucket(duration)
    if field_name == "word_count_bucket":
        return _word_count_bucket(raw_text)
    if field_name == "contains_digit":
        return "yes" if any(character.isdigit() for character in raw_text) else "no"
    if field_name == "contains_date_like":
        return "yes" if _contains_date_like(raw_text) else "no"
    if field_name == "contains_currency_or_amount":
        return "yes" if _contains_currency_or_amount(raw_text) else "no"
    value = row.get(field_name)
    if value is None:
        return "unknown"
    value = str(value).strip()
    return value or "unknown"


def _compute_text_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    from jiwer import cer, wer

    filtered_references = []
    filtered_predictions = []
    for reference, prediction in zip(references, predictions):
        if str(reference).strip():
            filtered_references.append(reference)
            filtered_predictions.append(prediction)

    if not filtered_references:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "scored_samples": 0,
            "skipped_empty_references": len(references),
        }

    return {
        "wer": wer(filtered_references, filtered_predictions),
        "cer": cer(filtered_references, filtered_predictions),
        "scored_samples": len(filtered_references),
        "skipped_empty_references": len(references) - len(filtered_references),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _top_counts(values: list[str], *, limit: int = 20) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return [
        {"value": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text_quality_flags(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    characters = [character for character in stripped if not character.isspace()]
    non_ascii = sum(1 for character in characters if ord(character) > 127)
    return {
        "empty": not stripped,
        "contains_digit": any(character.isdigit() for character in stripped),
        "contains_date_like": _contains_date_like(stripped),
        "contains_currency_or_amount": _contains_currency_or_amount(stripped),
        "contains_markup": bool(re.search(r"</?[^>]+>|\[[^\]]+\]|\{[^}]+\}|_+", stripped)),
        "non_ascii_ratio": (non_ascii / len(characters)) if characters else 0.0,
        "word_count": _word_count(stripped),
    }


def _speaker_key(row: dict[str, Any]) -> str:
    for field_name in ("speaker_key", "client_id", "speaker_id", "speaker", "speaker_id_hash", "user_id"):
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _score_training_row(
    row: dict[str, Any],
    *,
    text_column: str,
    audio_column: str,
    quality_preset: str,
) -> dict[str, Any]:
    text = str(row.get(text_column) or "")
    flags = _text_quality_flags(text)
    duration_seconds = _row_duration_seconds(row, audio_column=audio_column)
    up_votes = _safe_int(row.get("up_votes"))
    down_votes = _safe_int(row.get("down_votes"))
    accent = str(row.get("accents") or row.get("accent") or "")
    language = str(row.get("language") or row.get("locale") or row.get("lang") or "")
    preset = quality_preset.strip().lower()

    score = 100.0
    reject_reasons: list[str] = []
    warnings: list[str] = []

    if flags["empty"]:
        reject_reasons.append("empty_text")
    if duration_seconds is None:
        score -= 12
        warnings.append("missing_duration")
    elif duration_seconds < 1.0:
        score -= 20
        warnings.append("too_short_audio")
    elif duration_seconds > 8.0:
        score -= 18
        warnings.append("too_long_audio")
    elif 1.0 <= duration_seconds <= 6.0:
        score += 8

    word_count = int(flags["word_count"])
    if word_count <= 2:
        score -= 8
        warnings.append("very_short_text")
    elif 5 <= word_count <= 16:
        score += 8
    elif word_count > 22:
        score -= 10
        warnings.append("long_text")

    if flags["contains_markup"]:
        score -= 10
        warnings.append("markup_in_text")
    if flags["non_ascii_ratio"] > 0.05:
        score -= 8
        warnings.append("high_non_ascii_ratio")
    if flags["contains_digit"] or flags["contains_date_like"] or flags["contains_currency_or_amount"]:
        score -= 12
        warnings.append("format_sensitive_text")

    if up_votes is not None:
        if up_votes >= 3:
            score += 5
        elif up_votes < 2:
            score -= 10
            warnings.append("low_up_votes")
    if down_votes is not None and down_votes > 0:
        score -= min(30, down_votes * 10)
        warnings.append("downvoted")
        if down_votes >= 3:
            reject_reasons.append("many_down_votes")

    if accent:
        if "india" in accent.lower() or "south asia" in accent.lower():
            score += 12
        elif preset == "accent_safe":
            reject_reasons.append("non_indian_accent")
    elif preset == "accent_safe":
        score -= 8
        warnings.append("missing_accent_metadata")

    if language:
        normalized_language = language.strip().lower()
        if normalized_language not in {"en", "eng", "english"} and preset == "accent_safe":
            reject_reasons.append("non_english_language")

    return {
        "score": round(max(0.0, min(125.0, score)), 3),
        "reject": bool(reject_reasons),
        "reject_reasons": reject_reasons,
        "warnings": warnings,
        "duration_seconds": duration_seconds,
        "word_count": word_count,
        "contains_digit": "yes" if flags["contains_digit"] else "no",
        "contains_date_like": "yes" if flags["contains_date_like"] else "no",
        "contains_currency_or_amount": "yes" if flags["contains_currency_or_amount"] else "no",
        "contains_markup": "yes" if flags["contains_markup"] else "no",
        "non_ascii_ratio": round(float(flags["non_ascii_ratio"]), 4),
    }


def _summarize_numeric_values(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": _percentile(values, 0.25),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
    }


def _filter_analysis_indexes(
    metadata_rows: list[dict[str, Any]],
    row_filters: dict[str, str],
) -> list[int]:
    if not row_filters:
        return list(range(len(metadata_rows)))

    selected_indexes: list[int] = []
    normalized_filters = {key: str(value).strip() for key, value in row_filters.items()}
    for index, row in enumerate(metadata_rows):
        include = True
        for field_name, expected_value in normalized_filters.items():
            if _group_value(row, field_name) != expected_value:
                include = False
                break
        if include:
            selected_indexes.append(index)
    return selected_indexes


def _select_rows(values: list[Any], indexes: list[int]) -> list[Any]:
    return [values[index] for index in indexes]


def _profile_text_dataset_impl(config: DatasetProfileConfig) -> dict[str, Any]:
    token = _get_hf_token()
    dataset, audio_column, text_column = _load_dataset_split(config.dataset, token=token)
    dataset = _sample_dataset_rows(dataset, max_samples=config.sample_limit, seed=config.seed)
    metadata_dataset = dataset.remove_columns([audio_column])

    counts = {
        "contains_digit": {"yes": 0, "no": 0},
        "contains_date_like": {"yes": 0, "no": 0},
        "contains_currency_or_amount": {"yes": 0, "no": 0},
        "word_count_bucket": {},
    }
    duration_values: list[float] = []
    word_counts: list[float] = []
    metadata_samples: dict[str, list[str]] = {}
    duplicate_counts: dict[str, int] = {}
    quality_scores: list[float] = []
    warning_counts: dict[str, int] = {}
    reject_counts: dict[str, int] = {}

    metadata_fields = [
        field_name
        for field_name in (
            "accents",
            "accent",
            "age",
            "gender",
            "locale",
            "language",
            "state",
            "district",
            "primary_language",
            "native_place_state",
            "occupation_domain",
        )
        if field_name in dataset.column_names
    ]

    for index in range(len(dataset)):
        row = metadata_dataset[index]
        text = str(row[text_column])
        counts["contains_digit"]["yes" if any(character.isdigit() for character in text) else "no"] += 1
        counts["contains_date_like"]["yes" if _contains_date_like(text) else "no"] += 1
        counts["contains_currency_or_amount"]["yes" if _contains_currency_or_amount(text) else "no"] += 1
        bucket = _word_count_bucket(text)
        counts["word_count_bucket"][bucket] = counts["word_count_bucket"].get(bucket, 0) + 1
        duration_seconds = _row_duration_seconds(row, audio_column=audio_column)
        if duration_seconds is not None:
            duration_values.append(duration_seconds)
        word_count = _word_count(text)
        word_counts.append(float(word_count))
        normalized_text = re.sub(r"\s+", " ", text.strip().lower())
        duplicate_counts[normalized_text] = duplicate_counts.get(normalized_text, 0) + 1
        score_payload = _score_training_row(
            row,
            text_column=text_column,
            audio_column=audio_column,
            quality_preset="lenient",
        )
        quality_scores.append(float(score_payload["score"]))
        for warning in score_payload["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        for reason in score_payload["reject_reasons"]:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
        for field_name in metadata_fields:
            metadata_samples.setdefault(field_name, []).append(str(row.get(field_name) or "unknown"))

    focus_dataset = _select_format_focus_subset(
        metadata_dataset,
        text_column=text_column,
        max_samples=None,
        seed=config.seed,
        short_word_threshold=config.short_word_threshold,
    )

    return {
        "dataset": {
            "name": config.dataset.name,
            "config": config.dataset.config,
            "split": config.dataset.split,
            "audio_column": audio_column,
            "text_column": text_column,
            "sampled_rows": len(dataset),
        },
        "short_word_threshold": config.short_word_threshold,
        "focus_rows": len(focus_dataset),
        "focus_ratio": (len(focus_dataset) / len(dataset)) if len(dataset) else 0.0,
        "counts": counts,
        "duration_seconds": _summarize_numeric_values(duration_values),
        "word_count": _summarize_numeric_values(word_counts),
        "quality_score": _summarize_numeric_values(quality_scores),
        "quality_warnings": dict(sorted(warning_counts.items(), key=lambda item: (-item[1], item[0]))),
        "quality_reject_reasons": dict(sorted(reject_counts.items(), key=lambda item: (-item[1], item[0]))),
        "metadata_top_values": {
            field_name: _top_counts(values, limit=12)
            for field_name, values in metadata_samples.items()
        },
        "top_duplicate_texts": [
            {"text": text, "count": count}
            for text, count in sorted(
                ((text, count) for text, count in duplicate_counts.items() if text and count > 1),
                key=lambda item: (-item[1], item[0]),
            )[:20]
        ],
    }


def _contains_indian_lexical_marker(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "india",
        "indian",
        "delhi",
        "mumbai",
        "bombay",
        "bangalore",
        "bengaluru",
        "hyderabad",
        "chennai",
        "madras",
        "kolkata",
        "calcutta",
        "pune",
        "ahmedabad",
        "lucknow",
        "jaipur",
        "kochi",
        "cochin",
        "mysore",
        "mysuru",
        "surat",
        "kanpur",
        "nagpur",
        "patna",
        "bhopal",
        "visakhapatnam",
        "vijayawada",
        "coimbatore",
        "thiruvananthapuram",
        "trivandrum",
        "kerala",
        "karnataka",
        "telangana",
        "andhra",
        "tamil nadu",
        "maharashtra",
        "punjab",
        "gujarat",
        "rajasthan",
        "uttar pradesh",
        "madhya pradesh",
        "arunachal pradesh",
        "assam",
        "bihar",
        "odisha",
        "orissa",
        "haryana",
        "hindi",
        "tamil",
        "telugu",
        "kannada",
        "malayalam",
        "marathi",
        "punjabi",
        "gujarati",
        "bengali",
        "odia",
    )
    return any(marker in lowered for marker in markers)


def _text_shape_flags(raw_text: str, training_text: str) -> dict[str, Any]:
    raw = str(raw_text or "")
    tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", raw)
    capitalized_noninitial = sum(
        1
        for index, token in enumerate(tokens)
        if index > 0 and len(token) > 1 and token[0].isupper() and token[1:].islower()
    )
    all_caps_tokens = sum(1 for token in tokens if len(token) > 1 and token.isupper())
    mixed_case_tokens = sum(
        1
        for token in tokens
        if len(token) > 2 and any(character.islower() for character in token) and any(character.isupper() for character in token)
    )
    return {
        "contains_digit": "yes" if any(character.isdigit() for character in training_text) else "no",
        "contains_date_like": "yes" if _contains_date_like(training_text) else "no",
        "contains_currency_or_amount": "yes" if _contains_currency_or_amount(training_text) else "no",
        "contains_indian_lexical_marker": "yes" if _contains_indian_lexical_marker(training_text) else "no",
        "contains_apostrophe": "yes" if "'" in raw or "'" in training_text else "no",
        "contains_hyphen": "yes" if "-" in raw or "-" in training_text else "no",
        "raw_has_case_signal": "yes" if any(character.isupper() for character in raw) else "no",
        "capitalized_noninitial_tokens": capitalized_noninitial,
        "all_caps_tokens": all_caps_tokens,
        "mixed_case_tokens": mixed_case_tokens,
        "entity_like": "yes" if capitalized_noninitial > 0 or all_caps_tokens > 0 or mixed_case_tokens > 0 else "no",
    }


def _profile_selected_dataset_rows(
    dataset,
    *,
    audio_column: str,
    text_column: str,
    normalize_transcripts: bool,
    short_word_threshold: int,
    quality_preset: str = "lenient",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transcript_normalizer = None
    if normalize_transcripts:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        transcript_normalizer = BasicTextNormalizer()

    metadata_fields = [
        field_name
        for field_name in (
            "accents",
            "accent",
            "age",
            "gender",
            "locale",
            "language",
            "state",
            "district",
            "primary_language",
            "native_place_state",
            "occupation_domain",
            "source_dataset",
            "source_config",
            "speaker_key",
            "client_id",
            "up_votes",
            "down_votes",
            "quality_score",
            "dataset_index",
        )
        if field_name in dataset.column_names
    ]
    metadata_dataset = dataset.remove_columns([audio_column])

    counts = {
        "contains_digit": {"yes": 0, "no": 0},
        "contains_date_like": {"yes": 0, "no": 0},
        "contains_currency_or_amount": {"yes": 0, "no": 0},
        "contains_indian_lexical_marker": {"yes": 0, "no": 0},
        "contains_apostrophe": {"yes": 0, "no": 0},
        "contains_hyphen": {"yes": 0, "no": 0},
        "raw_has_case_signal": {"yes": 0, "no": 0},
        "entity_like": {"yes": 0, "no": 0},
        "word_count_bucket": {},
        "duration_bucket": {},
    }
    duration_values: list[float] = []
    word_counts: list[float] = []
    quality_scores: list[float] = []
    capitalized_noninitial_values: list[float] = []
    all_caps_values: list[float] = []
    mixed_case_values: list[float] = []
    warning_counts: dict[str, int] = {}
    reject_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    metadata_samples: dict[str, list[str]] = {}
    speaker_keys: list[str] = []
    row_payloads: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        metadata_row = metadata_dataset[index]
        source_text = str(metadata_row[text_column] or "")
        raw_text = str(metadata_row.get("raw_transcript") or source_text)
        training_text = (
            _clean_transcript_text(source_text, normalizer=transcript_normalizer)
            if transcript_normalizer is not None
            else _strip_transcript_markup(source_text)
        )
        full_row = None
        duration_seconds = _row_duration_seconds(metadata_row, audio_column=audio_column)
        if duration_seconds is None:
            full_row = dataset[index]
            duration_seconds = _row_duration_seconds(full_row, audio_column=audio_column)
        if duration_seconds is not None:
            duration_values.append(duration_seconds)
            duration_bucket = _duration_bucket(duration_seconds)
        else:
            duration_bucket = "unknown"
        counts["duration_bucket"][duration_bucket] = counts["duration_bucket"].get(duration_bucket, 0) + 1

        word_count = _word_count(training_text)
        word_counts.append(float(word_count))
        word_bucket = _word_count_bucket(training_text)
        counts["word_count_bucket"][word_bucket] = counts["word_count_bucket"].get(word_bucket, 0) + 1

        flags = _text_shape_flags(raw_text, training_text)
        for key in (
            "contains_digit",
            "contains_date_like",
            "contains_currency_or_amount",
            "contains_indian_lexical_marker",
            "contains_apostrophe",
            "contains_hyphen",
            "raw_has_case_signal",
            "entity_like",
        ):
            counts[key][flags[key]] += 1
        capitalized_noninitial_values.append(float(flags["capitalized_noninitial_tokens"]))
        all_caps_values.append(float(flags["all_caps_tokens"]))
        mixed_case_values.append(float(flags["mixed_case_tokens"]))

        normalized_training_text = re.sub(r"\s+", " ", training_text.strip().lower())
        duplicate_counts[normalized_training_text] = duplicate_counts.get(normalized_training_text, 0) + 1
        score_payload = _score_training_row(
            full_row if full_row is not None else metadata_row,
            text_column=text_column,
            audio_column=audio_column,
            quality_preset=quality_preset,
        )
        quality_scores.append(float(score_payload["score"]))
        for warning in score_payload["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        for reason in score_payload["reject_reasons"]:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

        speaker_key = _speaker_key(metadata_row)
        speaker_keys.append(speaker_key)
        for field_name in metadata_fields:
            metadata_samples.setdefault(field_name, []).append(str(metadata_row.get(field_name) or "unknown"))

        row_payload = {
            "selection_index": index + 1,
            "raw_text": raw_text,
            "source_text": source_text,
            "training_text": training_text,
            "duration_seconds": duration_seconds,
            "duration_bucket": duration_bucket,
            "word_count": word_count,
            "word_count_bucket": word_bucket,
            "speaker_key": speaker_key,
            "quality_score": score_payload["score"],
            "quality_warnings": score_payload["warnings"],
            "quality_reject_reasons": score_payload["reject_reasons"],
        }
        row_payload.update(flags)
        for field_name in metadata_fields:
            row_payload[field_name] = metadata_row.get(field_name)
        row_payloads.append(row_payload)

    focus_rows = sum(
        1
        for row in row_payloads
        if _is_format_focus_text(row["training_text"], short_word_threshold=short_word_threshold)
    )
    duplicate_texts = [
        {"text": text, "count": count}
        for text, count in sorted(
            ((text, count) for text, count in duplicate_counts.items() if text and count > 1),
            key=lambda item: (-item[1], item[0]),
        )[:20]
    ]
    speaker_count_values = _top_counts(speaker_keys, limit=20)

    profile = {
        "rows": len(dataset),
        "short_word_threshold": short_word_threshold,
        "focus_rows": focus_rows,
        "focus_ratio": (focus_rows / len(dataset)) if len(dataset) else 0.0,
        "counts": counts,
        "duration_seconds": _summarize_numeric_values(duration_values),
        "word_count": _summarize_numeric_values(word_counts),
        "quality_score": _summarize_numeric_values(quality_scores),
        "text_shape": {
            "capitalized_noninitial_tokens": _summarize_numeric_values(capitalized_noninitial_values),
            "all_caps_tokens": _summarize_numeric_values(all_caps_values),
            "mixed_case_tokens": _summarize_numeric_values(mixed_case_values),
        },
        "quality_warnings": dict(sorted(warning_counts.items(), key=lambda item: (-item[1], item[0]))),
        "quality_reject_reasons": dict(sorted(reject_counts.items(), key=lambda item: (-item[1], item[0]))),
        "metadata_top_values": {
            field_name: _top_counts(values, limit=12)
            for field_name, values in metadata_samples.items()
        },
        "speaker_summary": {
            "unique_speakers": len(set(speaker_keys)),
            "top_speakers": speaker_count_values,
        },
        "top_duplicate_texts": duplicate_texts,
    }
    return profile, row_payloads


def _write_profile_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _profile_train_selection_impl(config: TrainConfig) -> dict[str, Any]:
    token = _get_hf_token()
    profile_run_id = f"{config.experiment_name}-train-selection-profile-{_now_utc()}"
    profile_dir = ARTIFACTS_DIR / profile_run_id
    _ensure_dir(profile_dir)

    split_payload = _resolve_train_validation_splits(config, token=token)
    train_profile, train_rows = _profile_selected_dataset_rows(
        split_payload["train_split"],
        audio_column=split_payload["train_audio_column"],
        text_column=split_payload["train_text_column"],
        normalize_transcripts=config.normalize_transcripts,
        short_word_threshold=config.focus_short_word_threshold,
    )
    validation_profile, validation_rows = _profile_selected_dataset_rows(
        split_payload["validation_split"],
        audio_column=split_payload["validation_audio_column"],
        text_column=split_payload["validation_text_column"],
        normalize_transcripts=config.normalize_transcripts,
        short_word_threshold=config.focus_short_word_threshold,
    )

    train_rows_path = profile_dir / "train_rows.jsonl"
    validation_rows_path = profile_dir / "validation_rows.jsonl"
    _write_profile_rows(train_rows_path, train_rows)
    _write_profile_rows(validation_rows_path, validation_rows)

    report = {
        "profile_run_id": profile_run_id,
        "created_at_utc": _now_iso(),
        "train_config": asdict(config),
        "dataset": {
            "name": config.train_dataset.name,
            "config": config.train_dataset.config,
            "config_names": config.train_dataset.config_names,
            "split": config.train_dataset.split,
            "audio_column": split_payload["train_audio_column"],
            "text_column": split_payload["train_text_column"],
        },
        "selection": {
            "seed": config.seed,
            "train_validation_split": config.train_validation_split,
            "train_max_samples": config.train_max_samples,
            "validation_max_samples": config.validation_max_samples,
            "validation_source": split_payload["validation_source_summary"],
            "normalize_transcripts": config.normalize_transcripts,
        },
        "train_profile": train_profile,
        "validation_profile": validation_profile,
        "artifacts": {
            "profile_dir": str(profile_dir),
            "report": str(profile_dir / "report.json"),
            "train_rows": str(train_rows_path),
            "validation_rows": str(validation_rows_path),
        },
    }
    _write_json(profile_dir / "report.json", report)
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


def _verify_audio_manifest_impl(config: AudioManifestVerifyConfig) -> dict[str, Any]:
    import soundfile as sf

    manifest_path = Path(config.dataset.name)
    if not manifest_path.exists() or manifest_path.suffix.lower() != ".jsonl":
        raise ValueError("verify_audio_manifest currently expects a local JSONL manifest path")

    audio_column = config.dataset.audio_column or "audio"
    text_column = config.dataset.text_column or "text"
    verify_run_id = f"{config.verify_name}-{_now_utc()}"
    verify_dir = ARTIFACTS_DIR / verify_run_id
    _ensure_dir(verify_dir)

    rng = random.Random(config.seed)
    rows: list[dict[str, Any]] = []
    rows_seen = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            rows_seen += 1
            if config.sample_limit is None or config.sample_limit <= 0:
                rows.append(row)
            elif len(rows) < config.sample_limit:
                rows.append(row)
            else:
                replacement_index = rng.randint(0, rows_seen - 1)
                if replacement_index < config.sample_limit:
                    rows[replacement_index] = row

    print(
        f"[verify-audio:{verify_run_id}] loaded manifest_rows={rows_seen} "
        f"rows_to_check={len(rows)}"
    )
    _write_run_progress(
        verify_dir,
        {
            "stage": "verify_audio_manifest",
            "status": "loaded_manifest",
            "updated_at_utc": _now_iso(),
            "manifest_rows": rows_seen,
            "total_rows": len(rows),
            "checked_rows": 0,
            "failures": 0,
        },
        commit=True,
    )

    failures = []
    duration_values: list[float] = []
    sample_rate_counts: dict[str, int] = {}
    channels_counts: dict[str, int] = {}
    bytes_values: list[float] = []
    started_at = datetime.now(tz=UTC)

    for index, row in enumerate(rows, start=1):
        raw_audio_path = str(row.get(audio_column) or "").strip()
        if not raw_audio_path:
            failures.append(
                {
                    "row": index,
                    "reason": "missing_audio_path",
                    "text": str(row.get(text_column) or "")[:200],
                }
            )
            continue
        audio_path = Path(raw_audio_path)
        if not audio_path.is_absolute():
            audio_path = ARTIFACTS_DIR / raw_audio_path.lstrip("/")
        try:
            info = sf.info(str(audio_path))
            duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
            duration_values.append(duration)
            sample_rate_key = str(info.samplerate)
            sample_rate_counts[sample_rate_key] = sample_rate_counts.get(sample_rate_key, 0) + 1
            channels_key = str(info.channels)
            channels_counts[channels_key] = channels_counts.get(channels_key, 0) + 1
            if audio_path.exists():
                bytes_values.append(float(audio_path.stat().st_size))
        except Exception as exc:
            failures.append(
                {
                    "row": index,
                    "audio": str(audio_path),
                    "reason": type(exc).__name__,
                    "message": str(exc)[:500],
                    "text": str(row.get(text_column) or "")[:200],
                }
            )

        if index == 1 or index % 100 == 0 or index == len(rows):
            elapsed = max(0.001, (datetime.now(tz=UTC) - started_at).total_seconds())
            print(
                f"[verify-audio:{verify_run_id}] checked {index}/{len(rows)} "
                f"failures={len(failures)} rows_per_second={index / elapsed:.2f}"
            )
            _write_run_progress(
                verify_dir,
                {
                    "stage": "verify_audio_manifest",
                    "status": "checking",
                    "updated_at_utc": _now_iso(),
                    "checked_rows": index,
                    "total_rows": len(rows),
                    "failures": len(failures),
                    "rows_per_second": index / elapsed,
                },
                commit=True,
            )

    report = {
        "verify_run_id": verify_run_id,
        "created_at_utc": _now_iso(),
        "dataset": {
            "name": config.dataset.name,
            "audio_column": audio_column,
            "text_column": text_column,
            "rows_seen": rows_seen,
            "rows_checked": len(rows),
            "sample_limit": config.sample_limit,
            "seed": config.seed,
        },
        "audio": {
            "valid_rows": len(rows) - len(failures),
            "failed_rows": len(failures),
            "duration_seconds": _summarize_numeric_values(duration_values),
            "file_bytes": _summarize_numeric_values(bytes_values),
            "sample_rate_counts": sample_rate_counts,
            "channels_counts": channels_counts,
            "failures": failures[:50],
        },
        "artifacts": {
            "verify_dir": str(verify_dir),
            "report": str(verify_dir / "report.json"),
            "progress": str(verify_dir / "progress.json"),
        },
    }
    _write_json(verify_dir / "report.json", report)
    _write_run_progress(
        verify_dir,
        {
            "stage": "verify_audio_manifest",
            "status": "complete",
            "updated_at_utc": _now_iso(),
            "checked_rows": len(rows),
            "total_rows": len(rows),
            "failures": len(failures),
            "report_path": str(verify_dir / "report.json"),
        },
        commit=True,
    )
    artifacts_volume.commit()
    return report


def _summarize_group_metrics(
    *,
    metadata_rows: list[dict[str, Any]],
    normalized_references: list[str],
    normalized_predictions_by_model: dict[str, list[str]],
    group_fields: list[str],
    min_group_samples: int,
    max_groups_per_field: int,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    base_predictions = normalized_predictions_by_model["base"]

    for field_name in group_fields:
        grouped_indexes: dict[str, list[int]] = {}
        for index, row in enumerate(metadata_rows):
            group_key = _group_value(row, field_name)
            grouped_indexes.setdefault(group_key, []).append(index)

        rows = []
        for group_key, indexes in grouped_indexes.items():
            if len(indexes) < min_group_samples:
                continue
            references = [normalized_references[index] for index in indexes]
            metrics_by_model = {}
            for label, predictions in normalized_predictions_by_model.items():
                grouped_predictions = [predictions[index] for index in indexes]
                metrics_by_model[label] = _compute_text_metrics(references, grouped_predictions)

            row = {
                "group": group_key,
                "samples": len(indexes),
                "metrics": metrics_by_model,
                "deltas_vs_base": {
                    label: {
                        "wer": metrics_by_model[label]["wer"] - metrics_by_model["base"]["wer"],
                        "cer": metrics_by_model[label]["cer"] - metrics_by_model["base"]["cer"],
                    }
                    for label in normalized_predictions_by_model
                    if label != "base"
                },
            }
            rows.append(row)

        rows.sort(key=lambda item: item["samples"], reverse=True)
        rows = rows[:max_groups_per_field]
        summaries[field_name] = rows

    return summaries


def _summarize_pairwise_examples(
    *,
    metadata_rows: list[dict[str, Any]],
    raw_references: list[str],
    raw_predictions_by_model: dict[str, list[str]],
    normalized_references: list[str],
    normalized_predictions_by_model: dict[str, list[str]],
    top_examples: int,
) -> dict[str, Any]:
    from jiwer import cer

    summaries: dict[str, Any] = {}
    base_predictions = normalized_predictions_by_model["base"]

    for label, predictions in normalized_predictions_by_model.items():
        if label == "base":
            continue

        improved = 0
        worsened = 0
        unchanged = 0
        scored_examples = []

        for index, reference in enumerate(normalized_references):
            base_cer = cer(reference, base_predictions[index])
            candidate_cer = cer(reference, predictions[index])
            delta = candidate_cer - base_cer
            if delta < 0:
                improved += 1
            elif delta > 0:
                worsened += 1
            else:
                unchanged += 1

            scored_examples.append(
                {
                    "index": index,
                    "delta_cer_vs_base": delta,
                    "reference": raw_references[index],
                    "base_prediction": raw_predictions_by_model["base"][index],
                    "candidate_prediction": raw_predictions_by_model[label][index],
                    "metadata": metadata_rows[index],
                }
            )

        regressions = sorted(
            [item for item in scored_examples if item["delta_cer_vs_base"] > 0],
            key=lambda item: item["delta_cer_vs_base"],
            reverse=True,
        )[:top_examples]
        improvements = sorted(
            [item for item in scored_examples if item["delta_cer_vs_base"] < 0],
            key=lambda item: item["delta_cer_vs_base"],
        )[:top_examples]

        summaries[label] = {
            "counts_vs_base": {
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
            },
            "top_regressions_vs_base": regressions,
            "top_improvements_vs_base": improvements,
        }

    return summaries


def _write_pairwise_prediction_rows(
    *,
    output_path: Path,
    metadata_rows: list[dict[str, Any]],
    raw_references: list[str],
    raw_predictions_by_model: dict[str, list[str]],
    normalized_references: list[str],
    normalized_predictions_by_model: dict[str, list[str]],
) -> None:
    from jiwer import cer, wer

    base_predictions = normalized_predictions_by_model["base"]
    with output_path.open("w", encoding="utf-8") as handle:
        for index, reference in enumerate(normalized_references):
            base_prediction = base_predictions[index]
            base_cer = cer(reference, base_prediction)
            base_wer = wer(reference, base_prediction)
            adapters = {}
            for label, predictions in normalized_predictions_by_model.items():
                if label == "base":
                    continue
                candidate_prediction = predictions[index]
                candidate_cer = cer(reference, candidate_prediction)
                candidate_wer = wer(reference, candidate_prediction)
                adapters[label] = {
                    "prediction": raw_predictions_by_model[label][index],
                    "normalized_prediction": candidate_prediction,
                    "cer": candidate_cer,
                    "wer": candidate_wer,
                    "delta_cer_vs_base": candidate_cer - base_cer,
                    "delta_wer_vs_base": candidate_wer - base_wer,
                }
            payload = {
                "index": index,
                "reference": raw_references[index],
                "normalized_reference": reference,
                "base_prediction": raw_predictions_by_model["base"][index],
                "base_normalized_prediction": base_prediction,
                "base_cer": base_cer,
                "base_wer": base_wer,
                "metadata": metadata_rows[index],
                "adapters": adapters,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _analyze_svarah_impl(config: AnalysisConfig) -> dict[str, Any]:
    import torch
    from peft import PeftModel

    hf_token = _get_hf_token()
    processor = _build_processor(config.base_model, language=config.language, task=config.task)
    dataset, audio_column, text_column = _load_dataset_split(config.eval_dataset, token=hf_token)
    analysis_run_id = f"{config.analysis_name}-{_now_utc()}"
    analysis_dir = ARTIFACTS_DIR / analysis_run_id
    _ensure_dir(analysis_dir)
    progress_dir = _run_progress_dir(analysis_dir)
    _ensure_dir(progress_dir)

    predictions_by_model: dict[str, dict[str, Any]] = {}

    print(f"Starting analysis run {analysis_run_id} on {len(dataset)} samples")
    _write_run_progress(
        analysis_dir,
        {
            "stage": "analysis",
            "updated_at_utc": _now_iso(),
            "status": "starting",
            "samples_total": len(dataset),
            "models_total": 1 + len(config.adapter_runs),
        },
        commit=True,
    )
    print("Evaluating base model")
    base_model = _load_base_model(
        config.base_model,
        attn_implementation=config.attn_implementation,
    )
    predictions_by_model["base"] = _predict_dataset(
        model=base_model,
        processor=processor,
        dataset=dataset,
        audio_column=audio_column,
        text_column=text_column,
        language=config.language,
        task=config.task,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.per_device_eval_batch_size,
        phase_name="analysis base",
        progress_path=_phase_progress_path(analysis_dir, "base"),
        progress_log_interval_batches=25,
    )
    del base_model
    torch.cuda.empty_cache()

    for adapter_index, adapter_run in enumerate(config.adapter_runs, start=1):
        label = adapter_run["label"]
        adapter_dir = Path(adapter_run["adapter_dir"])
        print(f"Evaluating adapter {label} from {adapter_dir}")
        _write_run_progress(
            analysis_dir,
            {
                "stage": "analysis",
                "updated_at_utc": _now_iso(),
                "status": "running",
                "current_model": label,
                "models_done": adapter_index,
                "models_total": 1 + len(config.adapter_runs),
                "samples_total": len(dataset),
            },
            commit=True,
        )
        base_model = _load_base_model(
            config.base_model,
            attn_implementation=config.attn_implementation,
        )
        adapter_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        predictions_by_model[label] = _predict_dataset(
            model=adapter_model,
            processor=processor,
            dataset=dataset,
            audio_column=audio_column,
            text_column=text_column,
            language=config.language,
            task=config.task,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.per_device_eval_batch_size,
            phase_name=f"analysis {label}",
            progress_path=_phase_progress_path(analysis_dir, _sanitize_artifact_component(label)),
            progress_log_interval_batches=25,
        )
        del adapter_model
        del base_model
        torch.cuda.empty_cache()

    metadata_rows = predictions_by_model["base"]["metadata_rows"]
    normalized_references = predictions_by_model["base"]["normalized_references"]
    raw_references = predictions_by_model["base"]["references"]
    selected_indexes = _filter_analysis_indexes(metadata_rows, config.row_filters)
    filtered_metadata_rows = _select_rows(metadata_rows, selected_indexes)
    filtered_normalized_references = _select_rows(normalized_references, selected_indexes)
    filtered_raw_references = _select_rows(raw_references, selected_indexes)
    normalized_predictions_by_model = {
        label: _select_rows(payload["normalized_predictions"], selected_indexes)
        for label, payload in predictions_by_model.items()
    }
    raw_predictions_by_model = {
        label: _select_rows(payload["predictions"], selected_indexes)
        for label, payload in predictions_by_model.items()
    }

    overall_metrics = {
        label: _compute_text_metrics(filtered_normalized_references, predictions)
        for label, predictions in normalized_predictions_by_model.items()
    }
    pairwise_predictions_path = analysis_dir / "pairwise_predictions.jsonl"
    _write_pairwise_prediction_rows(
        output_path=pairwise_predictions_path,
        metadata_rows=filtered_metadata_rows,
        raw_references=filtered_raw_references,
        raw_predictions_by_model=raw_predictions_by_model,
        normalized_references=filtered_normalized_references,
        normalized_predictions_by_model=normalized_predictions_by_model,
    )

    report = {
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "analysis_run_id": analysis_run_id,
        "runtime": {
            "modal_gpu": os.environ.get("MODAL_GPU_LABEL", "unknown"),
            "attn_implementation": config.attn_implementation,
        },
        "dataset": {
            "name": config.eval_dataset.name,
            "config": config.eval_dataset.config,
            "split": config.eval_dataset.split,
            "audio_column": audio_column,
            "text_column": text_column,
            "samples": len(dataset),
            "filtered_samples": len(selected_indexes),
        },
        "row_filters": config.row_filters,
        "overall_metrics": overall_metrics,
        "group_metrics": _summarize_group_metrics(
            metadata_rows=filtered_metadata_rows,
            normalized_references=filtered_normalized_references,
            normalized_predictions_by_model=normalized_predictions_by_model,
            group_fields=config.group_fields,
            min_group_samples=config.min_group_samples,
            max_groups_per_field=config.max_groups_per_field,
        ),
        "pairwise_vs_base": _summarize_pairwise_examples(
            metadata_rows=filtered_metadata_rows,
            raw_references=filtered_raw_references,
            raw_predictions_by_model=raw_predictions_by_model,
            normalized_references=filtered_normalized_references,
            normalized_predictions_by_model=normalized_predictions_by_model,
            top_examples=config.top_examples,
        ),
        "adapter_runs": config.adapter_runs,
        "artifacts": {
            "report_path": str(analysis_dir / "report.json"),
            "pairwise_predictions_jsonl": str(pairwise_predictions_path),
            "progress_path": str(_run_progress_path(analysis_dir)),
        },
    }
    _write_json(analysis_dir / "report.json", report)
    _write_run_progress(
        analysis_dir,
        {
            "stage": "complete",
            "updated_at_utc": _now_iso(),
            "status": "complete",
            "samples_total": len(dataset),
            "models_total": 1 + len(config.adapter_runs),
        },
        commit=True,
    )
    artifacts_volume.commit()
    hf_cache_volume.commit()
    print(f"Finished analysis run {analysis_run_id}")
    return report


def _worker_script_path() -> Path:
    return Path(__file__)


def _distributed_worker_main(config_path: str) -> None:
    from datasets import load_from_disk
    from torch import distributed as dist
    from transformers import Seq2SeqTrainer

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = _normalize_train_config(payload["train_config"])
    prepared_train_features_dir = Path(payload["prepared_train_features_dir"])
    run_dir = Path(payload["run_dir"])
    adapter_dir = Path(payload["adapter_dir"])
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    import torch

    torch.cuda.set_device(local_rank)
    processor = _build_processor(config.base_model, language=config.language, task=config.task)
    train_features = load_from_disk(str(prepared_train_features_dir))
    model = _build_lora_model(config.base_model, config)
    parameter_counts = _describe_trainable_parameters(model)
    training_args = _build_seq2seq_training_arguments(
        config,
        output_dir=run_dir / "trainer",
        local_rank=local_rank,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_features,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        tokenizer=processor.feature_extractor,
    )
    trainer.add_callback(_VolumeProgressCallback(run_dir=run_dir).impl)
    train_result = trainer.train()

    rank_reports_dir = run_dir / "ddp_rank_reports"
    _ensure_dir(rank_reports_dir)
    rank_report = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(local_rank),
        "cuda_device_count": torch.cuda.device_count(),
        "trainer_metrics": train_result.metrics,
    }
    _write_json(rank_reports_dir / f"rank-{rank}.json", rank_report)

    barrier()

    if trainer.is_world_process_zero():
        trainer.save_model(str(adapter_dir))
        processor.save_pretrained(str(adapter_dir))
        distributed_report = {
            "trainer_metrics": train_result.metrics,
            "parameter_counts": parameter_counts,
            "distributed": {
                "enabled": True,
                "requested_gpu_count": config.distributed_gpu_count,
                "world_size": world_size,
                "effective_gradient_accumulation_steps": _effective_gradient_accumulation_steps(config),
                "dataloader_num_workers": _effective_dataloader_num_workers(config),
                "preprocess_num_workers": _effective_preprocess_num_workers(config),
                "preprocess_batch_size": _effective_preprocess_batch_size(config),
                "rank_reports": sorted(str(path) for path in rank_reports_dir.glob("rank-*.json")),
            },
        }
        _write_json(run_dir / "distributed_train_result.json", distributed_report)

    barrier()


def _load_eval_dataset_for_role(config: TrainConfig, *, role: str, token: str | None):
    if role == "validation":
        split_payload = _resolve_train_validation_splits(config, token=token)
        return (
            split_payload["validation_split"],
            split_payload["validation_audio_column"],
            split_payload["validation_text_column"],
        )
    if role == "svarah":
        return _load_dataset_split(
            DatasetConfig(
                name=config.eval_dataset.name,
                config=config.eval_dataset.config,
                config_names=config.eval_dataset.config_names,
                split=config.eval_dataset.split,
                audio_column=config.eval_dataset.audio_column,
                text_column=config.eval_dataset.text_column,
                max_samples=config.svarah_max_samples or config.eval_dataset.max_samples,
                max_word_count=config.eval_dataset.max_word_count,
                min_duration_seconds=config.eval_dataset.min_duration_seconds,
                max_duration_seconds=config.eval_dataset.max_duration_seconds,
                trust_remote_code=config.eval_dataset.trust_remote_code,
                require_text=config.eval_dataset.require_text,
                metadata_filters=config.eval_dataset.metadata_filters,
            ),
            token=token,
        )
    raise ValueError(f"Unsupported eval role '{role}'")


def _slice_dataset_rows(dataset, *, start: int | None, stop: int | None):
    if start is None and stop is None:
        return dataset
    start_index = max(0, int(start or 0))
    stop_index = len(dataset) if stop is None else min(len(dataset), int(stop))
    return dataset.select(range(start_index, stop_index))


def _eval_worker_main(config_path: str) -> None:
    from peft import PeftModel

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = _normalize_train_config(payload["train_config"])
    role = str(payload["role"])
    phase_key = str(payload["phase_key"])
    phase_name = str(payload["phase_name"])
    start = payload.get("slice_start")
    stop = payload.get("slice_stop")
    token = _get_hf_token()
    processor = _build_processor(config.base_model, language=config.language, task=config.task)
    dataset, audio_column, text_column = _load_eval_dataset_for_role(config, role=role, token=token)
    dataset = _slice_dataset_rows(dataset, start=start, stop=stop)
    base_model = _load_base_model(
        config.base_model,
        attn_implementation=config.attn_implementation,
    )
    if payload.get("model_role") == "adapter":
        model = PeftModel.from_pretrained(base_model, str(payload["adapter_dir"]))
    else:
        model = base_model

    prediction_payload = _predict_text_dataset(
        model=model,
        processor=processor,
        dataset=dataset,
        audio_column=audio_column,
        text_column=text_column,
        language=config.language,
        task=config.task,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.per_device_eval_batch_size,
        phase_name=phase_name,
        progress_path=Path(payload["progress_path"]),
    )
    result_payload = {
        "phase_key": phase_key,
        "phase_name": phase_name,
        "role": role,
        "model_role": payload["model_role"],
        "slice_start": start,
        "slice_stop": stop,
        "sample_count": len(prediction_payload["references"]),
        "predictions": prediction_payload["predictions"],
        "references": prediction_payload["references"],
        "normalized_predictions": prediction_payload["normalized_predictions"],
        "normalized_references": prediction_payload["normalized_references"],
    }
    _write_phase_progress(Path(payload["output_path"]), result_payload, commit=True)


def _eval_phase_worker_payloads(
    config: TrainConfig,
    *,
    run_dir: Path,
    adapter_dir: Path,
    validation_samples: int,
    svarah_samples: int,
) -> list[dict[str, Any]]:
    gpu_count = max(1, config.distributed_gpu_count)
    if gpu_count >= 5:
        validation_shards = 3
    elif gpu_count == 4:
        validation_shards = 2
    else:
        validation_shards = 1

    payloads: list[dict[str, Any]] = []
    progress_dir = _run_progress_dir(run_dir)
    _ensure_dir(progress_dir)

    validation_gpu_indexes = list(range(validation_shards))
    shard_size = (validation_samples + validation_shards - 1) // validation_shards
    for shard_index, gpu_index in enumerate(validation_gpu_indexes):
        slice_start = shard_index * shard_size
        slice_stop = min(validation_samples, (shard_index + 1) * shard_size)
        if slice_start >= slice_stop:
            continue
        phase_key = f"validation-shard-{shard_index}"
        payloads.append(
            {
                "phase_key": phase_key,
                "phase_name": f"validation shard {shard_index + 1}/{validation_shards}",
                "role": "validation",
                "model_role": "adapter",
                "slice_start": slice_start,
                "slice_stop": slice_stop,
                "sample_count": slice_stop - slice_start,
                "gpu_index": gpu_index,
                "progress_path": str(_phase_progress_path(run_dir, phase_key)),
                "output_path": str(progress_dir / f"{phase_key}.result.json"),
                "adapter_dir": str(adapter_dir),
                "train_config": asdict(config),
            }
        )

    next_gpu_index = validation_shards
    for model_role in ("base", "adapter"):
        phase_key = f"svarah-{model_role}"
        payloads.append(
            {
                "phase_key": phase_key,
                "phase_name": f"svarah {model_role}",
                "role": "svarah",
                "model_role": model_role,
                "slice_start": None,
                "slice_stop": None,
                "sample_count": svarah_samples,
                "gpu_index": next_gpu_index,
                "progress_path": str(_phase_progress_path(run_dir, phase_key)),
                "output_path": str(progress_dir / f"{phase_key}.result.json"),
                "adapter_dir": str(adapter_dir),
                "train_config": asdict(config),
            }
        )
        next_gpu_index += 1

    return payloads


def _aggregate_eval_progress(run_dir: Path, worker_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    samples_done = 0
    samples_total = 0
    for payload in worker_payloads:
        samples_total += int(payload["sample_count"])
        progress_path = Path(payload["progress_path"])
        phase_state = {
            "phase_name": payload["phase_name"],
            "samples_total": int(payload["sample_count"]),
            "samples_done": 0,
            "batches_done": 0,
            "batches_total": None,
        }
        if progress_path.exists():
            phase_state.update(json.loads(progress_path.read_text(encoding="utf-8")))
        phases[payload["phase_key"]] = phase_state
        samples_done += int(phase_state.get("samples_done", 0))

    return {
        "stage": "eval",
        "updated_at_utc": _now_iso(),
        "eval": {
            "samples_done": samples_done,
            "samples_total": samples_total,
            "percent_complete": (samples_done / samples_total) if samples_total else 0.0,
            "phases": phases,
        },
    }


def _compute_metrics_from_prediction_payload(prediction_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = _compute_text_metrics(
        prediction_payload["normalized_references"],
        prediction_payload["normalized_predictions"],
    )
    metrics.update(
        {
            "samples": len(prediction_payload["references"]),
            "preview": [
                {
                    "reference": prediction_payload["references"][index],
                    "prediction": prediction_payload["predictions"][index],
                }
                for index in range(min(5, len(prediction_payload["references"])))
            ],
        }
    )
    return metrics


def _merge_prediction_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(payloads, key=lambda item: int(item.get("slice_start") or 0))
    merged = {
        "predictions": [],
        "references": [],
        "normalized_predictions": [],
        "normalized_references": [],
    }
    for payload in ordered:
        for key in merged:
            merged[key].extend(payload[key])
    return merged


def _run_parallel_eval_impl(config: TrainConfig, *, partial_report: dict[str, Any]) -> dict[str, Any]:
    hf_token = _get_hf_token()
    validation_samples = 0
    if not config.skip_validation_eval:
        validation_dataset, _, _ = _load_eval_dataset_for_role(config, role="validation", token=hf_token)
        validation_samples = len(validation_dataset)
    svarah_dataset, _, _ = _load_eval_dataset_for_role(config, role="svarah", token=hf_token)
    adapter_dir = Path(partial_report["artifacts"]["adapter_dir"])
    run_dir = adapter_dir.parent

    worker_payloads = _eval_phase_worker_payloads(
        config,
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        validation_samples=validation_samples,
        svarah_samples=len(svarah_dataset),
    )
    _write_run_progress(
        run_dir,
        {
            "stage": "eval",
            "updated_at_utc": _now_iso(),
            "eval": {
                "status": "starting",
                "samples_total": sum(int(item["sample_count"]) for item in worker_payloads),
                "phases": {
                    item["phase_key"]: {
                        "phase_name": item["phase_name"],
                        "samples_total": int(item["sample_count"]),
                        "gpu_index": int(item["gpu_index"]),
                    }
                    for item in worker_payloads
                },
            },
        },
        commit=True,
    )

    processes: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
    try:
        for payload in worker_payloads:
            worker_config_path = _run_progress_dir(run_dir) / f"{payload['phase_key']}.worker.json"
            _write_phase_progress(worker_config_path, payload)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(payload["gpu_index"])
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(_worker_script_path()),
                    "--eval-worker-config-path",
                    str(worker_config_path),
                ],
                cwd=str(_worker_script_path().parent),
                env=env,
            )
            processes.append((payload, process))

        last_commit = time.monotonic()
        while processes:
            next_processes: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
            for payload, process in processes:
                return_code = process.poll()
                if return_code is None:
                    next_processes.append((payload, process))
                    continue
                if return_code != 0:
                    raise RuntimeError(
                        f"Eval worker {payload['phase_key']} failed with exit code {return_code}"
                    )

            processes = next_processes
            progress_payload = _aggregate_eval_progress(run_dir, worker_payloads)
            should_commit = (time.monotonic() - last_commit) >= 15 or not processes
            _write_run_progress(run_dir, progress_payload, commit=should_commit)
            if should_commit:
                last_commit = time.monotonic()
            if processes:
                time.sleep(5)
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()

    validation_payloads = [
        json.loads(Path(payload["output_path"]).read_text(encoding="utf-8"))
        for payload in worker_payloads
        if payload["role"] == "validation"
    ]
    svarah_base_payload = json.loads(
        Path(next(payload["output_path"] for payload in worker_payloads if payload["phase_key"] == "svarah-base")).read_text(
            encoding="utf-8"
        )
    )
    svarah_adapter_payload = json.loads(
        Path(next(payload["output_path"] for payload in worker_payloads if payload["phase_key"] == "svarah-adapter")).read_text(
            encoding="utf-8"
        )
    )

    validation_metrics = None
    if validation_payloads:
        validation_merged = _merge_prediction_payloads(validation_payloads)
        validation_metrics = _compute_metrics_from_prediction_payload(validation_merged)
    base_metrics = _compute_metrics_from_prediction_payload(svarah_base_payload)
    adapter_metrics = _compute_metrics_from_prediction_payload(svarah_adapter_payload)

    report = dict(partial_report)
    if validation_metrics is not None:
        report["training"]["validation_metrics"] = validation_metrics
    report["eval_dataset"] = {
        "name": config.eval_dataset.name,
        "config": config.eval_dataset.config,
        "config_names": config.eval_dataset.config_names,
        "split": config.eval_dataset.split,
        "audio_column": config.eval_dataset.audio_column or "audio_filepath",
        "text_column": config.eval_dataset.text_column or "text",
        "samples": len(svarah_dataset),
    }
    report["svarah"] = {
        "base_model": base_metrics,
        "adapter": adapter_metrics,
        "delta": {
            "wer": adapter_metrics["wer"] - base_metrics["wer"],
            "cer": adapter_metrics["cer"] - base_metrics["cer"],
        },
    }
    _write_json(run_dir / "report.json", report)
    _write_run_progress(
        run_dir,
        {
            "stage": "complete",
            "updated_at_utc": _now_iso(),
            "eval": {
                "status": "complete",
            },
        },
        commit=True,
    )
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


def _run_single_gpu_training_phase(
    config: TrainConfig,
    *,
    processor: Any,
    train_features,
    run_dir: Path,
    adapter_dir: Path,
) -> dict[str, Any]:
    from transformers import Seq2SeqTrainer

    model = _build_lora_model(config.base_model, config)
    parameter_counts = _describe_trainable_parameters(model)
    training_args = _build_seq2seq_training_arguments(
        config,
        output_dir=run_dir / "trainer",
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_features,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        tokenizer=processor.feature_extractor,
    )
    trainer.add_callback(_VolumeProgressCallback(run_dir=run_dir).impl)
    train_result = trainer.train()
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    return {
        "trainer_metrics": train_result.metrics,
        "parameter_counts": parameter_counts,
        "distributed": {
            "enabled": False,
            "requested_gpu_count": config.distributed_gpu_count,
            "world_size": 1,
            "effective_gradient_accumulation_steps": _effective_gradient_accumulation_steps(config),
            "dataloader_num_workers": _effective_dataloader_num_workers(config),
            "preprocess_num_workers": _effective_preprocess_num_workers(config),
            "preprocess_batch_size": _effective_preprocess_batch_size(config),
        },
    }


def _run_distributed_training_phase(
    config: TrainConfig,
    *,
    processor: Any,
    train_features,
    run_dir: Path,
    adapter_dir: Path,
) -> dict[str, Any]:
    staged_train_features_dir = _stage_prepared_train_features(train_features, run_id=run_dir.name)
    distributed_config_path = run_dir / "distributed_train_config.json"
    distributed_payload = {
        "train_config": asdict(config),
        "prepared_train_features_dir": str(staged_train_features_dir),
        "run_dir": str(run_dir),
        "adapter_dir": str(adapter_dir),
    }
    _write_json(distributed_config_path, distributed_payload)

    command = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={config.distributed_gpu_count}",
        str(_worker_script_path()),
        "--ddp-worker-config-path",
        str(distributed_config_path),
    ]
    subprocess.run(
        command,
        check=True,
        cwd=str(_worker_script_path().parent),
    )

    worker_report_path = run_dir / "distributed_train_result.json"
    if not worker_report_path.exists():
        raise RuntimeError(
            f"Distributed training completed without writing {worker_report_path}"
        )
    worker_report = json.loads(worker_report_path.read_text(encoding="utf-8"))
    worker_report["prepared_train_features_dir"] = str(staged_train_features_dir)
    return worker_report


def _build_partial_train_report(
    config: TrainConfig,
    *,
    run_id: str,
    train_split,
    train_audio_column: str,
    train_text_column: str,
    validation_split,
    validation_source_summary: dict[str, Any],
    train_source_summaries: list[dict[str, Any]],
    training_phase: dict[str, Any],
    adapter_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "base_model": config.base_model,
        "train_dataset": {
            "name": config.train_dataset.name,
            "config": config.train_dataset.config,
            "config_names": config.train_dataset.config_names,
            "audio_column": train_audio_column,
            "text_column": train_text_column,
            "train_samples": len(train_split),
            "validation_samples": len(validation_split),
        },
        "validation_dataset": validation_source_summary,
        "train_sources": train_source_summaries,
        "lora": {
            "rank": config.rank,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "target_modules": _target_modules(config.target_module_set),
            "parameter_counts": training_phase["parameter_counts"],
        },
        "training": {
            "hyperparameters": asdict(config),
            "trainer_metrics": training_phase["trainer_metrics"],
        },
        "runtime": {
            "modal_gpu": os.environ.get("MODAL_GPU_LABEL", "unknown"),
            "attn_implementation": config.attn_implementation,
            "distributed": training_phase["distributed"],
        },
        "artifacts": {
            "adapter_dir": str(adapter_dir),
            "report_path": str(run_dir / "report.json"),
            "train_config_path": str(run_dir / "train_config.json"),
            "distributed_train_result_path": str(run_dir / "distributed_train_result.json")
            if config.distributed_gpu_count > 1
            else None,
        },
    }


def _evaluate_saved_adapter_impl(
    config: TrainConfig,
    *,
    partial_report: dict[str, Any],
) -> dict[str, Any]:
    if config.distributed_gpu_count > 1:
        return _run_parallel_eval_impl(config, partial_report=partial_report)

    from peft import PeftModel

    hf_token = _get_hf_token()
    processor = _build_processor(config.base_model, language=config.language, task=config.task)
    validation_split = None
    validation_audio_column = None
    validation_text_column = None
    if not config.skip_validation_eval:
        split_payload = _resolve_train_validation_splits(config, token=hf_token)
        validation_split = split_payload["validation_split"]
        validation_audio_column = split_payload["validation_audio_column"]
        validation_text_column = split_payload["validation_text_column"]

    svarah_dataset, svarah_audio_column, svarah_text_column = _load_dataset_split(
        DatasetConfig(
            name=config.eval_dataset.name,
            config=config.eval_dataset.config,
            config_names=config.eval_dataset.config_names,
            split=config.eval_dataset.split,
            audio_column=config.eval_dataset.audio_column,
            text_column=config.eval_dataset.text_column,
            max_samples=config.svarah_max_samples or config.eval_dataset.max_samples,
            max_word_count=config.eval_dataset.max_word_count,
            min_duration_seconds=config.eval_dataset.min_duration_seconds,
            max_duration_seconds=config.eval_dataset.max_duration_seconds,
            trust_remote_code=config.eval_dataset.trust_remote_code,
            require_text=config.eval_dataset.require_text,
            metadata_filters=config.eval_dataset.metadata_filters,
        ),
        token=hf_token,
    )

    adapter_dir = Path(partial_report["artifacts"]["adapter_dir"])
    run_dir = adapter_dir.parent
    progress_dir = _run_progress_dir(run_dir)
    _ensure_dir(progress_dir)

    eval_phases: dict[str, Any] = {}
    if validation_split is not None:
        eval_phases["validation"] = {
            "phase_name": "validation",
            "samples_total": len(validation_split),
        }
    eval_phases["svarah-base"] = {
        "phase_name": "svarah base",
        "samples_total": len(svarah_dataset),
    }
    eval_phases["svarah-adapter"] = {
        "phase_name": "svarah adapter",
        "samples_total": len(svarah_dataset),
    }
    _write_run_progress(
        run_dir,
        {
            "stage": "eval",
            "updated_at_utc": _now_iso(),
            "eval": {
                "status": "starting",
                "phases": eval_phases,
            },
        },
        commit=True,
    )

    base_model = _load_base_model(
        config.base_model,
        attn_implementation=config.attn_implementation,
    )
    adapter_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    validation_metrics = None
    if validation_split is not None and validation_audio_column is not None and validation_text_column is not None:
        validation_metrics = _evaluate_model(
            model=adapter_model,
            processor=processor,
            dataset=validation_split,
            audio_column=validation_audio_column,
            text_column=validation_text_column,
            language=config.language,
            task=config.task,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.per_device_eval_batch_size,
            phase_name="validation",
            progress_path=_phase_progress_path(run_dir, "validation"),
        )

    _write_run_progress(
        run_dir,
        {
            "stage": "eval",
            "updated_at_utc": _now_iso(),
            "eval": {
                "status": "running",
                "current_phase": "svarah-base",
                "phases": eval_phases,
            },
        },
        commit=True,
    )
    base_metrics = _evaluate_model(
        model=_load_base_model(
            config.base_model,
            attn_implementation=config.attn_implementation,
        ),
        processor=processor,
        dataset=svarah_dataset,
        audio_column=svarah_audio_column,
        text_column=svarah_text_column,
        language=config.language,
        task=config.task,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.per_device_eval_batch_size,
        phase_name="svarah base",
        progress_path=_phase_progress_path(run_dir, "svarah-base"),
    )
    _write_run_progress(
        run_dir,
        {
            "stage": "eval",
            "updated_at_utc": _now_iso(),
            "eval": {
                "status": "running",
                "current_phase": "svarah-adapter",
                "phases": eval_phases,
            },
        },
        commit=True,
    )
    adapter_metrics = _evaluate_model(
        model=adapter_model,
        processor=processor,
        dataset=svarah_dataset,
        audio_column=svarah_audio_column,
        text_column=svarah_text_column,
        language=config.language,
        task=config.task,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.per_device_eval_batch_size,
        phase_name="svarah adapter",
        progress_path=_phase_progress_path(run_dir, "svarah-adapter"),
    )

    report = dict(partial_report)
    report["eval_dataset"] = {
        "name": config.eval_dataset.name,
        "config": config.eval_dataset.config,
        "config_names": config.eval_dataset.config_names,
        "split": config.eval_dataset.split,
        "audio_column": svarah_audio_column,
        "text_column": svarah_text_column,
        "samples": len(svarah_dataset),
    }
    if validation_metrics is not None:
        report["training"]["validation_metrics"] = validation_metrics
    report["svarah"] = {
        "base_model": base_metrics,
        "adapter": adapter_metrics,
        "delta": {
            "wer": adapter_metrics["wer"] - base_metrics["wer"],
            "cer": adapter_metrics["cer"] - base_metrics["cer"],
        },
    }

    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "train_config.json", asdict(config))
    _write_run_progress(
        run_dir,
        {
            "stage": "complete",
            "updated_at_utc": _now_iso(),
            "eval": {
                "status": "complete",
                "phases": eval_phases,
            },
        },
        commit=True,
    )
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


def _train_and_eval_impl(config: TrainConfig) -> dict[str, Any]:
    hf_token = _get_hf_token()
    run_id = f"{config.experiment_name}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    adapter_dir = run_dir / "adapter"
    _ensure_dir(adapter_dir)
    _write_run_progress(
        run_dir,
        {
            "stage": "startup",
            "updated_at_utc": _now_iso(),
            "experiment_name": config.experiment_name,
            "distributed_gpu_count": config.distributed_gpu_count,
        },
        commit=True,
    )
    _write_run_progress(
        run_dir,
        {
            "stage": "build_processor",
            "updated_at_utc": _now_iso(),
            "base_model": config.base_model,
        },
        commit=True,
    )
    processor = _build_processor(config.base_model, language=config.language, task=config.task)
    split_payload = _resolve_train_validation_splits(config, token=hf_token, run_dir=run_dir)
    train_split = split_payload["train_split"]
    train_audio_column = split_payload["train_audio_column"]
    train_text_column = split_payload["train_text_column"]
    validation_split = split_payload["validation_split"]
    validation_source_summary = split_payload["validation_source_summary"]

    train_features, train_source_summaries = _build_train_features_and_summaries(
        config,
        processor=processor,
        token=hf_token,
        train_split=train_split,
        train_audio_column=train_audio_column,
        train_text_column=train_text_column,
        run_dir=run_dir,
    )

    training_phase = _run_single_gpu_training_phase(
        config,
        processor=processor,
        train_features=train_features,
        run_dir=run_dir,
        adapter_dir=adapter_dir,
    )
    partial_report = _build_partial_train_report(
        config,
        run_id=run_id,
        train_split=train_split,
        train_audio_column=train_audio_column,
        train_text_column=train_text_column,
        validation_split=validation_split,
        validation_source_summary=validation_source_summary,
        train_source_summaries=train_source_summaries,
        training_phase=training_phase,
        adapter_dir=adapter_dir,
        run_dir=run_dir,
    )
    return _evaluate_saved_adapter_impl(config, partial_report=partial_report)


def _benchmark_single_step_impl(config: TrainConfig) -> dict[str, Any]:
    import torch

    hf_token = _get_hf_token()
    effective_gradient_accumulation_steps = _effective_gradient_accumulation_steps(config)
    effective_batch_size = config.per_device_train_batch_size * effective_gradient_accumulation_steps
    required_rows = effective_batch_size * 2

    dataset_started = datetime.now(tz=UTC)
    train_dataset, train_audio_column, train_text_column = _load_dataset_slice(
        config.train_dataset,
        token=hf_token,
        rows=required_rows,
    )
    dataset_loaded = datetime.now(tz=UTC)

    processor_started = dataset_loaded
    processor = _build_processor(config.base_model, language=config.language, task=config.task)
    model = _build_lora_model(config.base_model, config).to("cuda")
    model_loaded = datetime.now(tz=UTC)

    prepared_dataset, _ = _prepare_split(
        train_dataset,
        processor=processor,
        audio_column=train_audio_column,
        text_column=train_text_column,
        map_batch_size=_effective_preprocess_batch_size(config),
        num_proc=_effective_preprocess_num_workers(config),
    )
    features_prepared = datetime.now(tz=UTC)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    rows = [prepared_dataset[index] for index in range(len(prepared_dataset))]
    warmup_rows = rows[:effective_batch_size]
    measured_rows = rows[effective_batch_size : effective_batch_size * 2]

    warmup = _run_optimizer_step(
        model=model,
        optimizer=optimizer,
        collator=collator,
        feature_rows=warmup_rows,
        micro_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=effective_gradient_accumulation_steps,
        device="cuda",
    )
    measured = _run_optimizer_step(
        model=model,
        optimizer=optimizer,
        collator=collator,
        feature_rows=measured_rows,
        micro_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=effective_gradient_accumulation_steps,
        device="cuda",
    )

    hf_cache_volume.commit()

    return {
        "gpu": os.environ.get("MODAL_GPU_LABEL", "unknown"),
        "base_model": config.base_model,
        "dataset": {
            "name": config.train_dataset.name,
            "config": config.train_dataset.config,
            "split": config.train_dataset.split or "train",
            "audio_column": train_audio_column,
            "text_column": train_text_column,
            "loaded_rows": len(train_dataset),
        },
        "training_shape": {
            "micro_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": effective_gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
        },
        "timings_ms": {
            "dataset_load_ms": int((dataset_loaded - dataset_started).total_seconds() * 1000),
            "processor_and_model_load_ms": int((model_loaded - processor_started).total_seconds() * 1000),
            "feature_prepare_ms": int((features_prepared - model_loaded).total_seconds() * 1000),
            "warmup_step_ms": warmup["step_ms"],
            "measured_step_ms": measured["step_ms"],
        },
        "loss_preview": {
            "warmup": warmup["losses"],
            "measured": measured["losses"],
        },
    }


def _train_only_distributed_impl(config: TrainConfig) -> dict[str, Any]:
    hf_token = _get_hf_token()
    run_id = f"{config.experiment_name}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    adapter_dir = run_dir / "adapter"
    _ensure_dir(adapter_dir)
    _write_run_progress(
        run_dir,
        {
            "stage": "startup",
            "updated_at_utc": _now_iso(),
            "experiment_name": config.experiment_name,
            "distributed_gpu_count": config.distributed_gpu_count,
        },
        commit=True,
    )
    _write_run_progress(
        run_dir,
        {
            "stage": "build_processor",
            "updated_at_utc": _now_iso(),
            "base_model": config.base_model,
        },
        commit=True,
    )
    processor = _build_processor(config.base_model, language=config.language, task=config.task)
    split_payload = _resolve_train_validation_splits(config, token=hf_token, run_dir=run_dir)
    train_split = split_payload["train_split"]
    train_audio_column = split_payload["train_audio_column"]
    train_text_column = split_payload["train_text_column"]
    validation_split = split_payload["validation_split"]
    validation_source_summary = split_payload["validation_source_summary"]

    train_features, train_source_summaries = _build_train_features_and_summaries(
        config,
        processor=processor,
        token=hf_token,
        train_split=train_split,
        train_audio_column=train_audio_column,
        train_text_column=train_text_column,
        run_dir=run_dir,
    )

    training_phase = _run_distributed_training_phase(
        config,
        processor=processor,
        train_features=train_features,
        run_dir=run_dir,
        adapter_dir=adapter_dir,
    )
    partial_report = _build_partial_train_report(
        config,
        run_id=run_id,
        train_split=train_split,
        train_audio_column=train_audio_column,
        train_text_column=train_text_column,
        validation_split=validation_split,
        validation_source_summary=validation_source_summary,
        train_source_summaries=train_source_summaries,
        training_phase=training_phase,
        adapter_dir=adapter_dir,
        run_dir=run_dir,
    )
    _write_json(run_dir / "partial_report.json", partial_report)
    _write_json(run_dir / "train_config.json", asdict(config))
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return partial_report


@app.function(
    gpu=TRAIN_GPU,
    timeout=60 * 60 * 8,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def train_and_eval_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = TRAIN_GPU
    config = _normalize_train_config(config_payload)
    return _train_and_eval_impl(config)


@app.function(
    gpu=FOUR_GPU_TRAIN_GPU,
    timeout=60 * 60 * 8,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def train_only_4gpu_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FOUR_GPU_TRAIN_GPU
    config = _normalize_train_config(config_payload)
    if config.distributed_gpu_count != 4:
        config = replace(config, distributed_gpu_count=4)
    return _train_only_distributed_impl(config)


@app.function(
    gpu=FIVE_GPU_TRAIN_GPU,
    timeout=60 * 60 * 8,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def train_only_5gpu_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FIVE_GPU_TRAIN_GPU
    config = _normalize_train_config(config_payload)
    if config.distributed_gpu_count != 5:
        config = replace(config, distributed_gpu_count=5)
    return _train_only_distributed_impl(config)


@app.function(
    gpu=FOUR_GPU_TRAIN_GPU,
    timeout=60 * 60 * 12,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def train_and_eval_4gpu_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FOUR_GPU_TRAIN_GPU
    config = _normalize_train_config(config_payload)
    if config.distributed_gpu_count != 4:
        config = replace(config, distributed_gpu_count=4)
    partial_report = _train_only_distributed_impl(config)
    return _evaluate_saved_adapter_impl(config, partial_report=partial_report)


@app.function(
    gpu=FIVE_GPU_TRAIN_GPU,
    timeout=60 * 60 * 12,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def train_and_eval_5gpu_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FIVE_GPU_TRAIN_GPU
    config = _normalize_train_config(config_payload)
    if config.distributed_gpu_count != 5:
        config = replace(config, distributed_gpu_count=5)
    partial_report = _train_only_distributed_impl(config)
    return _evaluate_saved_adapter_impl(config, partial_report=partial_report)


@app.function(
    gpu=TRAIN_GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def evaluate_saved_adapter_remote(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = TRAIN_GPU
    config = _normalize_train_config(payload["config"])
    partial_report = dict(payload["partial_report"])
    return _evaluate_saved_adapter_impl(config, partial_report=partial_report)


@app.function(
    gpu=FIVE_GPU_TRAIN_GPU,
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def evaluate_saved_run_5gpu_remote(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FIVE_GPU_TRAIN_GPU
    config = _normalize_train_config(payload["config"])
    if config.distributed_gpu_count != 5:
        config = replace(config, distributed_gpu_count=5)
    run_id = str(payload["run_id"]).strip()
    if not run_id:
        raise ValueError("run_id is required")
    partial_report_path = ARTIFACTS_DIR / run_id / "partial_report.json"
    if not partial_report_path.exists():
        raise FileNotFoundError(f"Missing partial report at {partial_report_path}")
    partial_report = json.loads(partial_report_path.read_text(encoding="utf-8"))
    return _evaluate_saved_adapter_impl(config, partial_report=partial_report)


@app.function(
    gpu=TRAIN_GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def evaluate_saved_run_remote(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = TRAIN_GPU
    config = _normalize_train_config(payload["config"])
    run_id = str(payload["run_id"]).strip()
    if not run_id:
        raise ValueError("run_id is required")
    partial_report_path = ARTIFACTS_DIR / run_id / "partial_report.json"
    if not partial_report_path.exists():
        raise FileNotFoundError(f"Missing partial report at {partial_report_path}")
    partial_report = json.loads(partial_report_path.read_text(encoding="utf-8"))
    return _evaluate_saved_adapter_impl(config, partial_report=partial_report)


@app.function(
    gpu="A100",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={str(HF_CACHE_DIR): hf_cache_volume},
)
def benchmark_step_a100_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = "A100"
    config = _normalize_train_config(config_payload)
    return _benchmark_single_step_impl(config)


@app.function(
    gpu="L40S",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={str(HF_CACHE_DIR): hf_cache_volume},
)
def benchmark_step_l40s_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = "L40S"
    config = _normalize_train_config(config_payload)
    return _benchmark_single_step_impl(config)


@app.function(
    gpu=H100_BENCHMARK_GPU,
    timeout=60 * 30,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={str(HF_CACHE_DIR): hf_cache_volume},
)
def benchmark_step_h100_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = H100_BENCHMARK_GPU
    config = _normalize_train_config(config_payload)
    return _benchmark_single_step_impl(config)


@app.function(
    timeout=60 * 20,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def inspect_dataset_remote(dataset_payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_dataset_config(dataset_payload)
    token = _get_hf_token()
    dataset, audio_column, text_column = _load_dataset_split(config, token=token)
    sample = dataset[0] if len(dataset) else {}
    return {
        "name": config.name,
        "config": config.config,
        "split": config.split,
        "rows": len(dataset),
        "columns": dataset.column_names,
        "audio_column": audio_column,
        "text_column": text_column,
        "sample_keys": list(sample.keys()) if sample else [],
    }


@app.function(
    timeout=60 * 30,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def profile_dataset_text_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_dataset_profile_config(config_payload)
    return _profile_text_dataset_impl(config)


@app.function(
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def profile_train_selection_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_train_config(config_payload)
    return _profile_train_selection_impl(config)


@app.function(
    timeout=60 * 60,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def verify_audio_manifest_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_audio_manifest_verify_config(config_payload)
    return _verify_audio_manifest_impl(config)


@app.function(
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def survey_dataset_configs_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_dataset_survey_config(config_payload)
    return _survey_dataset_configs_impl(config)


@app.function(
    timeout=60 * 60,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def build_audit_manifest_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_audit_config(config_payload)
    return _build_audit_manifest_impl(config)


@app.function(
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def build_training_manifest_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_training_manifest_config(config_payload)
    return _build_training_manifest_impl(config)


@app.function(
    timeout=60 * 60 * 6,
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def download_external_archive_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_external_archive_download_config(config_payload)
    return _download_external_archive_impl(config)


@app.function(
    gpu=TRAIN_GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def analyze_svarah_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = TRAIN_GPU
    config = _normalize_analysis_config(config_payload)
    return _analyze_svarah_impl(config)


@app.local_entrypoint()
def main(
    mode: str = "train_eval",
    benchmark_gpu: str = "H100",
    experiment_name: str = "whisper-turbo-india-accent-lora",
    recipe: str = "baseline",
    train_dataset: str = "WillHeld/india_accent_cv",
    train_config_name: str = "",
    train_split: str = "train",
    train_audio_column: str = "",
    train_text_column: str = "",
    train_max_word_count: int = 0,
    train_min_duration_seconds: float = 0.0,
    train_max_duration_seconds: float = 0.0,
    train_trust_remote_code: bool = False,
    train_require_text: bool = False,
    train_metadata_filters: str = "",
    anchor_dataset: str = "",
    anchor_config_name: str = "",
    anchor_split: str = "",
    anchor_audio_column: str = "",
    anchor_text_column: str = "",
    anchor_max_word_count: int = 0,
    anchor_min_duration_seconds: float = 0.0,
    anchor_max_duration_seconds: float = 0.0,
    anchor_trust_remote_code: bool = False,
    anchor_require_text: bool = False,
    anchor_metadata_filters: str = "",
    eval_dataset: str = "ai4bharat/Svarah",
    eval_config_name: str = "",
    eval_split: str = "test",
    eval_audio_column: str = "",
    eval_text_column: str = "",
    eval_max_word_count: int = 0,
    eval_min_duration_seconds: float = 0.0,
    eval_max_duration_seconds: float = 0.0,
    eval_trust_remote_code: bool = False,
    eval_require_text: bool = False,
    eval_metadata_filters: str = "",
    validation_dataset: str = "",
    validation_config_name: str = "",
    validation_split_name: str = "",
    validation_audio_column: str = "",
    validation_text_column: str = "",
    validation_max_word_count: int = 0,
    validation_min_duration_seconds: float = 0.0,
    validation_max_duration_seconds: float = 0.0,
    validation_trust_remote_code: bool = False,
    validation_require_text: bool = False,
    validation_metadata_filters: str = "",
    train_max_samples: int = 0,
    anchor_max_samples: int = 0,
    validation_max_samples: int = 0,
    svarah_max_samples: int = 0,
    audit_name: str = "dataset-audit",
    audit_sample_limit: int = 500,
    audit_seed: int = 42,
    audit_group_fields: str = "primary_language,native_place_state,gender,language,state,district",
    audit_max_samples_per_group: int = 50,
    audit_export_audio: bool = True,
    audit_metadata_fields: str = "",
    audit_normalize_transcripts: bool = False,
    manifest_name: str = "training-manifest",
    manifest_sample_limit: int = 50_000,
    manifest_output_limit: int = 16_384,
    manifest_quality_preset: str = "accent_safe",
    manifest_selection_strategy: str = "score",
    manifest_max_samples_per_speaker: int = 50,
    manifest_export_audio: bool = True,
    manifest_normalize_transcripts: bool = True,
    manifest_metadata_fields: str = "",
    audio_verify_name: str = "audio-manifest-verify",
    survey_name: str = "dataset-config-survey",
    survey_config_regex: str = "",
    survey_config_names: str = "",
    survey_max_configs: int = 0,
    survey_max_workers: int = 6,
    survey_sample_transcripts_per_config: int = 3,
    survey_top_k: int = 25,
    download_name: str = "external-archives",
    download_manifest_path: str = "",
    download_target_subdir: str = "downloads",
    download_overwrite: bool = False,
    download_timeout_seconds: int = 120,
    num_train_epochs: float = 3.0,
    learning_rate: float = 1e-4,
    rank: int = 64,
    alpha: int = 32,
    dropout: float = 0.05,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    target_module_set: str = "full",
    lora_scope: str = "all",
    normalize_transcripts: bool = False,
    compare_run_ids: str = "",
    compare_labels: str = "",
    analysis_name: str = "svarah-analysis",
    analysis_group_fields: str = "duration_bucket,gender,age-group,primary_language,native_place_state,occupation_domain",
    analysis_filters: str = "",
    analysis_min_group_samples: int = 100,
    analysis_max_groups_per_field: int = 12,
    analysis_top_examples: int = 5,
    profile_sample_limit: int = 0,
    profile_short_word_threshold: int = 5,
    existing_run_id: str = "",
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    dataloader_num_workers: int = 0,
    preprocess_num_workers: int = 0,
    preprocess_batch_size: int = 8,
    distributed_gpu_count: int = 1,
    gradient_checkpointing: bool = True,
    ddp_find_unused_parameters: bool = False,
    skip_validation_eval: bool = False,
    optim: str = "adamw_torch_fused",
) -> None:
    train_config_names = _parse_config_names(train_config_name)
    eval_config_names = _parse_config_names(eval_config_name)
    validation_config_names = _parse_config_names(validation_config_name)
    anchor_config_names = _parse_config_names(anchor_config_name)

    train_dataset_config = DatasetConfig(
        name=train_dataset,
        config=train_config_names[0] if len(train_config_names) == 1 else None,
        config_names=train_config_names if len(train_config_names) > 1 else [],
        split=train_split or None,
        audio_column=train_audio_column or None,
        text_column=train_text_column or None,
        max_word_count=train_max_word_count or None,
        min_duration_seconds=train_min_duration_seconds or None,
        max_duration_seconds=train_max_duration_seconds or None,
        trust_remote_code=train_trust_remote_code,
        require_text=train_require_text,
        metadata_filters=_parse_metadata_filters(train_metadata_filters),
    )
    eval_dataset_config = DatasetConfig(
        name=eval_dataset,
        config=eval_config_names[0] if len(eval_config_names) == 1 else None,
        config_names=eval_config_names if len(eval_config_names) > 1 else [],
        split=eval_split or None,
        audio_column=eval_audio_column or None,
        text_column=eval_text_column or None,
        max_samples=svarah_max_samples or None,
        max_word_count=eval_max_word_count or None,
        min_duration_seconds=eval_min_duration_seconds or None,
        max_duration_seconds=eval_max_duration_seconds or None,
        trust_remote_code=eval_trust_remote_code,
        require_text=eval_require_text,
        metadata_filters=_parse_metadata_filters(eval_metadata_filters),
    )
    validation_dataset_config = None
    if validation_dataset:
        validation_dataset_config = DatasetConfig(
            name=validation_dataset,
            config=validation_config_names[0] if len(validation_config_names) == 1 else None,
            config_names=validation_config_names if len(validation_config_names) > 1 else [],
            split=validation_split_name or None,
            audio_column=validation_audio_column or None,
            text_column=validation_text_column or None,
            max_word_count=validation_max_word_count or None,
            min_duration_seconds=validation_min_duration_seconds or None,
            max_duration_seconds=validation_max_duration_seconds or None,
            trust_remote_code=validation_trust_remote_code,
            require_text=validation_require_text,
            metadata_filters=_parse_metadata_filters(validation_metadata_filters),
        )
    anchor_dataset_config = None
    if anchor_dataset:
        anchor_dataset_config = DatasetConfig(
            name=anchor_dataset,
            config=anchor_config_names[0] if len(anchor_config_names) == 1 else None,
            config_names=anchor_config_names if len(anchor_config_names) > 1 else [],
            split=anchor_split or None,
            audio_column=anchor_audio_column or None,
            text_column=anchor_text_column or None,
            max_word_count=anchor_max_word_count or None,
            min_duration_seconds=anchor_min_duration_seconds or None,
            max_duration_seconds=anchor_max_duration_seconds or None,
            trust_remote_code=anchor_trust_remote_code,
            require_text=anchor_require_text,
            metadata_filters=_parse_metadata_filters(anchor_metadata_filters),
        )

    if mode == "inspect_train":
        payload = inspect_dataset_remote.remote(asdict(train_dataset_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "inspect_eval":
        payload = inspect_dataset_remote.remote(asdict(eval_dataset_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "profile_train":
        payload = profile_dataset_text_remote.remote(
            asdict(
                DatasetProfileConfig(
                    dataset=train_dataset_config,
                    sample_limit=profile_sample_limit or None,
                    short_word_threshold=profile_short_word_threshold,
                )
            )
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "profile_anchor":
        if anchor_dataset_config is None:
            raise ValueError("anchor_dataset is required for profile_anchor mode")
        payload = profile_dataset_text_remote.remote(
            asdict(
                DatasetProfileConfig(
                    dataset=anchor_dataset_config,
                    sample_limit=profile_sample_limit or None,
                    short_word_threshold=profile_short_word_threshold,
                )
            )
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "survey_train_configs":
        selected_config_names = [value.strip() for value in survey_config_names.split(",") if value.strip()]
        if train_dataset_config.config and not selected_config_names:
            selected_config_names = [train_dataset_config.config]
        survey_config = DatasetConfigSurveyConfig(
            survey_name=survey_name,
            dataset=replace(train_dataset_config, config=None),
            config_name_regex=survey_config_regex,
            config_names=selected_config_names,
            max_configs=survey_max_configs or None,
            max_workers=survey_max_workers,
            sample_transcripts_per_config=survey_sample_transcripts_per_config,
            top_k=survey_top_k,
        )
        payload = survey_dataset_configs_remote.remote(asdict(survey_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "download_external_archives":
        if not download_manifest_path:
            raise ValueError("download_manifest_path is required for download_external_archives mode")
        manifest_path = Path(download_manifest_path).expanduser()
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_payload, list) or not manifest_payload:
            raise ValueError("download manifest must be a non-empty JSON array of {filename, url} objects")
        configs = [
            ExternalArchiveDownloadConfig(
                download_name=download_name,
                target_subdir=download_target_subdir,
                overwrite=download_overwrite,
                timeout_seconds=download_timeout_seconds,
                archive=ExternalArchive(
                    filename=str(item["filename"]),
                    url=str(item["url"]),
                ),
            )
            for item in manifest_payload
        ]
        payload = list(download_external_archive_remote.map([asdict(config) for config in configs]))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "build_audit_manifest":
        audit_config = AuditConfig(
            audit_name=audit_name,
            dataset=train_dataset_config,
            sample_limit=audit_sample_limit,
            seed=audit_seed,
            stratify_fields=[field.strip() for field in audit_group_fields.split(",") if field.strip()],
            max_samples_per_group=audit_max_samples_per_group or None,
            export_audio=audit_export_audio,
            metadata_fields=[field.strip() for field in audit_metadata_fields.split(",") if field.strip()],
            normalize_transcripts=audit_normalize_transcripts,
        )
        payload = build_audit_manifest_remote.remote(asdict(audit_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "build_training_manifest":
        manifest_config = TrainingManifestConfig(
            manifest_name=manifest_name,
            dataset=train_dataset_config,
            sample_limit=manifest_sample_limit,
            output_limit=manifest_output_limit,
            seed=audit_seed,
            quality_preset=manifest_quality_preset,
            selection_strategy=manifest_selection_strategy,
            max_samples_per_speaker=manifest_max_samples_per_speaker,
            export_audio=manifest_export_audio,
            normalize_transcripts=manifest_normalize_transcripts,
            metadata_fields=[field.strip() for field in manifest_metadata_fields.split(",") if field.strip()],
        )
        payload = build_training_manifest_remote.remote(asdict(manifest_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "verify_audio_manifest":
        verify_config = AudioManifestVerifyConfig(
            verify_name=audio_verify_name,
            dataset=train_dataset_config,
            sample_limit=profile_sample_limit or None,
            seed=audit_seed,
        )
        payload = verify_audio_manifest_remote.remote(asdict(verify_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "analyze_svarah":
        run_ids = [value.strip() for value in compare_run_ids.split(",") if value.strip()]
        labels = [value.strip() for value in compare_labels.split(",") if value.strip()]
        if not run_ids:
            raise ValueError("compare_run_ids is required for analyze_svarah mode")
        if labels and len(labels) != len(run_ids):
            raise ValueError("compare_labels must match compare_run_ids length when provided")

        adapter_runs = []
        for index, run_id in enumerate(run_ids):
            label = labels[index] if labels else f"adapter_{index + 1}"
            adapter_runs.append(
                {
                    "label": label,
                    "run_id": run_id,
                    "adapter_dir": str(ARTIFACTS_DIR / run_id / "adapter"),
                }
            )

        row_filters = {}
        for item in [value.strip() for value in analysis_filters.split(",") if value.strip()]:
            if "=" not in item:
                raise ValueError(
                    "analysis_filters entries must use key=value format, for example contains_digit=no"
                )
            key, value = item.split("=", maxsplit=1)
            row_filters[key.strip()] = value.strip()

        analysis_config = AnalysisConfig(
            analysis_name=analysis_name,
            base_model=BASE_MODEL,
            attn_implementation=attn_implementation,
            eval_dataset=eval_dataset_config,
            adapter_runs=adapter_runs,
            max_new_tokens=256,
            per_device_eval_batch_size=per_device_eval_batch_size,
            min_group_samples=analysis_min_group_samples,
            max_groups_per_field=analysis_max_groups_per_field,
            top_examples=analysis_top_examples,
            row_filters=row_filters,
            group_fields=[field.strip() for field in analysis_group_fields.split(",") if field.strip()],
        )
        payload = analyze_svarah_remote.remote(asdict(analysis_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "benchmark_step":
        config = TrainConfig(
            experiment_name=experiment_name,
            recipe=recipe,
            train_dataset=train_dataset_config,
            eval_dataset=eval_dataset_config,
            validation_dataset=validation_dataset_config,
            anchor_dataset=anchor_dataset_config,
            train_max_samples=train_max_samples or None,
            anchor_max_samples=anchor_max_samples or None,
            validation_max_samples=validation_max_samples or None,
            svarah_max_samples=svarah_max_samples or None,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            attn_implementation=attn_implementation,
            target_module_set=target_module_set,
            lora_scope=lora_scope,
            normalize_transcripts=normalize_transcripts,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            preprocess_num_workers=preprocess_num_workers,
            preprocess_batch_size=preprocess_batch_size,
            distributed_gpu_count=distributed_gpu_count,
            gradient_checkpointing=gradient_checkpointing,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
            skip_validation_eval=skip_validation_eval,
            optim=optim,
        )
        normalized_gpu = benchmark_gpu.upper()
        benchmark_function = {
            "A100": benchmark_step_a100_remote,
            "L40S": benchmark_step_l40s_remote,
            "H100": benchmark_step_h100_remote,
        }.get(normalized_gpu)
        if benchmark_function is None:
            raise ValueError(
                f"Unsupported benchmark GPU '{benchmark_gpu}'. Expected one of: A100, L40S, H100"
            )
        payload = benchmark_function.remote(asdict(config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    config = TrainConfig(
        experiment_name=experiment_name,
        recipe=recipe,
        train_dataset=train_dataset_config,
        eval_dataset=eval_dataset_config,
        validation_dataset=validation_dataset_config,
        anchor_dataset=anchor_dataset_config,
        train_max_samples=train_max_samples or None,
        anchor_max_samples=anchor_max_samples or None,
        validation_max_samples=validation_max_samples or None,
        svarah_max_samples=svarah_max_samples or None,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        attn_implementation=attn_implementation,
        target_module_set=target_module_set,
        lora_scope=lora_scope,
        normalize_transcripts=normalize_transcripts,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        preprocess_num_workers=preprocess_num_workers,
        preprocess_batch_size=preprocess_batch_size,
        distributed_gpu_count=distributed_gpu_count,
        gradient_checkpointing=gradient_checkpointing,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        skip_validation_eval=skip_validation_eval,
        optim=optim,
    )

    if mode == "train_only":
        if distributed_gpu_count == 4:
            partial_report = train_only_4gpu_remote.remote(asdict(config))
        elif distributed_gpu_count == 5:
            partial_report = train_only_5gpu_remote.remote(asdict(config))
        else:
            raise ValueError("train_only mode currently supports distributed_gpu_count=4 or 5 only")
        print(json.dumps(partial_report, indent=2, sort_keys=True))
        return

    if mode == "evaluate_saved_run":
        if not existing_run_id:
            raise ValueError("existing_run_id is required for evaluate_saved_run mode")
        payload = {
            "config": asdict(config),
            "run_id": existing_run_id,
        }
        if distributed_gpu_count == 5:
            report = evaluate_saved_run_5gpu_remote.remote(payload)
        else:
            report = evaluate_saved_run_remote.remote(payload)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if mode == "profile_train_selection":
        report = profile_train_selection_remote.remote(asdict(config))
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if mode != "train_eval":
        raise ValueError(
            "Unsupported mode "
            f"'{mode}'. Expected one of: train_eval, train_only, evaluate_saved_run, profile_train_selection, inspect_train, inspect_eval, profile_train, profile_anchor, survey_train_configs, download_external_archives, build_audit_manifest, build_training_manifest, verify_audio_manifest, benchmark_step, analyze_svarah"
        )

    if distributed_gpu_count == 4:
        report = train_and_eval_4gpu_remote.remote(asdict(config))
    elif distributed_gpu_count == 5:
        report = train_and_eval_5gpu_remote.remote(asdict(config))
    elif distributed_gpu_count == 1:
        report = train_and_eval_remote.remote(asdict(config))
    else:
        raise ValueError(
            f"Unsupported distributed_gpu_count '{distributed_gpu_count}'. Expected one of: 1, 4, 5"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__" and "--ddp-worker-config-path" in sys.argv:
    worker_args = sys.argv[sys.argv.index("--ddp-worker-config-path") + 1 :]
    if not worker_args:
        raise SystemExit("--ddp-worker-config-path requires a value")
    _distributed_worker_main(worker_args[0])
    raise SystemExit(0)


if __name__ == "__main__" and "--eval-worker-config-path" in sys.argv:
    worker_args = sys.argv[sys.argv.index("--eval-worker-config-path") + 1 :]
    if not worker_args:
        raise SystemExit("--eval-worker-config-path requires a value")
    _eval_worker_main(worker_args[0])
    raise SystemExit(0)
