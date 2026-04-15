"""Modal-native Whisper LoRA training + Svarah evaluation workflow.

This script is intended for one narrow experiment:

1. LoRA fine-tune `openai/whisper-large-v3-turbo` on Indian-accent English.
2. Evaluate the resulting adapter against AI4Bharat Svarah.

The default training dataset is `WillHeld/india_accent_cv` because it is public
and explicitly targets Indian-accent English. Svarah is used as the external
evaluation set and requires Hugging Face access approval.
"""

from __future__ import annotations

import json
import os
import re
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
H100_BENCHMARK_GPU = os.environ.get("LOCALWISPR_MODAL_LORA_H100_GPU", "H100!")
DEFAULT_ATTN_IMPLEMENTATION = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ATTN_IMPLEMENTATION", "sdpa"
)

ARTIFACTS_DIR = Path("/artifacts")
HF_CACHE_DIR = Path("/cache/huggingface")

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
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
    split: str | None = None
    audio_column: str | None = None
    text_column: str | None = None
    max_samples: int | None = None


@dataclass
class TrainConfig:
    experiment_name: str
    recipe: str = "baseline"
    base_model: str = BASE_MODEL
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
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
    seed: int = 42
    train_validation_split: float = 0.1
    max_new_tokens: int = 256
    logging_steps: int = 10
    save_total_limit: int = 2
    train_max_samples: int | None = None
    anchor_max_samples: int | None = None
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


def _normalize_dataset_config(value: DatasetConfig | dict[str, Any]) -> DatasetConfig:
    if isinstance(value, DatasetConfig):
        return value
    return DatasetConfig(**value)


def _normalize_train_config(value: TrainConfig | dict[str, Any]) -> TrainConfig:
    if isinstance(value, TrainConfig):
        return _apply_recipe_defaults(value)

    payload = dict(value)
    payload["train_dataset"] = _normalize_dataset_config(payload["train_dataset"])
    payload["eval_dataset"] = _normalize_dataset_config(payload["eval_dataset"])
    if payload.get("anchor_dataset") is not None:
        payload["anchor_dataset"] = _normalize_dataset_config(payload["anchor_dataset"])
    return _apply_recipe_defaults(TrainConfig(**payload))


def _normalize_analysis_config(value: AnalysisConfig | dict[str, Any]) -> AnalysisConfig:
    if isinstance(value, AnalysisConfig):
        return value

    payload = dict(value)
    payload["eval_dataset"] = _normalize_dataset_config(payload["eval_dataset"])
    payload["adapter_runs"] = [dict(item) for item in payload.get("adapter_runs", [])]
    payload["group_fields"] = list(payload.get("group_fields", []))
    return AnalysisConfig(**payload)


def _apply_recipe_defaults(config: TrainConfig) -> TrainConfig:
    recipe = config.recipe.strip().lower()
    if recipe in ("", "baseline"):
        return config

    if recipe != "mixed-anchor-v1":
        raise ValueError(
            f"Unsupported recipe '{config.recipe}'. Expected one of: baseline, mixed-anchor-v1"
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


def _load_dataset_split(config: DatasetConfig, *, token: str | None):
    from datasets import Audio, Dataset, DatasetDict, load_dataset

    if config.split:
        dataset_or_dict = load_dataset(config.name, config.config, split=config.split, token=token)
    else:
        dataset_or_dict = load_dataset(config.name, config.config, token=token)
    if isinstance(dataset_or_dict, DatasetDict):
        split = _resolve_split(dataset_or_dict, config.split)
        dataset = dataset_or_dict[split]
    else:
        dataset = dataset_or_dict

    audio_column = config.audio_column or _infer_audio_column(dataset.features)
    text_column = config.text_column or _infer_text_column(dataset.features)
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16_000))

    if config.max_samples is not None:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    return dataset, audio_column, text_column


def _sample_dataset_rows(dataset, *, max_samples: int | None, seed: int):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(max_samples))


def _load_dataset_slice(config: DatasetConfig, *, token: str | None, rows: int):
    from datasets import Audio, load_dataset

    split = config.split or "train"
    sliced_split = f"{split}[:{rows}]"
    dataset = load_dataset(config.name, config.config, split=sliced_split, token=token)
    audio_column = config.audio_column or _infer_audio_column(dataset.features)
    text_column = config.text_column or _infer_text_column(dataset.features)
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16_000))
    return dataset, audio_column, text_column


def _build_processor(model_name: str, *, language: str, task: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(language=language, task=task)
    return processor


def _prepare_split(
    dataset,
    *,
    processor: Any,
    audio_column: str,
    text_column: str,
    normalize_transcripts: bool = False,
):
    transcript_normalizer = None
    if normalize_transcripts:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        transcript_normalizer = BasicTextNormalizer()

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch[audio_column]
        transcript = str(batch[text_column])
        if transcript_normalizer is not None:
            transcript = transcript_normalizer(transcript).strip()
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = processor.tokenizer(transcript).input_ids
        return batch

    return dataset.map(
        prepare_batch,
        remove_columns=dataset.column_names,
    )


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

        pred_str = [normalizer(text).strip() for text in pred_str]
        label_str = [normalizer(text).strip() for text in label_str]

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
            "samples": len(pred_str),
        }

    return compute_metrics


def _target_modules() -> list[str]:
    # Following the LoRA guidance in the Thinking Machines post, we cover both
    # attention and MLP projections rather than attention-only adapters.
    return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


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
        target_modules=_target_modules(),
    )
    model = get_peft_model(model, lora_config)
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
) -> dict[str, Any]:
    import evaluate
    import torch
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    normalizer = BasicTextNormalizer()

    model = model.to("cuda")
    model.eval()

    predictions: list[str] = []
    references: list[str] = []

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, stop))
        audios = [row[audio_column] for row in batch]
        refs = [str(row[text_column]) for row in batch]
        inputs = processor.feature_extractor(
            [audio["array"] for audio in audios],
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

    normalized_predictions = [normalizer(text).strip() for text in predictions]
    normalized_references = [normalizer(text).strip() for text in references]

    return {
        "samples": len(predictions),
        "wer": wer_metric.compute(predictions=normalized_predictions, references=normalized_references),
        "cer": cer_metric.compute(predictions=normalized_predictions, references=normalized_references),
        "preview": [
            {
                "reference": references[index],
                "prediction": predictions[index],
            }
            for index in range(min(5, len(predictions)))
        ],
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
) -> dict[str, Any]:
    import torch
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    normalizer = BasicTextNormalizer()

    model = model.to("cuda")
    model.eval()

    predictions: list[str] = []
    references: list[str] = []
    normalized_predictions: list[str] = []
    normalized_references: list[str] = []

    metadata_dataset = dataset.remove_columns([audio_column])
    metadata_rows = [metadata_dataset[index] for index in range(len(metadata_dataset))]

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, stop))
        audios = [row[audio_column] for row in batch]
        refs = [str(row[text_column]) for row in batch]
        inputs = processor.feature_extractor(
            [audio["array"] for audio in audios],
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
        normalized_predictions.extend(normalizer(text).strip() for text in preds)
        normalized_references.extend(normalizer(text).strip() for text in refs)

    return {
        "predictions": predictions,
        "references": references,
        "normalized_predictions": normalized_predictions,
        "normalized_references": normalized_references,
        "metadata_rows": metadata_rows,
    }


def _duration_bucket(duration_seconds: float) -> str:
    if duration_seconds < 3:
        return "<3s"
    if duration_seconds < 6:
        return "3-6s"
    if duration_seconds < 10:
        return "6-10s"
    return "10s+"


def _word_count_bucket(text: str) -> str:
    word_count = len([token for token in text.split() if token.strip()])
    if word_count <= 2:
        return "1-2 words"
    if word_count <= 5:
        return "3-5 words"
    if word_count <= 10:
        return "6-10 words"
    return "11+ words"


def _group_value(row: dict[str, Any], field_name: str) -> str:
    raw_text = str(row.get("text") or "")
    lowered_text = raw_text.lower()
    if field_name == "duration_bucket":
        duration = float(row.get("duration") or 0.0)
        return _duration_bucket(duration)
    if field_name == "word_count_bucket":
        return _word_count_bucket(raw_text)
    if field_name == "contains_digit":
        return "yes" if any(character.isdigit() for character in raw_text) else "no"
    if field_name == "contains_date_like":
        patterns = (
            r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
            r"\bdate of birth\b",
            r"\bdob\b",
        )
        return "yes" if any(re.search(pattern, raw_text) for pattern in patterns) else "no"
    if field_name == "contains_currency_or_amount":
        patterns = (
            r"\b(rs|rupees|inr|usd|dollars?)\b",
            r"₹",
            r"\brefund\b",
            r"\bamount\b",
        )
        return "yes" if any(re.search(pattern, lowered_text) for pattern in patterns) else "no"
    value = row.get(field_name)
    if value is None:
        return "unknown"
    value = str(value).strip()
    return value or "unknown"


def _compute_text_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    from jiwer import cer, wer

    return {
        "wer": wer(references, predictions),
        "cer": cer(references, predictions),
    }


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


def _analyze_svarah_impl(config: AnalysisConfig) -> dict[str, Any]:
    import torch
    from peft import PeftModel

    hf_token = _get_hf_token()
    processor = _build_processor(config.base_model, language=config.language, task=config.task)
    dataset, audio_column, text_column = _load_dataset_split(config.eval_dataset, token=hf_token)
    analysis_run_id = f"{config.analysis_name}-{_now_utc()}"
    analysis_dir = ARTIFACTS_DIR / analysis_run_id
    _ensure_dir(analysis_dir)

    predictions_by_model: dict[str, dict[str, Any]] = {}

    print(f"Starting analysis run {analysis_run_id} on {len(dataset)} samples")
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
    )
    del base_model
    torch.cuda.empty_cache()

    for adapter_run in config.adapter_runs:
        label = adapter_run["label"]
        adapter_dir = Path(adapter_run["adapter_dir"])
        print(f"Evaluating adapter {label} from {adapter_dir}")
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
        )
        del adapter_model
        del base_model
        torch.cuda.empty_cache()

    metadata_rows = predictions_by_model["base"]["metadata_rows"]
    normalized_references = predictions_by_model["base"]["normalized_references"]
    raw_references = predictions_by_model["base"]["references"]
    normalized_predictions_by_model = {
        label: payload["normalized_predictions"] for label, payload in predictions_by_model.items()
    }
    raw_predictions_by_model = {
        label: payload["predictions"] for label, payload in predictions_by_model.items()
    }

    overall_metrics = {
        label: _compute_text_metrics(normalized_references, payload["normalized_predictions"])
        for label, payload in predictions_by_model.items()
    }

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
        },
        "overall_metrics": overall_metrics,
        "group_metrics": _summarize_group_metrics(
            metadata_rows=metadata_rows,
            normalized_references=normalized_references,
            normalized_predictions_by_model=normalized_predictions_by_model,
            group_fields=config.group_fields,
            min_group_samples=config.min_group_samples,
            max_groups_per_field=config.max_groups_per_field,
        ),
        "pairwise_vs_base": _summarize_pairwise_examples(
            metadata_rows=metadata_rows,
            raw_references=raw_references,
            raw_predictions_by_model=raw_predictions_by_model,
            normalized_references=normalized_references,
            normalized_predictions_by_model=normalized_predictions_by_model,
            top_examples=config.top_examples,
        ),
        "adapter_runs": config.adapter_runs,
        "artifacts": {
            "report_path": str(analysis_dir / "report.json"),
        },
    }
    _write_json(analysis_dir / "report.json", report)
    artifacts_volume.commit()
    hf_cache_volume.commit()
    print(f"Finished analysis run {analysis_run_id}")
    return report


def _train_and_eval_impl(config: TrainConfig) -> dict[str, Any]:
    import torch
    from datasets import concatenate_datasets
    from peft import PeftModel
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    hf_token = _get_hf_token()
    processor = _build_processor(config.base_model, language=config.language, task=config.task)

    train_dataset, train_audio_column, train_text_column = _load_dataset_split(
        config.train_dataset,
        token=hf_token,
    )
    train_validation = train_dataset.train_test_split(
        test_size=config.train_validation_split,
        seed=config.seed,
    )
    train_split = train_validation["train"]
    validation_split = train_validation["test"]

    train_split = _sample_dataset_rows(train_split, max_samples=config.train_max_samples, seed=config.seed)
    validation_split = _sample_dataset_rows(
        validation_split,
        max_samples=config.validation_max_samples,
        seed=config.seed + 1,
    )

    train_sources = []
    train_source_summaries = [
        {
            "role": "primary",
            "name": config.train_dataset.name,
            "config": config.train_dataset.config,
            "split": config.train_dataset.split,
            "audio_column": train_audio_column,
            "text_column": train_text_column,
            "samples": len(train_split),
        }
    ]

    primary_train_features = _prepare_split(
        train_split,
        processor=processor,
        audio_column=train_audio_column,
        text_column=train_text_column,
        normalize_transcripts=config.normalize_transcripts,
    )
    train_sources.append(primary_train_features)

    if config.anchor_dataset is not None:
        anchor_dataset, anchor_audio_column, anchor_text_column = _load_dataset_split(
            config.anchor_dataset,
            token=hf_token,
        )
        anchor_dataset = _sample_dataset_rows(
            anchor_dataset,
            max_samples=config.anchor_max_samples,
            seed=config.seed + 2,
        )
        anchor_features = _prepare_split(
            anchor_dataset,
            processor=processor,
            audio_column=anchor_audio_column,
            text_column=anchor_text_column,
            normalize_transcripts=config.normalize_transcripts,
        )
        train_sources.append(anchor_features)
        train_source_summaries.append(
            {
                "role": "anchor",
                "name": config.anchor_dataset.name,
                "config": config.anchor_dataset.config,
                "split": config.anchor_dataset.split,
                "audio_column": anchor_audio_column,
                "text_column": anchor_text_column,
                "samples": len(anchor_dataset),
            }
        )

    if len(train_sources) == 1:
        train_features = train_sources[0].shuffle(seed=config.seed)
    else:
        train_features = concatenate_datasets(train_sources).shuffle(seed=config.seed)

    model = _build_lora_model(config.base_model, config)
    parameter_counts = _describe_trainable_parameters(model)

    run_id = f"{config.experiment_name}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    adapter_dir = run_dir / "adapter"
    _ensure_dir(adapter_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "trainer"),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_features,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        tokenizer=processor.feature_extractor,
    )

    train_result = trainer.train()

    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    svarah_dataset, svarah_audio_column, svarah_text_column = _load_dataset_split(
        DatasetConfig(
            name=config.eval_dataset.name,
            config=config.eval_dataset.config,
            split=config.eval_dataset.split,
            audio_column=config.eval_dataset.audio_column,
            text_column=config.eval_dataset.text_column,
            max_samples=config.svarah_max_samples or config.eval_dataset.max_samples,
        ),
        token=hf_token,
    )

    base_model = _load_base_model(
        config.base_model,
        attn_implementation=config.attn_implementation,
    )
    adapter_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    validation_metrics = _evaluate_model(
        model=adapter_model,
        processor=processor,
        dataset=validation_split,
        audio_column=train_audio_column,
        text_column=train_text_column,
        language=config.language,
        task=config.task,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.per_device_eval_batch_size,
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
    )

    report = {
        "run_id": run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "base_model": config.base_model,
        "train_dataset": {
            "name": config.train_dataset.name,
            "config": config.train_dataset.config,
            "audio_column": train_audio_column,
            "text_column": train_text_column,
            "train_samples": len(train_split),
            "validation_samples": len(validation_split),
        },
        "train_sources": train_source_summaries,
        "eval_dataset": {
            "name": config.eval_dataset.name,
            "config": config.eval_dataset.config,
            "split": config.eval_dataset.split,
            "audio_column": svarah_audio_column,
            "text_column": svarah_text_column,
            "samples": len(svarah_dataset),
        },
        "lora": {
            "rank": config.rank,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "target_modules": _target_modules(),
            "parameter_counts": parameter_counts,
        },
        "training": {
            "hyperparameters": asdict(config),
            "trainer_metrics": train_result.metrics,
            "validation_metrics": validation_metrics,
        },
        "runtime": {
            "modal_gpu": os.environ.get("MODAL_GPU_LABEL", "unknown"),
            "attn_implementation": config.attn_implementation,
        },
        "svarah": {
            "base_model": base_metrics,
            "adapter": adapter_metrics,
            "delta": {
                "wer": adapter_metrics["wer"] - base_metrics["wer"],
                "cer": adapter_metrics["cer"] - base_metrics["cer"],
            },
        },
        "artifacts": {
            "adapter_dir": str(adapter_dir),
            "report_path": str(run_dir / "report.json"),
        },
    }

    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "train_config.json", asdict(config))
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


def _benchmark_single_step_impl(config: TrainConfig) -> dict[str, Any]:
    import torch

    hf_token = _get_hf_token()
    effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
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

    prepared_dataset = _prepare_split(
        train_dataset,
        processor=processor,
        audio_column=train_audio_column,
        text_column=train_text_column,
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
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        device="cuda",
    )
    measured = _run_optimizer_step(
        model=model,
        optimizer=optimizer,
        collator=collator,
        feature_rows=measured_rows,
        micro_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
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
    volumes={str(HF_CACHE_DIR): hf_cache_volume},
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
    anchor_dataset: str = "",
    anchor_config_name: str = "",
    anchor_split: str = "",
    anchor_audio_column: str = "",
    anchor_text_column: str = "",
    eval_dataset: str = "ai4bharat/Svarah",
    eval_config_name: str = "",
    eval_split: str = "test",
    eval_audio_column: str = "",
    eval_text_column: str = "",
    train_max_samples: int = 0,
    anchor_max_samples: int = 0,
    validation_max_samples: int = 0,
    svarah_max_samples: int = 0,
    num_train_epochs: float = 3.0,
    learning_rate: float = 1e-4,
    rank: int = 64,
    alpha: int = 32,
    dropout: float = 0.05,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    normalize_transcripts: bool = False,
    compare_run_ids: str = "",
    compare_labels: str = "",
    analysis_name: str = "svarah-analysis",
    analysis_group_fields: str = "duration_bucket,gender,age-group,primary_language,native_place_state,occupation_domain",
    analysis_min_group_samples: int = 100,
    analysis_max_groups_per_field: int = 12,
    analysis_top_examples: int = 5,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
) -> None:
    train_dataset_config = DatasetConfig(
        name=train_dataset,
        config=train_config_name or None,
        split=train_split or None,
        audio_column=train_audio_column or None,
        text_column=train_text_column or None,
    )
    eval_dataset_config = DatasetConfig(
        name=eval_dataset,
        config=eval_config_name or None,
        split=eval_split or None,
        audio_column=eval_audio_column or None,
        text_column=eval_text_column or None,
    )
    anchor_dataset_config = None
    if anchor_dataset:
        anchor_dataset_config = DatasetConfig(
            name=anchor_dataset,
            config=anchor_config_name or None,
            split=anchor_split or None,
            audio_column=anchor_audio_column or None,
            text_column=anchor_text_column or None,
        )

    if mode == "inspect_train":
        payload = inspect_dataset_remote.remote(asdict(train_dataset_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if mode == "inspect_eval":
        payload = inspect_dataset_remote.remote(asdict(eval_dataset_config))
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
            normalize_transcripts=normalize_transcripts,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
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

    if mode != "train_eval":
        raise ValueError(
            "Unsupported mode "
            f"'{mode}'. Expected one of: train_eval, inspect_train, inspect_eval, benchmark_step, analyze_svarah"
        )

    config = TrainConfig(
        experiment_name=experiment_name,
        recipe=recipe,
        train_dataset=train_dataset_config,
        eval_dataset=eval_dataset_config,
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
        normalize_transcripts=normalize_transcripts,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    report = train_and_eval_remote.remote(asdict(config))
    print(json.dumps(report, indent=2, sort_keys=True))
