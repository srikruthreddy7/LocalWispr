"""Modal Svarah ASR bakeoff for Whisper, Cohere Transcribe, and Parakeet.

This is intentionally separate from ``modal_whisper_lora_experiment.py``.
The LoRA trainer is pinned to an older Transformers stack, while Cohere
Transcribe and NVIDIA Parakeet need newer/different runtime dependencies.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LOCALWISPR_MODAL_ASR_BAKEOFF_APP_NAME", "localwispr-asr-bakeoff")
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
HF_CACHE_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_HF_CACHE_VOLUME", "localwispr-hf-cache"
)
HF_SECRET_NAME = os.environ.get("LOCALWISPR_MODAL_LORA_HF_SECRET_NAME", "huggingface-secret")
TRAIN_GPU = os.environ.get("LOCALWISPR_MODAL_LORA_TRAIN_GPU", "H100!")
FIVE_GPU = f"{TRAIN_GPU}:5"

ARTIFACTS_DIR = Path("/artifacts")
HF_CACHE_DIR = Path("/cache/huggingface")

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

COMMON_ENV = {
    "HF_HOME": str(HF_CACHE_DIR),
    "HF_DATASETS_CACHE": str(HF_CACHE_DIR / "datasets"),
    "TRANSFORMERS_CACHE": str(HF_CACHE_DIR / "transformers"),
    "HF_HUB_ETAG_TIMEOUT": "30",
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    "TOKENIZERS_PARALLELISM": "false",
}

transformers_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1", "wget")
    .pip_install(
        "accelerate",
        "datasets[audio]",
        "jiwer",
        "librosa",
        "numpy",
        "protobuf",
        "peft",
        "sentencepiece",
        "soundfile",
        "torch",
        "torchaudio",
        "transformers>=5.4.0",
    )
    .env(COMMON_ENV)
)

nemo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1", "wget")
    .pip_install(
        "datasets[audio]",
        "jiwer",
        "librosa",
        "numpy",
        "soundfile",
        "torch",
        "torchaudio",
        "transformers",
    )
    .pip_install("nemo_toolkit[asr]")
    .env(COMMON_ENV)
)

nemo_salm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1", "wget")
    .pip_install("uv")
    .run_commands(
        'uv pip install --system "datasets[audio]" "jiwer" "librosa" "numpy" '
        '"soundfile" "transformers"'
    )
    .run_commands(
        'uv pip install --system "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"'
    )
    .run_commands(
        "uv pip install --system --force-reinstall "
        "--index-url https://download.pytorch.org/whl/cu129 "
        '"torch==2.9.1" "torchaudio==2.9.1"'
    )
    .run_commands('uv pip install --system --force-reinstall "torchcodec==0.9.0"')
    .env(COMMON_ENV)
)

combine_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("jiwer", "transformers")
    .env(COMMON_ENV)
)

app = modal.App(APP_NAME)


@dataclass
class DatasetConfig:
    name: str = "ai4bharat/Svarah"
    config: str | None = None
    split: str | None = "test"
    audio_column: str | None = None
    text_column: str | None = None
    max_samples: int | None = None
    trust_remote_code: bool = False


@dataclass
class AsrModelSpec:
    label: str
    backend: str
    model_name: str
    adapter_scale: float | None = None


@dataclass
class AsrBakeoffConfig:
    bakeoff_name: str = "svarah-asr-bakeoff"
    eval_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model_specs: list[AsrModelSpec] = field(default_factory=list)
    per_device_eval_batch_size: int = 8
    distributed_gpu_count: int = 5
    max_new_tokens: int = 256
    language: str = "english"
    language_code: str = "en"
    punctuation: bool = True
    progress_log_interval_batches: int = 10


DEFAULT_MODEL_SPECS = [
    AsrModelSpec(
        label="whisper_turbo",
        backend="whisper_transformers",
        model_name="openai/whisper-large-v3-turbo",
    ),
    AsrModelSpec(
        label="whisper_large_v3",
        backend="whisper_transformers",
        model_name="openai/whisper-large-v3",
    ),
    AsrModelSpec(
        label="cohere_transcribe",
        backend="cohere_transformers",
        model_name="CohereLabs/cohere-transcribe-03-2026",
    ),
    AsrModelSpec(
        label="parakeet_tdt_v2",
        backend="nemo",
        model_name="nvidia/parakeet-tdt-0.6b-v2",
    ),
    AsrModelSpec(
        label="parakeet_unified",
        backend="nemo",
        model_name="nvidia/parakeet-unified-en-0.6b",
    ),
]

COHERE_BASE_MODEL = "CohereLabs/cohere-transcribe-03-2026"


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _sanitize_artifact_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-._")
    return sanitized or "item"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _run_dir(run_id: str) -> Path:
    return ARTIFACTS_DIR / run_id


def _progress_dir(run_id: str) -> Path:
    return _run_dir(run_id) / "progress"


def _progress_path(run_id: str) -> Path:
    return _run_dir(run_id) / "progress.json"


def _phase_progress_path(run_id: str, phase_key: str) -> Path:
    return _progress_dir(run_id) / f"{phase_key}.json"


def _worker_script_path() -> Path:
    return Path(__file__).resolve()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _normalize_audio_array(array: Any):
    import numpy as np

    if hasattr(array, "get_all_samples"):
        samples = array.get_all_samples()
        array = samples.data
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    audio_array = np.asarray(array, dtype=np.float32)
    if audio_array.ndim <= 1:
        return audio_array.reshape(-1)

    sample_axis = int(np.argmax(audio_array.shape))
    reduce_axes = tuple(index for index in range(audio_array.ndim) if index != sample_axis)
    if reduce_axes:
        audio_array = audio_array.mean(axis=reduce_axes)
    return np.asarray(audio_array, dtype=np.float32).reshape(-1)


def _audio_to_array_and_rate(audio: Any) -> tuple[Any, int]:
    if isinstance(audio, dict):
        return _normalize_audio_array(audio["array"]), int(audio.get("sampling_rate") or 16_000)
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        sample_rate = int(getattr(samples, "sample_rate", 16_000) or 16_000)
        return _normalize_audio_array(samples.data), sample_rate
    return _normalize_audio_array(audio), 16_000


def _audio_seconds(audio: dict[str, Any]) -> float:
    array, sampling_rate = _audio_to_array_and_rate(audio)
    return float(len(array)) / float(sampling_rate) if sampling_rate else 0.0


def _text_normalizer():
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        return BasicTextNormalizer()
    except Exception:
        punctuation = re.compile(r"[^\w\s']+", flags=re.UNICODE)

        def normalize(text: str) -> str:
            return re.sub(r"\s+", " ", punctuation.sub(" ", str(text).lower())).strip()

        return normalize


def _resolve_local_manifest_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    return ARTIFACTS_DIR / name.lstrip("/")


def _looks_like_local_manifest(name: str) -> bool:
    path = Path(name)
    if path.is_absolute():
        return True
    return name.startswith("/artifacts/") or path.suffix.lower() in {".jsonl", ".json", ".csv"}


def _canonicalize_manifest_row(row: dict[str, Any], *, audio_column: str | None) -> dict[str, Any]:
    canonical = dict(row)
    candidate_audio_columns = [audio_column] if audio_column else []
    candidate_audio_columns.extend(["audio", "audio_path", "audio_filepath", "path", "file", "wav"])
    for column in candidate_audio_columns:
        if not column or column not in canonical or not isinstance(canonical[column], str):
            continue
        audio_path = canonical[column].strip()
        if not audio_path:
            continue
        if not Path(audio_path).is_absolute():
            audio_path = str(ARTIFACTS_DIR / audio_path.lstrip("/"))
        canonical[column] = audio_path
        canonical["local_audio_path"] = audio_path
        break
    return canonical


def _load_local_manifest(config: DatasetConfig):
    from datasets import Dataset, load_dataset

    path = _resolve_local_manifest_path(config.name)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(
                        _canonicalize_manifest_row(json.loads(line), audio_column=config.audio_column)
                    )
        return Dataset.from_list(rows)

    builder_name = "json" if suffix == ".json" else "csv"
    split_name = config.split or "train"
    dataset = load_dataset(builder_name, data_files={split_name: str(path)}, split=split_name)
    return dataset.map(
        lambda row: _canonicalize_manifest_row(row, audio_column=config.audio_column),
        desc="canonicalize local manifest",
    )


def _load_dataset_split(config: DatasetConfig, *, token: str | None):
    from datasets import Audio, load_dataset

    if _looks_like_local_manifest(config.name):
        dataset = _load_local_manifest(config)
    else:
        kwargs: dict[str, Any] = {
            "path": config.name,
            "split": config.split or "test",
            "token": token,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.config:
            kwargs["name"] = config.config
        dataset = load_dataset(**kwargs)
    if config.max_samples and len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))

    audio_column = config.audio_column or _infer_audio_column(dataset)
    text_column = config.text_column or _infer_text_column(dataset)
    if audio_column not in dataset.column_names:
        raise ValueError(f"Audio column '{audio_column}' not found in {dataset.column_names}")
    if text_column not in dataset.column_names:
        raise ValueError(f"Text column '{text_column}' not found in {dataset.column_names}")
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16_000))
    return dataset, audio_column, text_column


def _infer_audio_column(dataset) -> str:
    for column_name, feature in dataset.features.items():
        if feature.__class__.__name__ == "Audio":
            return str(column_name)
    for column_name in ("audio", "audio_filepath", "path", "file", "wav"):
        if column_name in dataset.column_names:
            return column_name
    raise ValueError(f"Could not infer audio column from {dataset.column_names}")


def _infer_text_column(dataset) -> str:
    for column_name in (
        "text",
        "transcript",
        "transcription",
        "sentence",
        "normalized_text",
        "english_transcription",
    ):
        if column_name in dataset.column_names:
            return column_name
    raise ValueError(f"Could not infer text column from {dataset.column_names}")


def _metadata_rows(dataset, *, audio_column: str) -> list[dict[str, Any]]:
    metadata_dataset = dataset.remove_columns([audio_column])
    return [_json_safe(metadata_dataset[index]) for index in range(len(metadata_dataset))]


def _split_even_ranges(total: int, shards: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    shard_count = max(1, min(shards, total))
    return [
        ((total * shard_index) // shard_count, (total * (shard_index + 1)) // shard_count)
        for shard_index in range(shard_count)
    ]


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


def _summarize_numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _compute_text_metrics(references: list[str], predictions: list[str]) -> dict[str, Any]:
    from jiwer import cer, wer

    filtered_references: list[str] = []
    filtered_predictions: list[str] = []
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


def _progress_payload(
    *,
    phase_name: str,
    status: str,
    samples_done: int,
    samples_total: int,
    batches_done: int,
    batches_total: int | None,
    audio_seconds_done: float,
    started_at: float,
    inference_seconds_done: float,
) -> dict[str, Any]:
    elapsed = time.monotonic() - started_at
    percent_complete = (samples_done / samples_total) if samples_total else 0.0
    eta_seconds = None
    if samples_done > 0 and samples_done < samples_total:
        eta_seconds = (elapsed / samples_done) * (samples_total - samples_done)
    return {
        "phase": phase_name,
        "status": status,
        "updated_at_utc": _now_iso(),
        "samples_done": samples_done,
        "samples_total": samples_total,
        "percent_complete": percent_complete,
        "batches_done": batches_done,
        "batches_total": batches_total,
        "audio_seconds_done": audio_seconds_done,
        "elapsed_seconds": elapsed,
        "inference_seconds_done": inference_seconds_done,
        "current_rtfx": (audio_seconds_done / inference_seconds_done)
        if inference_seconds_done > 0
        else None,
        "eta_seconds": eta_seconds,
    }


def _load_transformers_model(spec: AsrModelSpec, config: AsrBakeoffConfig, *, token: str | None):
    import torch

    if spec.backend in {"cohere_transformers", "cohere_transformers_peft"}:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        model_name = spec.model_name
        adapter_path: Path | None = None
        if spec.backend == "cohere_transformers_peft":
            candidate = Path(spec.model_name)
            adapter_path = candidate if candidate.is_absolute() else ARTIFACTS_DIR / spec.model_name / "adapter"
            if not adapter_path.exists():
                raise FileNotFoundError(f"Cohere PEFT adapter not found: {adapter_path}")
            adapter_config_path = adapter_path / "adapter_config.json"
            if adapter_config_path.exists():
                adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                model_name = str(adapter_config.get("base_model_name_or_path") or COHERE_BASE_MODEL)
            else:
                model_name = COHERE_BASE_MODEL

        processor = AutoProcessor.from_pretrained(model_name, token=token)
        model = CohereAsrForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=token,
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            if spec.adapter_scale is not None:
                _apply_peft_adapter_scale(model, spec.adapter_scale)
        model.eval()
        return model, processor

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(spec.model_name, token=token)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(language=config.language, task="transcribe")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        spec.model_name,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    model.to("cuda")
    model.eval()
    return model, processor


def _apply_peft_adapter_scale(model: Any, adapter_scale: float) -> None:
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name, current_scale in list(scaling.items()):
            base_scales = getattr(module, "_localwispr_base_scaling", None)
            if base_scales is None:
                base_scales = {}
                setattr(module, "_localwispr_base_scaling", base_scales)
            if adapter_name not in base_scales:
                base_scales[adapter_name] = float(current_scale)
            scaling[adapter_name] = base_scales[adapter_name] * adapter_scale


def _predict_transformers(
    *,
    spec: AsrModelSpec,
    model: Any,
    processor: Any,
    dataset,
    audio_column: str,
    text_column: str,
    config: AsrBakeoffConfig,
    phase_name: str,
    progress_path: Path,
) -> dict[str, Any]:
    import torch

    normalizer = _text_normalizer()
    total_samples = len(dataset)
    batch_size = max(1, config.per_device_eval_batch_size)
    total_batches = int(math.ceil(total_samples / batch_size)) if total_samples else 0
    started_at = time.monotonic()
    inference_seconds_done = 0.0
    audio_seconds_done = 0.0
    predictions: list[str] = []
    references: list[str] = []
    audio_second_values: list[float] = []
    batch_latency_values: list[float] = []
    amortized_latency_values: list[float] = []

    for batch_index, start in enumerate(range(0, total_samples, batch_size), start=1):
        stop = min(start + batch_size, total_samples)
        batch = dataset.select(range(start, stop))
        audios = [row[audio_column] for row in batch]
        decoded_audios = [_audio_to_array_and_rate(audio) for audio in audios]
        arrays = [array for array, _ in decoded_audios]
        refs = [str(row[text_column]) for row in batch]
        audio_seconds_batch = [
            (float(len(array)) / float(sample_rate)) if sample_rate else 0.0
            for array, sample_rate in decoded_audios
        ]
        batch_audio_seconds = sum(audio_seconds_batch)

        batch_started = time.monotonic()
        if spec.backend in {"cohere_transformers", "cohere_transformers_peft"}:
            model_device = getattr(model, "device", None) or next(model.parameters()).device
            model_dtype = getattr(model, "dtype", None) or next(model.parameters()).dtype
            inputs = processor(
                arrays,
                sampling_rate=16_000,
                return_tensors="pt",
                language=config.language_code,
                punctuation=config.punctuation,
            )
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs.to(model_device, dtype=model_dtype)
            with torch.no_grad():
                try:
                    generated_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
                except ValueError as exc:
                    if "['length']" not in str(exc):
                        raise
                    inputs.pop("length", None)
                    generated_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
            decoded = processor.decode(
                generated_ids,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=config.language_code,
            )
            preds = [decoded] if isinstance(decoded, str) else list(decoded)
        else:
            inputs = processor.feature_extractor(
                arrays,
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
                    language=config.language,
                    task="transcribe",
                    max_new_tokens=config.max_new_tokens,
                )
            preds = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        batch_elapsed = time.monotonic() - batch_started
        inference_seconds_done += batch_elapsed
        audio_seconds_done += batch_audio_seconds
        predictions.extend(str(prediction) for prediction in preds)
        references.extend(refs)
        audio_second_values.extend(audio_seconds_batch)
        batch_latency_values.append(batch_elapsed)
        amortized_latency_values.extend([batch_elapsed / max(1, len(audios))] * len(audios))

        if (
            batch_index == 1
            or batch_index % max(1, config.progress_log_interval_batches) == 0
            or stop >= total_samples
        ):
            payload = _progress_payload(
                phase_name=phase_name,
                status="predicting",
                samples_done=stop,
                samples_total=total_samples,
                batches_done=batch_index,
                batches_total=total_batches,
                audio_seconds_done=audio_seconds_done,
                started_at=started_at,
                inference_seconds_done=inference_seconds_done,
            )
            _write_json(progress_path, payload)
            artifacts_volume.commit()
            print(
                f"[asr:{phase_name}] batches {batch_index}/{total_batches} "
                f"samples {stop}/{total_samples} rtfx={payload['current_rtfx']}"
            )

    normalized_predictions = [normalizer(text).strip() for text in predictions]
    normalized_references = [normalizer(text).strip() for text in references]
    return {
        "predictions": predictions,
        "references": references,
        "normalized_predictions": normalized_predictions,
        "normalized_references": normalized_references,
        "timing": {
            "inference_seconds": inference_seconds_done,
            "audio_seconds_total": audio_seconds_done,
            "rtfx": (audio_seconds_done / inference_seconds_done)
            if inference_seconds_done > 0
            else None,
            "audio_seconds_values": audio_second_values,
            "batch_latency_seconds_values": batch_latency_values,
            "amortized_sample_latency_seconds_values": amortized_latency_values,
            "audio_seconds": _summarize_numeric(audio_second_values),
            "batch_latency_seconds": _summarize_numeric(batch_latency_values),
            "amortized_sample_latency_seconds": _summarize_numeric(amortized_latency_values),
        },
    }


def _load_nemo_model(spec: AsrModelSpec):
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=spec.model_name)
    model.to("cuda")
    model.eval()
    return model


def _load_nemo_salm_model(spec: AsrModelSpec):
    from nemo.collections.speechlm2.models import SALM

    model = SALM.from_pretrained(spec.model_name)
    model.to("cuda")
    model.eval()
    return model


def _prefetch_nemo_models(model_specs: list[AsrModelSpec]) -> None:
    """Download and restore NeMo checkpoints once before parallel workers start."""
    import gc

    import nemo.collections.asr as nemo_asr
    import torch

    seen: set[str] = set()
    for spec in model_specs:
        if spec.backend != "nemo" or spec.model_name in seen:
            continue
        seen.add(spec.model_name)
        started = time.monotonic()
        print(f"[asr:nemo-prefetch] loading {spec.model_name}", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=spec.model_name)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[asr:nemo-prefetch] ready {spec.model_name} in {time.monotonic() - started:.2f}s",
            flush=True,
        )


def _prefetch_nemo_salm_models(model_specs: list[AsrModelSpec]) -> None:
    """Download and restore NeMo SALM checkpoints once before parallel workers start."""
    import gc

    import torch
    from nemo.collections.speechlm2.models import SALM

    seen: set[str] = set()
    for spec in model_specs:
        if spec.backend != "nemo_salm" or spec.model_name in seen:
            continue
        seen.add(spec.model_name)
        started = time.monotonic()
        print(f"[asr:nemo-salm-prefetch] loading {spec.model_name}", flush=True)
        model = SALM.from_pretrained(spec.model_name)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[asr:nemo-salm-prefetch] ready {spec.model_name} in {time.monotonic() - started:.2f}s",
            flush=True,
        )


def _nemo_prediction_text(item: Any) -> str:
    if hasattr(item, "text"):
        return str(item.text)
    if isinstance(item, dict) and "text" in item:
        return str(item["text"])
    return str(item)


def _decode_salm_outputs(model: Any, answer_ids: Any) -> list[str]:
    if hasattr(answer_ids, "detach"):
        answer_items = list(answer_ids)
    else:
        answer_items = list(answer_ids or [])
    predictions: list[str] = []
    for item in answer_items:
        if hasattr(item, "cpu"):
            item = item.cpu()
        predictions.append(str(model.tokenizer.ids_to_text(item)))
    return predictions


def _predict_nemo_salm(
    *,
    spec: AsrModelSpec,
    model: Any,
    dataset,
    audio_column: str,
    text_column: str,
    config: AsrBakeoffConfig,
    phase_name: str,
    phase_key: str,
    progress_path: Path,
) -> dict[str, Any]:
    import soundfile as sf

    normalizer = _text_normalizer()
    total_samples = len(dataset)
    batch_size = max(1, config.per_device_eval_batch_size)
    total_batches = int(math.ceil(total_samples / batch_size)) if total_samples else 0
    temp_dir = Path("/tmp/localwispr-asr-bakeoff") / phase_key
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.monotonic()
    inference_seconds_done = 0.0
    audio_seconds_done = 0.0
    predictions: list[str] = []
    references: list[str] = []
    audio_second_values: list[float] = []
    batch_latency_values: list[float] = []
    amortized_latency_values: list[float] = []

    try:
        for batch_index, start in enumerate(range(0, total_samples, batch_size), start=1):
            stop = min(start + batch_size, total_samples)
            batch = dataset.select(range(start, stop))
            paths: list[str] = []
            refs: list[str] = []
            batch_audio_seconds = 0.0
            for offset, row in enumerate(batch):
                audio = row[audio_column]
                array, sample_rate = _audio_to_array_and_rate(audio)
                audio_seconds = (float(len(array)) / float(sample_rate)) if sample_rate else 0.0
                wav_path = temp_dir / f"{start + offset:08d}.wav"
                sf.write(str(wav_path), array, sample_rate)
                paths.append(str(wav_path))
                refs.append(str(row[text_column]))
                batch_audio_seconds += audio_seconds
                audio_second_values.append(audio_seconds)

            prompts = [
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {model.audio_locator_tag}",
                        "audio": [path],
                    }
                ]
                for path in paths
            ]
            batch_started = time.monotonic()
            answer_ids = model.generate(prompts=prompts, max_new_tokens=config.max_new_tokens)
            preds = _decode_salm_outputs(model, answer_ids)
            batch_elapsed = time.monotonic() - batch_started

            inference_seconds_done += batch_elapsed
            audio_seconds_done += batch_audio_seconds
            predictions.extend(preds)
            references.extend(refs)
            batch_latency_values.append(batch_elapsed)
            amortized_latency_values.extend([batch_elapsed / max(1, len(paths))] * len(paths))

            for path in paths:
                try:
                    Path(path).unlink()
                except FileNotFoundError:
                    pass

            if (
                batch_index == 1
                or batch_index % max(1, config.progress_log_interval_batches) == 0
                or stop >= total_samples
            ):
                payload = _progress_payload(
                    phase_name=phase_name,
                    status="predicting",
                    samples_done=stop,
                    samples_total=total_samples,
                    batches_done=batch_index,
                    batches_total=total_batches,
                    audio_seconds_done=audio_seconds_done,
                    started_at=started_at,
                    inference_seconds_done=inference_seconds_done,
                )
                _write_json(progress_path, payload)
                artifacts_volume.commit()
                print(
                    f"[asr:{phase_name}] batches {batch_index}/{total_batches} "
                    f"samples {stop}/{total_samples} rtfx={payload['current_rtfx']}"
                )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    normalized_predictions = [normalizer(text).strip() for text in predictions]
    normalized_references = [normalizer(text).strip() for text in references]
    return {
        "predictions": predictions,
        "references": references,
        "normalized_predictions": normalized_predictions,
        "normalized_references": normalized_references,
        "timing": {
            "inference_seconds": inference_seconds_done,
            "audio_seconds_total": audio_seconds_done,
            "rtfx": (audio_seconds_done / inference_seconds_done)
            if inference_seconds_done > 0
            else None,
            "audio_seconds_values": audio_second_values,
            "batch_latency_seconds_values": batch_latency_values,
            "amortized_sample_latency_seconds_values": amortized_latency_values,
            "audio_seconds": _summarize_numeric(audio_second_values),
            "batch_latency_seconds": _summarize_numeric(batch_latency_values),
            "amortized_sample_latency_seconds": _summarize_numeric(amortized_latency_values),
        },
    }


def _predict_nemo(
    *,
    spec: AsrModelSpec,
    model: Any,
    dataset,
    audio_column: str,
    text_column: str,
    config: AsrBakeoffConfig,
    phase_name: str,
    phase_key: str,
    progress_path: Path,
) -> dict[str, Any]:
    import soundfile as sf

    normalizer = _text_normalizer()
    total_samples = len(dataset)
    batch_size = max(1, config.per_device_eval_batch_size)
    total_batches = int(math.ceil(total_samples / batch_size)) if total_samples else 0
    temp_dir = Path("/tmp/localwispr-asr-bakeoff") / phase_key
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.monotonic()
    inference_seconds_done = 0.0
    audio_seconds_done = 0.0
    predictions: list[str] = []
    references: list[str] = []
    audio_second_values: list[float] = []
    batch_latency_values: list[float] = []
    amortized_latency_values: list[float] = []

    try:
        for batch_index, start in enumerate(range(0, total_samples, batch_size), start=1):
            stop = min(start + batch_size, total_samples)
            batch = dataset.select(range(start, stop))
            paths: list[str] = []
            refs: list[str] = []
            batch_audio_seconds = 0.0
            for offset, row in enumerate(batch):
                audio = row[audio_column]
                array, sample_rate = _audio_to_array_and_rate(audio)
                audio_seconds = (float(len(array)) / float(sample_rate)) if sample_rate else 0.0
                wav_path = temp_dir / f"{start + offset:08d}.wav"
                sf.write(str(wav_path), array, sample_rate)
                paths.append(str(wav_path))
                refs.append(str(row[text_column]))
                batch_audio_seconds += audio_seconds
                audio_second_values.append(audio_seconds)

            batch_started = time.monotonic()
            output = model.transcribe(paths, batch_size=len(paths))
            preds = [_nemo_prediction_text(item) for item in output]
            batch_elapsed = time.monotonic() - batch_started

            inference_seconds_done += batch_elapsed
            audio_seconds_done += batch_audio_seconds
            predictions.extend(preds)
            references.extend(refs)
            batch_latency_values.append(batch_elapsed)
            amortized_latency_values.extend([batch_elapsed / max(1, len(paths))] * len(paths))

            for path in paths:
                try:
                    Path(path).unlink()
                except FileNotFoundError:
                    pass

            if (
                batch_index == 1
                or batch_index % max(1, config.progress_log_interval_batches) == 0
                or stop >= total_samples
            ):
                payload = _progress_payload(
                    phase_name=phase_name,
                    status="predicting",
                    samples_done=stop,
                    samples_total=total_samples,
                    batches_done=batch_index,
                    batches_total=total_batches,
                    audio_seconds_done=audio_seconds_done,
                    started_at=started_at,
                    inference_seconds_done=inference_seconds_done,
                )
                _write_json(progress_path, payload)
                artifacts_volume.commit()
                print(
                    f"[asr:{phase_name}] batches {batch_index}/{total_batches} "
                    f"samples {stop}/{total_samples} rtfx={payload['current_rtfx']}"
                )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    normalized_predictions = [normalizer(text).strip() for text in predictions]
    normalized_references = [normalizer(text).strip() for text in references]
    return {
        "predictions": predictions,
        "references": references,
        "normalized_predictions": normalized_predictions,
        "normalized_references": normalized_references,
        "timing": {
            "inference_seconds": inference_seconds_done,
            "audio_seconds_total": audio_seconds_done,
            "rtfx": (audio_seconds_done / inference_seconds_done)
            if inference_seconds_done > 0
            else None,
            "audio_seconds_values": audio_second_values,
            "batch_latency_seconds_values": batch_latency_values,
            "amortized_sample_latency_seconds_values": amortized_latency_values,
            "audio_seconds": _summarize_numeric(audio_second_values),
            "batch_latency_seconds": _summarize_numeric(batch_latency_values),
            "amortized_sample_latency_seconds": _summarize_numeric(amortized_latency_values),
        },
    }


def _asr_worker_main_impl(config_path: str) -> None:
    import torch

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = _normalize_config(payload["config"])
    spec = _normalize_model_spec(payload["model_spec"])
    run_id = str(payload["run_id"])
    phase_key = str(payload["phase_key"])
    phase_name = str(payload["phase_name"])
    shard_start = int(payload["shard_start"])
    shard_stop = int(payload["shard_stop"])
    progress_path = Path(payload["progress_path"])
    output_path = Path(payload["output_path"])
    hf_token = _get_hf_token()
    worker_started = time.monotonic()

    _write_json(
        progress_path,
        _progress_payload(
            phase_name=phase_name,
            status="loading_dataset",
            samples_done=0,
            samples_total=shard_stop - shard_start,
            batches_done=0,
            batches_total=None,
            audio_seconds_done=0.0,
            started_at=worker_started,
            inference_seconds_done=0.0,
        ),
    )
    artifacts_volume.commit()
    dataset_load_started = time.monotonic()
    dataset, audio_column, text_column = _load_dataset_split(config.eval_dataset, token=hf_token)
    shard_dataset = dataset.select(range(shard_start, shard_stop))
    metadata_rows = _metadata_rows(shard_dataset, audio_column=audio_column)
    dataset_load_seconds = time.monotonic() - dataset_load_started

    _write_json(
        progress_path,
        _progress_payload(
            phase_name=phase_name,
            status="loading_model",
            samples_done=0,
            samples_total=shard_stop - shard_start,
            batches_done=0,
            batches_total=None,
            audio_seconds_done=0.0,
            started_at=worker_started,
            inference_seconds_done=0.0,
        ),
    )
    artifacts_volume.commit()
    model_load_started = time.monotonic()
    if spec.backend == "nemo":
        model = _load_nemo_model(spec)
    elif spec.backend == "nemo_salm":
        model = _load_nemo_salm_model(spec)
    else:
        model, processor = _load_transformers_model(spec, config, token=hf_token)
    model_load_seconds = time.monotonic() - model_load_started

    if spec.backend == "nemo":
        prediction_payload = _predict_nemo(
            spec=spec,
            model=model,
            dataset=shard_dataset,
            audio_column=audio_column,
            text_column=text_column,
            config=config,
            phase_name=phase_name,
            phase_key=phase_key,
            progress_path=progress_path,
        )
    elif spec.backend == "nemo_salm":
        prediction_payload = _predict_nemo_salm(
            spec=spec,
            model=model,
            dataset=shard_dataset,
            audio_column=audio_column,
            text_column=text_column,
            config=config,
            phase_name=phase_name,
            phase_key=phase_key,
            progress_path=progress_path,
        )
    else:
        prediction_payload = _predict_transformers(
            spec=spec,
            model=model,
            processor=processor,
            dataset=shard_dataset,
            audio_column=audio_column,
            text_column=text_column,
            config=config,
            phase_name=phase_name,
            progress_path=progress_path,
        )

    worker_wall_seconds = time.monotonic() - worker_started
    result_payload = {
        "run_id": run_id,
        "phase_key": phase_key,
        "phase_name": phase_name,
        "model_spec": asdict(spec),
        "shard_index": int(payload["shard_index"]),
        "shards_total": int(payload["shards_total"]),
        "shard_start": shard_start,
        "shard_stop": shard_stop,
        "sample_count": len(prediction_payload["references"]),
        "predictions": prediction_payload["predictions"],
        "references": prediction_payload["references"],
        "normalized_predictions": prediction_payload["normalized_predictions"],
        "normalized_references": prediction_payload["normalized_references"],
        "metadata_rows": metadata_rows,
        "timing": {
            **prediction_payload["timing"],
            "dataset_load_seconds": dataset_load_seconds,
            "model_load_seconds": model_load_seconds,
            "worker_wall_seconds": worker_wall_seconds,
        },
    }
    _write_json(output_path, result_payload)
    _write_json(
        progress_path,
        {
            **_progress_payload(
                phase_name=phase_name,
                status="complete",
                samples_done=shard_stop - shard_start,
                samples_total=shard_stop - shard_start,
                batches_done=int(math.ceil((shard_stop - shard_start) / max(1, config.per_device_eval_batch_size))),
                batches_total=int(math.ceil((shard_stop - shard_start) / max(1, config.per_device_eval_batch_size))),
                audio_seconds_done=float(prediction_payload["timing"]["audio_seconds_total"]),
                started_at=worker_started,
                inference_seconds_done=float(prediction_payload["timing"]["inference_seconds"]),
            ),
            "result_path": str(output_path),
        },
    )
    artifacts_volume.commit()
    hf_cache_volume.commit()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _asr_worker_main(config_path: str) -> None:
    try:
        _asr_worker_main_impl(config_path)
    except Exception as exc:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        progress_path = Path(payload["progress_path"])
        output_path = Path(payload["output_path"])
        error_payload = {
            "run_id": payload.get("run_id"),
            "phase_key": payload.get("phase_key"),
            "phase_name": payload.get("phase_name"),
            "model_spec": payload.get("model_spec"),
            "shard_index": payload.get("shard_index"),
            "shards_total": payload.get("shards_total"),
            "shard_start": payload.get("shard_start"),
            "shard_stop": payload.get("shard_stop"),
            "sample_count": int(payload.get("sample_count") or 0),
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        _write_json(output_path, error_payload)
        _write_json(
            progress_path,
            {
                "phase": payload.get("phase_name"),
                "status": "failed",
                "updated_at_utc": _now_iso(),
                "samples_done": 0,
                "samples_total": payload.get("sample_count"),
                "error": error_payload["error"],
            },
        )
        artifacts_volume.commit()
        print(traceback.format_exc(), file=sys.stderr)


def _known_worker_payloads(run_id: str, worker_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_phase_key: dict[str, dict[str, Any]] = {}
    for path in sorted(_progress_dir(run_id).glob("*.worker.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        phase_key = str(payload.get("phase_key") or "")
        if phase_key:
            by_phase_key[phase_key] = payload
    for payload in worker_payloads:
        by_phase_key[str(payload["phase_key"])] = payload
    return list(by_phase_key.values())


def _backend_progress_path(run_id: str, backend_group: str) -> Path:
    return _run_dir(run_id) / f"{backend_group}.progress.json"


def _aggregate_progress(
    run_id: str,
    worker_payloads: list[dict[str, Any]],
    *,
    backend_group: str | None = None,
) -> dict[str, Any]:
    worker_payloads = _known_worker_payloads(run_id, worker_payloads)
    phases: dict[str, Any] = {}
    samples_done = 0
    samples_total = 0
    audio_seconds_done = 0.0
    failed = 0
    for payload in worker_payloads:
        samples_total += int(payload["sample_count"])
        progress_path = Path(payload["progress_path"])
        phase_state: dict[str, Any] = {
            "phase_name": payload["phase_name"],
            "model_spec": payload["model_spec"],
            "samples_total": int(payload["sample_count"]),
            "samples_done": 0,
            "batches_done": 0,
            "batches_total": None,
            "gpu_index": payload.get("gpu_index"),
            "status": "pending",
        }
        if progress_path.exists():
            phase_state.update(json.loads(progress_path.read_text(encoding="utf-8")))
        if Path(payload["output_path"]).exists():
            output_payload = json.loads(Path(payload["output_path"]).read_text(encoding="utf-8"))
            if output_payload.get("error"):
                phase_state["status"] = "failed"
                phase_state["error"] = output_payload["error"]
                failed += 1
            else:
                phase_state["status"] = "complete"
                phase_state["samples_done"] = int(payload["sample_count"])
        samples_done += int(phase_state.get("samples_done") or 0)
        audio_seconds_done += float(phase_state.get("audio_seconds_done") or 0.0)
        phases[payload["phase_key"]] = phase_state

    return {
        "stage": "asr_bakeoff",
        "backend_group": backend_group,
        "status": "running" if samples_done < samples_total and failed == 0 else "collecting",
        "updated_at_utc": _now_iso(),
        "run_id": run_id,
        "samples_done": samples_done,
        "samples_total": samples_total,
        "percent_complete": (samples_done / samples_total) if samples_total else 0.0,
        "audio_seconds_done": audio_seconds_done,
        "failed_workers": failed,
        "phases": phases,
    }


def _run_worker_processes(
    *,
    run_id: str,
    backend_group: str,
    worker_payloads: list[dict[str, Any]],
    gpu_count: int,
) -> None:
    pending = list(worker_payloads)
    running: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
    gpu_indexes = list(range(max(1, gpu_count)))
    last_commit = time.monotonic()
    for payload in worker_payloads:
        worker_config_path = _progress_dir(run_id) / f"{payload['phase_key']}.worker.json"
        _write_json(worker_config_path, payload)

    try:
        while pending or running:
            used_gpu_indexes = {int(payload["gpu_index"]) for payload, _ in running}
            available_gpu_indexes = [index for index in gpu_indexes if index not in used_gpu_indexes]
            while pending and available_gpu_indexes:
                payload = pending.pop(0)
                payload["gpu_index"] = available_gpu_indexes.pop(0)
                worker_config_path = _progress_dir(run_id) / f"{payload['phase_key']}.worker.json"
                _write_json(worker_config_path, payload)
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(payload["gpu_index"])
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(_worker_script_path()),
                        "--asr-worker-config-path",
                        str(worker_config_path),
                    ],
                    cwd=str(_worker_script_path().parent),
                    env=env,
                )
                running.append((payload, process))

            next_running: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
            for payload, process in running:
                return_code = process.poll()
                if return_code is None:
                    next_running.append((payload, process))
                    continue
                if return_code != 0:
                    raise RuntimeError(
                        f"ASR bakeoff worker {payload['phase_key']} failed with exit code {return_code}"
                    )

            running = next_running
            progress_payload = _aggregate_progress(
                run_id,
                worker_payloads,
                backend_group=backend_group,
            )
            should_commit = (time.monotonic() - last_commit) >= 10 or not (pending or running)
            _write_json(_backend_progress_path(run_id, backend_group), progress_payload)
            _write_json(_progress_path(run_id), progress_payload)
            if should_commit:
                artifacts_volume.commit()
                last_commit = time.monotonic()
            if pending or running:
                time.sleep(3)
    finally:
        for _, process in running:
            if process.poll() is None:
                process.terminate()


def _build_worker_payloads(
    *,
    run_id: str,
    config: AsrBakeoffConfig,
    model_specs: list[AsrModelSpec],
    rows: int,
) -> list[dict[str, Any]]:
    shard_ranges = _split_even_ranges(rows, max(1, config.distributed_gpu_count))
    worker_payloads: list[dict[str, Any]] = []
    for spec in model_specs:
        for shard_index, (shard_start, shard_stop) in enumerate(shard_ranges):
            phase_key = _sanitize_artifact_component(
                f"{spec.label}-shard-{shard_index + 1}-of-{len(shard_ranges)}"
            )
            worker_payloads.append(
                {
                    "run_id": run_id,
                    "phase_key": phase_key,
                    "phase_name": f"{spec.label} shard {shard_index + 1}/{len(shard_ranges)}",
                    "model_spec": asdict(spec),
                    "shard_index": shard_index,
                    "shards_total": len(shard_ranges),
                    "shard_start": shard_start,
                    "shard_stop": shard_stop,
                    "sample_count": shard_stop - shard_start,
                    "gpu_index": None,
                    "progress_path": str(_phase_progress_path(run_id, phase_key)),
                    "output_path": str(_progress_dir(run_id) / f"{phase_key}.result.json"),
                    "config": asdict(config),
                }
            )
    return worker_payloads


def _prepare_backend_run(
    *,
    run_id: str,
    config: AsrBakeoffConfig,
    backend_group: str,
    model_specs: list[AsrModelSpec],
) -> list[dict[str, Any]]:
    hf_token = _get_hf_token()
    dataset, audio_column, text_column = _load_dataset_split(config.eval_dataset, token=hf_token)
    _progress_dir(run_id).mkdir(parents=True, exist_ok=True)
    backend_payload = {
        "stage": "asr_bakeoff",
        "status": "starting",
        "backend_group": backend_group,
        "updated_at_utc": _now_iso(),
        "run_id": run_id,
        "dataset": {
            "name": config.eval_dataset.name,
            "config": config.eval_dataset.config,
            "split": config.eval_dataset.split,
            "audio_column": audio_column,
            "text_column": text_column,
            "samples": len(dataset),
            "max_samples": config.eval_dataset.max_samples,
        },
        "models": [asdict(spec) for spec in model_specs],
        "distributed_gpu_count": config.distributed_gpu_count,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
    }
    _write_json(_run_dir(run_id) / f"{backend_group}.start.json", backend_payload)
    _write_json(_backend_progress_path(run_id, backend_group), backend_payload)
    _write_json(_progress_path(run_id), backend_payload)
    artifacts_volume.commit()
    return _build_worker_payloads(
        run_id=run_id,
        config=config,
        model_specs=model_specs,
        rows=len(dataset),
    )


def _run_backend_group(
    *,
    run_id: str,
    config: AsrBakeoffConfig,
    backend_group: str,
    model_specs: list[AsrModelSpec],
) -> dict[str, Any]:
    started_at = time.monotonic()
    worker_payloads = _prepare_backend_run(
        run_id=run_id,
        config=config,
        backend_group=backend_group,
        model_specs=model_specs,
    )
    if backend_group in {"nemo", "nemo_salm"}:
        _write_json(
            _backend_progress_path(run_id, backend_group),
            {
                "stage": "asr_bakeoff",
                "backend_group": backend_group,
                "status": "prefetching_models",
                "updated_at_utc": _now_iso(),
                "run_id": run_id,
                "models": [asdict(spec) for spec in model_specs],
            },
        )
        artifacts_volume.commit()
        if backend_group == "nemo":
            _prefetch_nemo_models(model_specs)
        else:
            _prefetch_nemo_salm_models(model_specs)
    _run_worker_processes(
        run_id=run_id,
        backend_group=backend_group,
        worker_payloads=worker_payloads,
        gpu_count=max(1, min(config.distributed_gpu_count, len(worker_payloads))),
    )
    payload = {
        "backend_group": backend_group,
        "run_id": run_id,
        "status": "complete",
        "wall_seconds": time.monotonic() - started_at,
        "workers": [
            {
                "phase_key": worker["phase_key"],
                "output_path": worker["output_path"],
                "model_spec": worker["model_spec"],
            }
            for worker in worker_payloads
        ],
    }
    _write_json(_run_dir(run_id) / f"{backend_group}.complete.json", payload)
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return payload


def _normalize_model_spec(payload: dict[str, Any] | AsrModelSpec) -> AsrModelSpec:
    if isinstance(payload, AsrModelSpec):
        return payload
    return AsrModelSpec(
        label=str(payload["label"]),
        backend=str(payload["backend"]),
        model_name=str(payload["model_name"]),
        adapter_scale=float(payload["adapter_scale"])
        if payload.get("adapter_scale") is not None
        else None,
    )


def _normalize_config(payload: dict[str, Any] | AsrBakeoffConfig) -> AsrBakeoffConfig:
    if isinstance(payload, AsrBakeoffConfig):
        return payload
    eval_dataset_payload = payload.get("eval_dataset") or {}
    model_specs = [_normalize_model_spec(item) for item in payload.get("model_specs") or []]
    return AsrBakeoffConfig(
        bakeoff_name=str(payload.get("bakeoff_name") or "svarah-asr-bakeoff"),
        eval_dataset=DatasetConfig(
            name=str(eval_dataset_payload.get("name") or "ai4bharat/Svarah"),
            config=eval_dataset_payload.get("config"),
            split=eval_dataset_payload.get("split") or "test",
            audio_column=eval_dataset_payload.get("audio_column"),
            text_column=eval_dataset_payload.get("text_column"),
            max_samples=eval_dataset_payload.get("max_samples"),
            trust_remote_code=bool(eval_dataset_payload.get("trust_remote_code") or False),
        ),
        model_specs=model_specs,
        per_device_eval_batch_size=int(payload.get("per_device_eval_batch_size") or 8),
        distributed_gpu_count=int(payload.get("distributed_gpu_count") or 5),
        max_new_tokens=int(payload.get("max_new_tokens") or 256),
        language=str(payload.get("language") or "english"),
        language_code=str(payload.get("language_code") or "en"),
        punctuation=bool(payload.get("punctuation", True)),
        progress_log_interval_batches=int(payload.get("progress_log_interval_batches") or 10),
    )


def _collect_results(run_id: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for path in sorted(_progress_dir(run_id).glob("*.result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = payload.get("model_spec") or {}
        label = str(spec.get("label") or payload.get("phase_key") or path.stem)
        if payload.get("error"):
            errors.append(
                {
                    "label": label,
                    "phase_key": payload.get("phase_key"),
                    "error": payload["error"],
                }
            )
            continue
        by_label.setdefault(label, []).append(payload)
    for payloads in by_label.values():
        payloads.sort(key=lambda item: int(item.get("shard_start") or 0))
    return by_label, errors


def _merge_model_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "predictions": [],
        "references": [],
        "normalized_predictions": [],
        "normalized_references": [],
        "metadata_rows": [],
    }
    timing_workers: list[dict[str, Any]] = []
    for payload in payloads:
        for key in merged:
            merged[key].extend(payload[key])
        timing_workers.append(payload["timing"])
    return {
        **merged,
        "timing": _merge_timing(timing_workers),
        "workers": timing_workers,
        "model_spec": payloads[0]["model_spec"],
        "sample_count": len(merged["references"]),
    }


def _merge_timing(timing_workers: list[dict[str, Any]]) -> dict[str, Any]:
    audio_seconds_total = sum(float(item.get("audio_seconds_total") or 0.0) for item in timing_workers)
    inference_seconds_sum = sum(float(item.get("inference_seconds") or 0.0) for item in timing_workers)
    parallel_inference_seconds = max(
        [float(item.get("inference_seconds") or 0.0) for item in timing_workers] or [0.0]
    )
    parallel_worker_wall_seconds = max(
        [float(item.get("worker_wall_seconds") or 0.0) for item in timing_workers] or [0.0]
    )
    model_load_seconds_max = max(
        [float(item.get("model_load_seconds") or 0.0) for item in timing_workers] or [0.0]
    )
    dataset_load_seconds_max = max(
        [float(item.get("dataset_load_seconds") or 0.0) for item in timing_workers] or [0.0]
    )
    batch_latency_values: list[float] = []
    amortized_latency_values: list[float] = []
    audio_second_values: list[float] = []
    for item in timing_workers:
        for values_key, summary_key, target in (
            ("batch_latency_seconds_values", "batch_latency_seconds", batch_latency_values),
            (
                "amortized_sample_latency_seconds_values",
                "amortized_sample_latency_seconds",
                amortized_latency_values,
            ),
            ("audio_seconds_values", "audio_seconds", audio_second_values),
        ):
            raw_values = item.get(values_key)
            if isinstance(raw_values, list):
                target.extend(float(value) for value in raw_values)
                continue
            summary = item.get(summary_key) or {}
            count = int(summary.get("count") or 0)
            mean = summary.get("mean")
            if count and mean is not None:
                target.extend([float(mean)] * count)
    return {
        "audio_seconds_total": audio_seconds_total,
        "inference_seconds_sum": inference_seconds_sum,
        "parallel_inference_seconds": parallel_inference_seconds,
        "parallel_worker_wall_seconds": parallel_worker_wall_seconds,
        "model_load_seconds_max": model_load_seconds_max,
        "dataset_load_seconds_max": dataset_load_seconds_max,
        "parallel_rtfx": (audio_seconds_total / parallel_inference_seconds)
        if parallel_inference_seconds > 0
        else None,
        "single_gpu_equivalent_rtfx": (audio_seconds_total / inference_seconds_sum)
        if inference_seconds_sum > 0
        else None,
        "batch_latency_seconds": _summarize_numeric(batch_latency_values),
        "amortized_sample_latency_seconds": _summarize_numeric(amortized_latency_values),
        "audio_seconds": _summarize_numeric(audio_second_values),
    }


def _write_pairwise_predictions(
    *,
    output_path: Path,
    metadata_rows: list[dict[str, Any]],
    raw_references: list[str],
    normalized_references: list[str],
    predictions_by_model: dict[str, list[str]],
    normalized_predictions_by_model: dict[str, list[str]],
) -> None:
    from jiwer import cer, wer

    if output_path.exists():
        output_path.unlink()
    rows: list[dict[str, Any]] = []
    labels = list(predictions_by_model)
    for index, reference in enumerate(normalized_references):
        row: dict[str, Any] = {
            "index": index,
            "reference": raw_references[index],
            "normalized_reference": reference,
            "metadata": metadata_rows[index] if index < len(metadata_rows) else {},
        }
        for label in labels:
            normalized_prediction = normalized_predictions_by_model[label][index]
            row[f"{label}__prediction"] = predictions_by_model[label][index]
            row[f"{label}__normalized_prediction"] = normalized_prediction
            if reference:
                row[f"{label}__wer"] = wer(reference, normalized_prediction)
                row[f"{label}__cer"] = cer(reference, normalized_prediction)
        rows.append(row)
        if len(rows) >= 500:
            _append_jsonl(output_path, rows)
            rows = []
    if rows:
        _append_jsonl(output_path, rows)


def _compute_oracle(
    *,
    normalized_references: list[str],
    normalized_predictions_by_model: dict[str, list[str]],
    baseline_label: str,
) -> dict[str, Any]:
    from jiwer import cer, wer

    oracle_predictions: list[str] = []
    choice_counts: dict[str, int] = {}
    labels = list(normalized_predictions_by_model)
    for index, reference in enumerate(normalized_references):
        best_label = baseline_label
        best_score = (
            wer(reference, normalized_predictions_by_model[baseline_label][index]),
            cer(reference, normalized_predictions_by_model[baseline_label][index]),
            0,
        )
        for label in labels:
            score = (
                wer(reference, normalized_predictions_by_model[label][index]),
                cer(reference, normalized_predictions_by_model[label][index]),
                0 if label == baseline_label else 1,
            )
            if score < best_score:
                best_label = label
                best_score = score
        choice_counts[best_label] = choice_counts.get(best_label, 0) + 1
        oracle_predictions.append(normalized_predictions_by_model[best_label][index])
    metrics = _compute_text_metrics(normalized_references, oracle_predictions)
    baseline_metrics = _compute_text_metrics(
        normalized_references,
        normalized_predictions_by_model[baseline_label],
    )
    return {
        "metrics": metrics,
        "delta_vs_baseline": {
            "wer": metrics["wer"] - baseline_metrics["wer"],
            "cer": metrics["cer"] - baseline_metrics["cer"],
        },
        "choice_counts": choice_counts,
    }


def _combine_impl(config: AsrBakeoffConfig, *, run_id: str) -> dict[str, Any]:
    by_label, errors = _collect_results(run_id)
    if not by_label:
        raise RuntimeError(f"No successful ASR bakeoff result files found for {run_id}")

    merged_by_label = {
        label: _merge_model_payloads(payloads) for label, payloads in sorted(by_label.items())
    }
    baseline_label = "whisper_turbo" if "whisper_turbo" in merged_by_label else next(iter(merged_by_label))
    baseline_payload = merged_by_label[baseline_label]
    raw_references = baseline_payload["references"]
    normalized_references = baseline_payload["normalized_references"]
    metadata_rows = baseline_payload["metadata_rows"]
    predictions_by_model = {
        label: payload["predictions"] for label, payload in merged_by_label.items()
    }
    normalized_predictions_by_model = {
        label: payload["normalized_predictions"] for label, payload in merged_by_label.items()
    }
    overall_metrics = {
        label: _compute_text_metrics(normalized_references, predictions)
        for label, predictions in normalized_predictions_by_model.items()
    }
    deltas_vs_baseline = {
        label: {
            "wer": metrics["wer"] - overall_metrics[baseline_label]["wer"],
            "cer": metrics["cer"] - overall_metrics[baseline_label]["cer"],
        }
        for label, metrics in overall_metrics.items()
        if label != baseline_label
    }
    pairwise_path = _run_dir(run_id) / "pairwise_predictions.jsonl"
    _write_pairwise_predictions(
        output_path=pairwise_path,
        metadata_rows=metadata_rows,
        raw_references=raw_references,
        normalized_references=normalized_references,
        predictions_by_model=predictions_by_model,
        normalized_predictions_by_model=normalized_predictions_by_model,
    )
    report = {
        "created_at_utc": _now_iso(),
        "run_id": run_id,
        "config": asdict(config),
        "baseline_label": baseline_label,
        "models": {
            label: payload["model_spec"] for label, payload in merged_by_label.items()
        },
        "overall_metrics": overall_metrics,
        "deltas_vs_baseline": deltas_vs_baseline,
        "timing": {
            label: payload["timing"] for label, payload in merged_by_label.items()
        },
        "oracle": _compute_oracle(
            normalized_references=normalized_references,
            normalized_predictions_by_model=normalized_predictions_by_model,
            baseline_label=baseline_label,
        )
        if len(merged_by_label) > 1
        else None,
        "errors": errors,
        "artifacts": {
            "run_dir": str(_run_dir(run_id)),
            "report_path": str(_run_dir(run_id) / "report.json"),
            "pairwise_predictions_jsonl": str(pairwise_path),
            "progress_path": str(_progress_path(run_id)),
        },
    }
    _write_json(_run_dir(run_id) / "report.json", report)
    _write_json(
        _progress_path(run_id),
        {
            "stage": "complete",
            "status": "complete",
            "updated_at_utc": _now_iso(),
            "run_id": run_id,
            "report_path": str(_run_dir(run_id) / "report.json"),
            "models_completed": sorted(merged_by_label),
            "failed_workers": len(errors),
        },
    )
    artifacts_volume.commit()
    return report


@app.function(
    image=transformers_image,
    gpu=FIVE_GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def asr_bakeoff_transformers_5gpu_remote(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FIVE_GPU
    config = _normalize_config(payload["config"])
    run_id = str(payload["run_id"])
    model_specs = [_normalize_model_spec(item) for item in payload["model_specs"]]
    return _run_backend_group(
        run_id=run_id,
        config=config,
        backend_group="transformers",
        model_specs=model_specs,
    )


@app.function(
    image=nemo_image,
    gpu=FIVE_GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def asr_bakeoff_nemo_5gpu_remote(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FIVE_GPU
    config = _normalize_config(payload["config"])
    run_id = str(payload["run_id"])
    model_specs = [_normalize_model_spec(item) for item in payload["model_specs"]]
    return _run_backend_group(
        run_id=run_id,
        config=config,
        backend_group="nemo",
        model_specs=model_specs,
    )


@app.function(
    image=nemo_salm_image,
    gpu=FIVE_GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def asr_bakeoff_nemo_salm_5gpu_remote(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FIVE_GPU
    config = _normalize_config(payload["config"])
    run_id = str(payload["run_id"])
    model_specs = [_normalize_model_spec(item) for item in payload["model_specs"]]
    return _run_backend_group(
        run_id=run_id,
        config=config,
        backend_group="nemo_salm",
        model_specs=model_specs,
    )


@app.function(
    image=transformers_image,
    gpu=FIVE_GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def asr_bakeoff_transformers_full_remote(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = FIVE_GPU
    config = _normalize_config(payload["config"])
    run_id = str(payload["run_id"])
    model_specs = [_normalize_model_spec(item) for item in payload["model_specs"]]
    unsupported = [
        spec.backend
        for spec in model_specs
        if spec.backend not in {"whisper_transformers", "cohere_transformers", "cohere_transformers_peft"}
    ]
    if unsupported:
        raise ValueError(f"full_remote only supports Transformers backends, got {sorted(set(unsupported))}")
    _run_backend_group(
        run_id=run_id,
        config=config,
        backend_group="transformers",
        model_specs=model_specs,
    )
    report = _combine_impl(config, run_id=run_id)
    hf_cache_volume.commit()
    return report


@app.function(
    image=combine_image,
    timeout=60 * 30,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={str(ARTIFACTS_DIR): artifacts_volume},
)
def asr_bakeoff_combine_remote(payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_config(payload["config"])
    return _combine_impl(config, run_id=str(payload["run_id"]))


def _parse_models(value: str) -> list[AsrModelSpec]:
    if not value.strip():
        return list(DEFAULT_MODEL_SPECS)
    aliases = {spec.label: spec for spec in DEFAULT_MODEL_SPECS}
    specs: list[AsrModelSpec] = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if item in aliases:
            specs.append(aliases[item])
            continue
        pieces = [piece.strip() for piece in item.split("|")]
        if len(pieces) not in {3, 4}:
            raise ValueError(
                "bakeoff_models entries must be aliases or label|backend|model_name[|adapter_scale]"
            )
        specs.append(
            AsrModelSpec(
                label=pieces[0],
                backend=pieces[1],
                model_name=pieces[2],
                adapter_scale=float(pieces[3]) if len(pieces) == 4 else None,
            )
        )
    return specs


def _run_bakeoff(config: AsrBakeoffConfig, *, run_id: str, parallel_backend_jobs: bool) -> dict[str, Any]:
    transformers_specs = [
        spec
        for spec in config.model_specs
        if spec.backend in {"whisper_transformers", "cohere_transformers", "cohere_transformers_peft"}
    ]
    nemo_specs = [spec for spec in config.model_specs if spec.backend == "nemo"]
    nemo_salm_specs = [spec for spec in config.model_specs if spec.backend == "nemo_salm"]
    remote_payloads = []
    if transformers_specs:
        remote_payloads.append(
            (
                asr_bakeoff_transformers_5gpu_remote,
                {
                    "run_id": run_id,
                    "config": asdict(config),
                    "model_specs": [asdict(spec) for spec in transformers_specs],
                },
            )
        )
    if nemo_specs:
        remote_payloads.append(
            (
                asr_bakeoff_nemo_5gpu_remote,
                {
                    "run_id": run_id,
                    "config": asdict(config),
                    "model_specs": [asdict(spec) for spec in nemo_specs],
                },
            )
        )
    if nemo_salm_specs:
        remote_payloads.append(
            (
                asr_bakeoff_nemo_salm_5gpu_remote,
                {
                    "run_id": run_id,
                    "config": asdict(config),
                    "model_specs": [asdict(spec) for spec in nemo_salm_specs],
                },
            )
        )

    if parallel_backend_jobs and len(remote_payloads) > 1:
        calls = [remote.spawn(payload) for remote, payload in remote_payloads]
        for call in calls:
            call.get()
    else:
        for remote, payload in remote_payloads:
            remote.remote(payload)

    return asr_bakeoff_combine_remote.remote({"run_id": run_id, "config": asdict(config)})


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    bakeoff_name: str = "svarah-asr-bakeoff",
    eval_dataset: str = "ai4bharat/Svarah",
    eval_config_name: str = "",
    eval_split: str = "test",
    eval_audio_column: str = "",
    eval_text_column: str = "",
    eval_trust_remote_code: bool = False,
    svarah_max_samples: int = 0,
    smoke_samples: int = 64,
    bakeoff_models: str = "",
    per_device_eval_batch_size: int = 8,
    distributed_gpu_count: int = 5,
    max_new_tokens: int = 256,
    progress_log_interval_batches: int = 10,
    parallel_backend_jobs: bool = False,
    existing_run_id: str = "",
) -> None:
    if mode not in {"smoke", "full", "asr_bakeoff", "combine", "full_remote", "remote_full"}:
        raise ValueError("mode must be one of: smoke, full, asr_bakeoff, combine, full_remote")

    max_samples = svarah_max_samples or None
    if mode == "smoke":
        max_samples = smoke_samples

    config = AsrBakeoffConfig(
        bakeoff_name=bakeoff_name,
        eval_dataset=DatasetConfig(
            name=eval_dataset,
            config=eval_config_name or None,
            split=eval_split,
            audio_column=eval_audio_column or None,
            text_column=eval_text_column or None,
            max_samples=max_samples,
            trust_remote_code=eval_trust_remote_code,
        ),
        model_specs=_parse_models(bakeoff_models),
        per_device_eval_batch_size=per_device_eval_batch_size,
        distributed_gpu_count=distributed_gpu_count,
        max_new_tokens=max_new_tokens,
        progress_log_interval_batches=progress_log_interval_batches,
    )

    if mode == "combine":
        run_id = existing_run_id.strip()
        if not run_id:
            raise ValueError("--existing-run-id is required for --mode combine")
        report = asr_bakeoff_combine_remote.remote({"run_id": run_id, "config": asdict(config)})
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    run_id = f"{bakeoff_name}-{_now_utc()}"
    if mode in {"full_remote", "remote_full"}:
        report = asr_bakeoff_transformers_full_remote.remote(
            {
                "run_id": run_id,
                "config": asdict(config),
                "model_specs": [asdict(spec) for spec in config.model_specs],
            }
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    report = _run_bakeoff(config, run_id=run_id, parallel_backend_jobs=parallel_backend_jobs)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__" and "--asr-worker-config-path" in sys.argv:
    worker_args = sys.argv[sys.argv.index("--asr-worker-config-path") + 1 :]
    if not worker_args:
        raise SystemExit("--asr-worker-config-path requires a value")
    _asr_worker_main(worker_args[0])
