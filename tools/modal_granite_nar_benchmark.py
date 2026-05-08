"""Modal benchmark for Granite Speech 4.1 NAR on Svarah."""

from __future__ import annotations

import json
import io
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LOCALWISPR_MODAL_GRANITE_NAR_APP_NAME", "localwispr-granite-nar-benchmark")
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
HF_CACHE_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_HF_CACHE_VOLUME", "localwispr-hf-cache"
)
HF_SECRET_NAME = os.environ.get("LOCALWISPR_MODAL_LORA_HF_SECRET_NAME", "huggingface-secret")
GPU = os.environ.get("LOCALWISPR_MODAL_LORA_TRAIN_GPU", "H100")

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

granite_nar_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libsndfile1", "wget")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "
        '"torch==2.9.1" "torchaudio==2.9.1"'
    )
    .run_commands(
        'uv pip install --system "accelerate==1.13.0" "datasets[audio]" "jiwer" '
        '"librosa" "numpy" "safetensors==0.7.0" "soundfile" '
        '"tokenizers==0.22.2" "transformers==4.57.6" "packaging" "psutil" "wheel"'
    )
    .run_commands(
        "uv pip install --system "
        '"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/'
        'flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"'
    )
    .env(
        {
            **COMMON_ENV,
            "CUDA_HOME": "/usr/local/cuda",
            "MAX_JOBS": "2",
            "TORCH_CUDA_ARCH_LIST": "9.0",
        }
    )
)

app = modal.App(APP_NAME)


@dataclass
class GraniteNarConfig:
    benchmark_name: str = "granite-nar-svarah"
    model_name: str = "ibm-granite/granite-speech-4.1-2b-nar"
    eval_dataset: str = "ai4bharat/Svarah"
    eval_config_name: str | None = None
    eval_split: str = "test"
    eval_audio_column: str | None = None
    eval_text_column: str | None = None
    max_samples: int | None = None
    batch_size: int = 64
    progress_log_interval_batches: int = 4
    attn_implementation: str = "flash_attention_2"


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _run_dir(run_id: str) -> Path:
    return ARTIFACTS_DIR / run_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _infer_audio_column(dataset) -> str:
    for column_name in ("audio", "audio_filepath", "path", "file", "wav"):
        if column_name in dataset.column_names:
            return column_name
    raise ValueError(f"Could not infer audio column from {dataset.column_names}")


def _infer_text_column(dataset) -> str:
    for column_name in (
        "text",
        "transcription",
        "transcript",
        "sentence",
        "normalized_text",
        "english_text",
        "english_transcription",
    ):
        if column_name in dataset.column_names:
            return column_name
    raise ValueError(f"Could not infer text column from {dataset.column_names}")


def _text_normalizer():
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        return BasicTextNormalizer()
    except Exception:
        import re

        def normalize(text: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s']", " ", text.lower())).strip()

        return normalize


def _compute_text_metrics(references: list[str], predictions: list[str]) -> dict[str, Any]:
    from jiwer import cer, wer

    filtered_references: list[str] = []
    filtered_predictions: list[str] = []
    for reference, prediction in zip(references, predictions):
        if str(reference).strip():
            filtered_references.append(reference)
            filtered_predictions.append(prediction)
    return {
        "wer": wer(filtered_references, filtered_predictions),
        "cer": cer(filtered_references, filtered_predictions),
        "scored_samples": len(filtered_references),
        "skipped_empty_references": len(references) - len(filtered_references),
    }


def _summarize_numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _decode_audio(audio: Any):
    import numpy as np

    if isinstance(audio, dict):
        audio_bytes = audio.get("bytes")
        audio_path = audio.get("path")
    else:
        audio_bytes = None
        audio_path = audio

    if audio_bytes:
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(audio_bytes)) as sound_file:
            array = sound_file.read(dtype="float32", always_2d=False)
            sample_rate = int(sound_file.samplerate)
    elif audio_path:
        import librosa

        array, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
    else:
        raise ValueError(f"Unsupported audio payload: {type(audio)!r}")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=0)
    if int(sample_rate or 16_000) != 16_000:
        import librosa

        array = librosa.resample(array, orig_sr=int(sample_rate), target_sr=16_000)
    return array.reshape(-1)


@app.function(
    image=granite_nar_image,
    gpu=GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def granite_nar_benchmark_remote(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from datasets import Audio, load_dataset
    from transformers import AutoFeatureExtractor, AutoModel

    config = GraniteNarConfig(**payload)
    run_id = f"{config.benchmark_name}-{_now_utc()}"
    run_dir = _run_dir(run_id)
    progress_path = run_dir / "progress.json"
    report_path = run_dir / "report.json"
    token = _get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    run_started = time.monotonic()
    _write_json(progress_path, {"run_id": run_id, "stage": "loading_dataset", "status": "running"})
    artifacts_volume.commit()

    dataset_kwargs: dict[str, Any] = {"path": config.eval_dataset, "split": config.eval_split}
    if config.eval_config_name:
        dataset_kwargs["name"] = config.eval_config_name
    if token:
        dataset_kwargs["token"] = token
    dataset = load_dataset(**dataset_kwargs)
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    audio_column = config.eval_audio_column or _infer_audio_column(dataset)
    text_column = config.eval_text_column or _infer_text_column(dataset)
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16_000, decode=False))

    _write_json(
        progress_path,
        {
            "run_id": run_id,
            "stage": "loading_model",
            "status": "running",
            "samples_total": len(dataset),
            "updated_at_utc": _now_iso(),
        },
    )
    artifacts_volume.commit()
    model_load_started = time.monotonic()
    model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        attn_implementation=config.attn_implementation,
        device_map="cuda",
        dtype=torch.bfloat16,
    ).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name, trust_remote_code=True)
    model_load_seconds = time.monotonic() - model_load_started

    predictions: list[str] = []
    references: list[str] = []
    audio_seconds: list[float] = []
    batch_seconds: list[float] = []
    inference_started = time.monotonic()
    batch_size = max(1, config.batch_size)
    normalizer = _text_normalizer()
    pairwise_path = run_dir / "pairwise_predictions.jsonl"

    for batch_start in range(0, len(dataset), batch_size):
        batch = dataset[batch_start : batch_start + batch_size]
        arrays = [_decode_audio(audio) for audio in batch[audio_column]]
        refs = [str(value or "") for value in batch[text_column]]
        seconds = [float(len(array)) / 16_000.0 for array in arrays]
        waveforms = [torch.from_numpy(array) for array in arrays]
        batch_started = time.monotonic()
        with torch.inference_mode():
            inputs = feature_extractor(waveforms, device="cuda")
            output = model.generate(**inputs)
        batch_seconds.append(time.monotonic() - batch_started)
        texts = [str(text or "") for text in output.text_preds]
        predictions.extend(texts)
        references.extend(refs)
        audio_seconds.extend(seconds)

        rows = []
        for offset, text in enumerate(texts):
            index = batch_start + offset
            rows.append(
                {
                    "index": index,
                    "reference": refs[offset],
                    "normalized_reference": normalizer(refs[offset]),
                    "granite_nar__prediction": text,
                    "granite_nar__normalized_prediction": normalizer(text),
                    "audio_seconds": seconds[offset],
                }
            )
        _append_jsonl(pairwise_path, rows)

        batches_done = (batch_start // batch_size) + 1
        batches_total = (len(dataset) + batch_size - 1) // batch_size
        elapsed = time.monotonic() - inference_started
        samples_done = len(predictions)
        eta_seconds = (elapsed / samples_done) * (len(dataset) - samples_done) if samples_done else None
        progress = {
            "run_id": run_id,
            "stage": "transcribing",
            "status": "running" if samples_done < len(dataset) else "complete",
            "samples_done": samples_done,
            "samples_total": len(dataset),
            "batches_done": batches_done,
            "batches_total": batches_total,
            "percent_complete": samples_done / len(dataset) if len(dataset) else 1.0,
            "audio_seconds_done": sum(audio_seconds),
            "elapsed_seconds": elapsed,
            "current_throughput_rtfx": sum(audio_seconds) / elapsed if elapsed > 0 else None,
            "eta_seconds": eta_seconds,
            "updated_at_utc": _now_iso(),
        }
        _write_json(progress_path, progress)
        artifacts_volume.commit()
        if batches_done % max(1, config.progress_log_interval_batches) == 0 or samples_done == len(dataset):
            print(
                f"[granite-nar] batches {batches_done}/{batches_total} "
                f"samples {samples_done}/{len(dataset)} rtfx={progress['current_throughput_rtfx']:.2f} "
                f"eta={eta_seconds}",
                flush=True,
            )

    inference_seconds = time.monotonic() - inference_started
    normalized_references = [normalizer(text).strip() for text in references]
    normalized_predictions = [normalizer(text).strip() for text in predictions]
    metrics = _compute_text_metrics(normalized_references, normalized_predictions)
    report = {
        "run_id": run_id,
        "created_at_utc": _now_iso(),
        "config": asdict(config),
        "dataset": {
            "name": config.eval_dataset,
            "config": config.eval_config_name,
            "split": config.eval_split,
            "audio_column": audio_column,
            "text_column": text_column,
            "samples": len(dataset),
            "audio_seconds_total": sum(audio_seconds),
        },
        "model": {
            "label": "granite_speech_4p1_2b_nar",
            "model_name": config.model_name,
            "backend": f"transformers_{config.attn_implementation}",
        },
        "metrics": metrics,
        "timing": {
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "end_to_end_seconds": time.monotonic() - run_started,
            "throughput_rtfx": sum(audio_seconds) / inference_seconds if inference_seconds > 0 else None,
            "end_to_end_rtfx": sum(audio_seconds) / (time.monotonic() - run_started),
            "batch_latency_seconds": _summarize_numeric(batch_seconds),
            "audio_seconds": _summarize_numeric(audio_seconds),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "progress_path": str(progress_path),
            "report_path": str(report_path),
            "pairwise_predictions_jsonl": str(pairwise_path),
        },
    }
    _write_json(report_path, report)
    _write_json(progress_path, {**report, "stage": "complete", "status": "complete"})
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


@app.local_entrypoint()
def main(
    benchmark_name: str = "granite-nar-svarah-full",
    model_name: str = "ibm-granite/granite-speech-4.1-2b-nar",
    eval_dataset: str = "ai4bharat/Svarah",
    eval_config_name: str = "",
    eval_split: str = "test",
    eval_audio_column: str = "",
    eval_text_column: str = "",
    max_samples: int = 0,
    batch_size: int = 64,
    progress_log_interval_batches: int = 4,
    attn_implementation: str = "flash_attention_2",
) -> None:
    config = GraniteNarConfig(
        benchmark_name=benchmark_name,
        model_name=model_name,
        eval_dataset=eval_dataset,
        eval_config_name=eval_config_name or None,
        eval_split=eval_split,
        eval_audio_column=eval_audio_column or None,
        eval_text_column=eval_text_column or None,
        max_samples=max_samples or None,
        batch_size=batch_size,
        progress_log_interval_batches=progress_log_interval_batches,
        attn_implementation=attn_implementation,
    )
    report = granite_nar_benchmark_remote.remote(asdict(config))
    print(json.dumps(report, indent=2, sort_keys=True))
