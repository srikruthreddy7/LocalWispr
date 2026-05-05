"""Modal-native bounded Cohere ASR LoRA training experiment.

This file intentionally does not launch full training from its local
entrypoint. Use ``main`` for smoke, dataset inspection, or config rendering.
The remote ``train_lora_remote`` function is the explicit training entrypoint.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LOCALWISPR_MODAL_COHERE_LORA_APP_NAME", "localwispr-cohere-lora")
BASE_MODEL = os.environ.get(
    "LOCALWISPR_MODAL_COHERE_LORA_BASE_MODEL",
    "CohereLabs/cohere-transcribe-03-2026",
)
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
HF_CACHE_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_HF_CACHE_VOLUME", "localwispr-hf-cache"
)
HF_SECRET_NAME = os.environ.get("LOCALWISPR_MODAL_LORA_HF_SECRET_NAME", "huggingface-secret")
TRAIN_GPU = os.environ.get("LOCALWISPR_MODAL_LORA_TRAIN_GPU", "H100!")

ARTIFACTS_DIR = Path("/artifacts")
HF_CACHE_DIR = Path("/cache/huggingface")
LOCAL_WORK_DIR = Path("/tmp/localwispr-cohere-lora")

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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1", "wget")
    .pip_install(
        "accelerate",
        "datasets[audio]",
        "jiwer",
        "librosa",
        "numpy",
        "peft>=0.14.0",
        "protobuf",
        "sentencepiece",
        "soundfile",
        "torch",
        "torchaudio",
        "transformers>=5.4.0",
    )
    .env(COMMON_ENV)
)

app = modal.App(APP_NAME, image=image)


@dataclass
class DatasetConfig:
    name: str = "WillHeld/india_accent_cv"
    config: str | None = None
    split: str | None = "train"
    audio_column: str | None = None
    text_column: str | None = None
    max_samples: int | None = None
    min_duration_seconds: float | None = 0.2
    max_duration_seconds: float | None = 30.0
    trust_remote_code: bool = False
    require_text: bool = True


@dataclass
class TrainConfig:
    experiment_name: str = "cohere-asr-lora"
    base_model: str = BASE_MODEL
    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    validation_dataset: DatasetConfig | None = None
    train_validation_split: float = 0.1
    train_max_samples: int | None = 64
    validation_max_samples: int | None = 16
    max_steps: int = 20
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 0
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)
    target_module_set: str = "attention"
    lora_scope: str = "encoder"
    gradient_checkpointing: bool = True
    audio_speed_perturb_factors: list[float] = field(default_factory=list)
    audio_speed_perturb_probability: float = 0.0
    audio_gain_jitter_db: float = 0.0
    audio_noise_snr_db: float = 0.0
    optim: str = "adamw_torch"
    seed: int = 42
    language_code: str = "en"
    punctuation: bool = True
    max_new_tokens: int = 128
    progress_log_interval_steps: int = 1
    save_adapter: bool = True
    run_validation: bool = True
    safety_max_train_samples: int = 1024
    safety_max_steps: int = 200
    allow_large_run: bool = False
    smoke: bool = False
    smoke_samples: int = 2


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _sanitize_artifact_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return cleaned.strip("-._") or "artifact"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _write_progress(run_dir: Path, payload: dict[str, Any], *, commit: bool = False) -> None:
    progress = {"updated_at_utc": _now_iso(), **payload}
    _write_json(run_dir / "progress.json", progress)
    if commit:
        artifacts_volume.commit()


def _normalize_dataset_config(value: DatasetConfig | dict[str, Any] | None) -> DatasetConfig | None:
    if value is None:
        return None
    if isinstance(value, DatasetConfig):
        return value
    payload = dict(value)
    return DatasetConfig(**payload)


def _normalize_train_config(value: TrainConfig | dict[str, Any]) -> TrainConfig:
    if isinstance(value, TrainConfig):
        config = value
    else:
        payload = dict(value)
        payload["train_dataset"] = _normalize_dataset_config(payload.get("train_dataset", {}))
        payload["validation_dataset"] = _normalize_dataset_config(payload.get("validation_dataset"))
        if isinstance(payload.get("target_modules"), str):
            payload["target_modules"] = [
                item.strip() for item in payload["target_modules"].split(",") if item.strip()
            ]
        if isinstance(payload.get("audio_speed_perturb_factors"), str):
            payload["audio_speed_perturb_factors"] = [
                float(item.strip())
                for item in payload["audio_speed_perturb_factors"].split(",")
                if item.strip()
            ]
        config = TrainConfig(**payload)

    if config.train_dataset is None:
        raise ValueError("train_dataset is required")

    if config.smoke:
        smoke_samples = max(1, int(config.smoke_samples))
        train_dataset = replace(
            config.train_dataset,
            max_samples=min(config.train_dataset.max_samples or smoke_samples, smoke_samples),
        )
        validation_dataset = None
        if config.validation_dataset is not None:
            validation_dataset = replace(
                config.validation_dataset,
                max_samples=min(
                    config.validation_dataset.max_samples or smoke_samples,
                    smoke_samples,
                ),
            )
        config = replace(
            config,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_max_samples=min(config.train_max_samples or smoke_samples, smoke_samples),
            validation_max_samples=min(config.validation_max_samples or smoke_samples, smoke_samples),
            max_steps=1,
            num_train_epochs=1.0,
            save_adapter=False,
            run_validation=False,
        )

    if not config.allow_large_run:
        if config.train_max_samples is None:
            raise ValueError("train_max_samples must be set unless allow_large_run=True")
        if config.train_max_samples > config.safety_max_train_samples:
            raise ValueError(
                "train_max_samples exceeds safety_max_train_samples. "
                "Set allow_large_run=True only after reviewing the run size."
            )
        if config.max_steps <= 0 or config.max_steps > config.safety_max_steps:
            raise ValueError(
                "max_steps must be in 1..safety_max_steps unless allow_large_run=True"
            )

    if config.rank <= 0:
        raise ValueError("rank must be positive")
    if config.per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    if not 0.0 <= config.audio_speed_perturb_probability <= 1.0:
        raise ValueError("audio_speed_perturb_probability must be in [0, 1]")
    if any(factor <= 0.0 for factor in config.audio_speed_perturb_factors):
        raise ValueError("audio_speed_perturb_factors must all be positive")
    if config.audio_gain_jitter_db < 0.0:
        raise ValueError("audio_gain_jitter_db must be non-negative")
    if config.audio_noise_snr_db < 0.0:
        raise ValueError("audio_noise_snr_db must be non-negative")
    return config


def _resolve_local_manifest_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ARTIFACTS_DIR / raw_path.lstrip("/")


def _local_manifest_exists(raw_path: str) -> bool:
    path = _resolve_local_manifest_path(raw_path)
    return path.exists() and path.is_file() and path.suffix.lower() in {".jsonl", ".json", ".csv"}


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
            canonical[column] = str(ARTIFACTS_DIR / audio_path.lstrip("/"))
        break
    return canonical


def _load_local_manifest(config: DatasetConfig, *, sample_max_hint: int | None, seed: int):
    from datasets import Dataset, load_dataset

    path = _resolve_local_manifest_path(config.name)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rng = random.Random(seed)
        rows: list[dict[str, Any]] = []
        rows_seen = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows_seen += 1
                row = _canonicalize_manifest_row(json.loads(line), audio_column=config.audio_column)
                if sample_max_hint is None or sample_max_hint <= 0:
                    rows.append(row)
                elif len(rows) < sample_max_hint:
                    rows.append(row)
                else:
                    replacement_index = rng.randint(0, rows_seen - 1)
                    if replacement_index < sample_max_hint:
                        rows[replacement_index] = row
        return Dataset.from_list(rows)

    builder_name = "json" if suffix == ".json" else "csv"
    split_name = config.split or "train"
    dataset = load_dataset(builder_name, data_files={split_name: str(path)}, split=split_name)
    return dataset.map(
        lambda row: _canonicalize_manifest_row(row, audio_column=config.audio_column),
        desc="canonicalize local manifest",
    )


def _infer_audio_column(dataset) -> str:
    for column_name, feature in dataset.features.items():
        if feature.__class__.__name__ == "Audio":
            return str(column_name)
    for column_name in ("audio", "audio_path", "audio_filepath", "path", "file", "wav"):
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
        "normalized_transcript",
        "raw_transcript",
        "english_transcription",
    ):
        if column_name in dataset.column_names:
            return column_name
    raise ValueError(f"Could not infer text column from {dataset.column_names}")


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


def _resample_audio(array: Any, sample_rate: int, *, target_rate: int = 16_000) -> tuple[Any, int]:
    audio_array = _normalize_audio_array(array)
    if sample_rate and sample_rate != target_rate:
        import librosa

        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_rate)
        sample_rate = target_rate
    return _normalize_audio_array(audio_array), int(sample_rate or target_rate)


def _decode_audio_path(path: str) -> tuple[Any, int]:
    import soundfile as sf

    try:
        array, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa

        array, sample_rate = librosa.load(path, sr=None, mono=False)
    return _resample_audio(array, int(sample_rate or 16_000))


def _audio_to_array_and_rate(audio: Any) -> tuple[Any, int]:
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            return _resample_audio(
                _normalize_audio_array(audio["array"]),
                int(audio.get("sampling_rate") or 16_000),
            )
        if audio.get("path"):
            return _decode_audio_path(str(audio["path"]))
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        sample_rate = int(getattr(samples, "sample_rate", 16_000) or 16_000)
        return _resample_audio(_normalize_audio_array(samples.data), sample_rate)
    return _resample_audio(_normalize_audio_array(audio), 16_000)


def _row_duration_seconds(row: dict[str, Any], *, audio_column: str) -> float | None:
    for key in ("duration_seconds", "duration_sec", "duration"):
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    if row.get("duration_ms") is not None:
        try:
            return float(row["duration_ms"]) / 1000.0
        except (TypeError, ValueError):
            pass
    try:
        audio_array, sample_rate = _audio_to_array_and_rate(row[audio_column])
        return float(len(audio_array)) / float(sample_rate) if sample_rate else None
    except Exception:
        return None


def _filter_loaded_dataset(dataset, *, config: DatasetConfig, audio_column: str, text_column: str):
    from datasets import Audio

    if audio_column not in dataset.column_names:
        raise ValueError(f"Audio column '{audio_column}' not found in {dataset.column_names}")
    if text_column not in dataset.column_names:
        raise ValueError(f"Text column '{text_column}' not found in {dataset.column_names}")

    if config.require_text:
        dataset = dataset.filter(
            lambda row: bool(str(row.get(text_column) or "").strip()),
            desc="filter empty transcripts",
        )

    audio_feature = dataset.features.get(audio_column)
    if not (audio_feature and audio_feature.__class__.__name__ == "Audio"):
        dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16_000))
    else:
        try:
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16_000))
        except Exception:
            pass

    if config.min_duration_seconds is not None or config.max_duration_seconds is not None:
        min_seconds = config.min_duration_seconds
        max_seconds = config.max_duration_seconds

        def duration_ok(row: dict[str, Any]) -> bool:
            duration = _row_duration_seconds(row, audio_column=audio_column)
            if duration is None:
                return True
            if min_seconds is not None and duration < min_seconds:
                return False
            if max_seconds is not None and duration > max_seconds:
                return False
            return True

        dataset = dataset.filter(duration_ok, desc="filter audio duration")

    if config.max_samples is not None and len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))
    return dataset


def _sample_dataset_rows(dataset, *, max_samples: int | None, seed: int):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def _load_dataset_split(
    config: DatasetConfig,
    *,
    token: str | None,
    sample_max_hint: int | None,
    seed: int,
):
    from datasets import load_dataset

    if _local_manifest_exists(config.name):
        dataset = _load_local_manifest(config, sample_max_hint=sample_max_hint, seed=seed)
    else:
        kwargs: dict[str, Any] = {
            "path": config.name,
            "split": config.split or "train",
            "token": token,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.config:
            kwargs["name"] = config.config
        dataset = load_dataset(**kwargs)
        if sample_max_hint is not None and sample_max_hint > 0:
            dataset = _sample_dataset_rows(dataset, max_samples=sample_max_hint, seed=seed)

    audio_column = config.audio_column or _infer_audio_column(dataset)
    text_column = config.text_column or _infer_text_column(dataset)
    dataset = _filter_loaded_dataset(
        dataset,
        config=config,
        audio_column=audio_column,
        text_column=text_column,
    )
    return dataset, audio_column, text_column


def _resolve_train_validation_splits(
    config: TrainConfig,
    *,
    token: str | None,
    run_dir: Path,
) -> dict[str, Any]:
    _write_progress(
        run_dir,
        {
            "stage": "load_train_split",
            "status": "running",
            "dataset": asdict(config.train_dataset),
        },
        commit=True,
    )
    sample_hint = config.train_max_samples
    if config.validation_dataset is None and config.train_max_samples:
        sample_hint = int(config.train_max_samples / max(0.01, 1.0 - config.train_validation_split)) + 8
    train_dataset, train_audio_column, train_text_column = _load_dataset_split(
        config.train_dataset,
        token=token,
        sample_max_hint=sample_hint,
        seed=config.seed,
    )

    if config.validation_dataset is not None:
        _write_progress(
            run_dir,
            {
                "stage": "load_validation_split",
                "status": "running",
                "dataset": asdict(config.validation_dataset),
            },
            commit=True,
        )
        validation_dataset, validation_audio_column, validation_text_column = _load_dataset_split(
            config.validation_dataset,
            token=token,
            sample_max_hint=config.validation_max_samples,
            seed=config.seed + 1,
        )
        train_split = train_dataset
        validation_summary = {
            "role": "explicit_validation",
            "name": config.validation_dataset.name,
            "config": config.validation_dataset.config,
            "split": config.validation_dataset.split,
            "audio_column": validation_audio_column,
            "text_column": validation_text_column,
        }
    else:
        if len(train_dataset) < 2:
            train_split = train_dataset
            validation_dataset = train_dataset.select([])
        else:
            train_validation = train_dataset.train_test_split(
                test_size=min(max(config.train_validation_split, 0.01), 0.5),
                seed=config.seed,
            )
            train_split = train_validation["train"]
            validation_dataset = train_validation["test"]
        validation_audio_column = train_audio_column
        validation_text_column = train_text_column
        validation_summary = {
            "role": "internal_random_split",
            "name": config.train_dataset.name,
            "config": config.train_dataset.config,
            "split": config.train_dataset.split,
            "audio_column": train_audio_column,
            "text_column": train_text_column,
            "test_size": config.train_validation_split,
        }

    train_split = _sample_dataset_rows(train_split, max_samples=config.train_max_samples, seed=config.seed)
    validation_dataset = _sample_dataset_rows(
        validation_dataset,
        max_samples=config.validation_max_samples,
        seed=config.seed + 1,
    )
    validation_summary["samples"] = len(validation_dataset)

    return {
        "train_split": train_split,
        "train_audio_column": train_audio_column,
        "train_text_column": train_text_column,
        "validation_split": validation_dataset,
        "validation_audio_column": validation_audio_column,
        "validation_text_column": validation_text_column,
        "validation_source_summary": validation_summary,
    }


def _time_resample_audio(array: Any, *, factor: float) -> Any:
    import numpy as np

    audio = np.asarray(array, dtype=np.float32)
    if factor <= 0.0 or audio.shape[0] < 2:
        return audio
    target_length = max(1, int(round(audio.shape[0] / factor)))
    if target_length == audio.shape[0]:
        return audio
    source_positions = np.arange(audio.shape[0], dtype=np.float32)
    target_positions = np.linspace(0, audio.shape[0] - 1, num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _augment_audio_array(
    array: Any,
    *,
    rng: random.Random,
    speed_factors: list[float],
    speed_probability: float,
    gain_jitter_db: float,
    noise_snr_db: float,
) -> Any:
    import numpy as np

    audio = np.asarray(array, dtype=np.float32)
    if speed_factors and rng.random() < speed_probability:
        audio = _time_resample_audio(audio, factor=rng.choice(speed_factors))

    if gain_jitter_db > 0.0:
        gain_db = rng.uniform(-gain_jitter_db, gain_jitter_db)
        audio = audio * float(10 ** (gain_db / 20.0))

    if noise_snr_db > 0.0 and audio.size:
        signal_power = float(np.mean(np.square(audio)))
        if signal_power > 0.0:
            noise_power = signal_power / float(10 ** (noise_snr_db / 10.0))
            noise = np.random.default_rng(rng.randrange(2**32)).normal(
                0.0,
                noise_power**0.5,
                size=audio.shape,
            )
            audio = audio + noise.astype(np.float32)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


class CohereAsrTrainingCollator:
    def __init__(
        self,
        *,
        processor: Any,
        audio_column: str,
        text_column: str,
        language_code: str,
        punctuation: bool,
        audio_speed_perturb_factors: list[float] | None = None,
        audio_speed_perturb_probability: float = 0.0,
        audio_gain_jitter_db: float = 0.0,
        audio_noise_snr_db: float = 0.0,
        augmentation_seed: int = 42,
    ):
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        self.language_code = language_code
        self.punctuation = punctuation
        self.audio_speed_perturb_factors = list(audio_speed_perturb_factors or [])
        self.audio_speed_perturb_probability = float(audio_speed_perturb_probability)
        self.audio_gain_jitter_db = float(audio_gain_jitter_db)
        self.audio_noise_snr_db = float(audio_noise_snr_db)
        self.rng = random.Random(augmentation_seed)
        self.prompt_token_count = self._infer_prompt_token_count()

    def _infer_prompt_token_count(self) -> int:
        import numpy as np

        try:
            prompt_inputs = self.processor(
                np.zeros(16000, dtype=np.float32),
                sampling_rate=16_000,
                return_tensors="pt",
                language=self.language_code,
                punctuation=self.punctuation,
            )
            decoder_input_ids = prompt_inputs.get("decoder_input_ids")
            if decoder_input_ids is not None:
                return int(decoder_input_ids.shape[-1])
        except Exception as exc:
            print(f"[collator] could not infer decoder prompt length: {type(exc).__name__}: {exc}")
        return 0

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        arrays = []
        transcripts = []
        for row in rows:
            array, _ = _audio_to_array_and_rate(row[self.audio_column])
            array = _augment_audio_array(
                array,
                rng=self.rng,
                speed_factors=self.audio_speed_perturb_factors,
                speed_probability=self.audio_speed_perturb_probability,
                gain_jitter_db=self.audio_gain_jitter_db,
                noise_snr_db=self.audio_noise_snr_db,
            )
            arrays.append(array)
            transcripts.append(str(row[self.text_column]))

        batch = self.processor(
            arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            language=self.language_code,
            punctuation=self.punctuation,
            padding=True,
        )

        allowed_keys = {
            "input_features",
            "length",
            "positions",
            "attention_mask",
            "cross_attention_mask",
        }
        output = {key: value for key, value in dict(batch).items() if key in allowed_keys}

        prompt_ids = self.processor.get_decoder_prompt_ids(
            language=self.language_code,
            punctuation=self.punctuation,
        )
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        encoded = self.processor.tokenizer(
            transcripts,
            add_special_tokens=False,
            padding=False,
        )
        text_ids_batch = encoded["input_ids"]
        full_sequences = [
            list(prompt_ids)
            + list(text_ids)
            + ([] if eos_token_id is None else [int(eos_token_id)])
            for text_ids in text_ids_batch
        ]
        max_length = max(len(sequence) for sequence in full_sequences)
        decoder_input_ids = torch.full(
            (len(full_sequences), max_length),
            int(pad_token_id),
            dtype=torch.long,
        )
        decoder_attention_mask = torch.zeros(
            (len(full_sequences), max_length),
            dtype=torch.long,
        )
        labels = torch.full(
            (len(full_sequences), max_length),
            -100,
            dtype=torch.long,
        )
        for row_index, sequence in enumerate(full_sequences):
            sequence_tensor = torch.tensor(sequence, dtype=torch.long)
            sequence_len = int(sequence_tensor.shape[0])
            decoder_input_ids[row_index, :sequence_len] = sequence_tensor
            decoder_attention_mask[row_index, :sequence_len] = 1
            labels[row_index, :sequence_len] = sequence_tensor
            labels[row_index, : len(prompt_ids)] = -100

        output["decoder_input_ids"] = decoder_input_ids
        output["decoder_attention_mask"] = decoder_attention_mask
        output["labels"] = labels

        if output.get("length") is not None:
            output["length"] = output["length"].to(dtype=torch.long)
        if output.get("positions") is not None:
            output["positions"] = output["positions"].to(dtype=torch.long)
        if output.get("attention_mask") is not None:
            output["attention_mask"] = output["attention_mask"].to(dtype=torch.long)
        if output.get("cross_attention_mask") is not None:
            output["cross_attention_mask"] = output["cross_attention_mask"].to(dtype=torch.long)

        return output


def _load_processor_and_model(config: TrainConfig, *, token: str | None):
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(config.base_model, token=token)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return processor, model


def _linear_module_suffixes(model: Any) -> dict[str, int]:
    import torch

    suffix_counts: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return suffix_counts


def _target_module_candidates(model: Any, config: TrainConfig) -> tuple[list[str], dict[str, Any]]:
    suffix_counts = _linear_module_suffixes(model)
    present = set(suffix_counts)

    if config.target_modules:
        targets = [item for item in config.target_modules if item]
        return targets, {
            "selection": "explicit",
            "linear_suffix_counts": suffix_counts,
            "selected": targets,
        }

    attention_names = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "query",
        "key",
        "value",
        "linear_q",
        "linear_k",
        "linear_v",
        "linear_out",
    ]
    mlp_names = [
        "fc1",
        "fc2",
        "gate_proj",
        "up_proj",
        "down_proj",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    normalized = config.target_module_set.strip().lower()
    if normalized == "attention":
        desired = attention_names
    elif normalized == "full":
        desired = attention_names + mlp_names
    else:
        raise ValueError("target_module_set must be one of: attention, full")

    targets = [name for name in desired if name in present]
    if not targets:
        blocked = {"lm_head", "embed_tokens", "embed_positions"}
        targets = [
            suffix
            for suffix, _ in sorted(suffix_counts.items(), key=lambda item: (-item[1], item[0]))
            if suffix not in blocked and "embed" not in suffix.lower()
        ][:12]
    if not targets:
        raise ValueError("No linear modules were discovered for LoRA targeting")

    return targets, {
        "selection": "auto",
        "target_module_set": normalized,
        "linear_suffix_counts": suffix_counts,
        "selected": targets,
    }


def _apply_lora(model: Any, config: TrainConfig) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model

    targets, target_report = _target_module_candidates(model, config)
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)

    scope = config.lora_scope.strip().lower()
    if scope not in {"all", "encoder", "decoder"}:
        raise ValueError("lora_scope must be one of: all, encoder, decoder")
    if scope != "all":
        for name, param in model.named_parameters():
            if not (param.requires_grad and "lora_" in name):
                continue
            normalized_name = name.lower()
            in_encoder = ".encoder." in normalized_name or normalized_name.startswith("encoder.")
            in_decoder = ".decoder." in normalized_name or normalized_name.startswith("decoder.")
            if scope == "encoder" and not in_encoder:
                param.requires_grad = False
            if scope == "decoder" and not in_decoder:
                param.requires_grad = False

    if config.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    if trainable == 0:
        raise ValueError(
            "LoRA configuration produced zero trainable parameters. "
            "Check target_modules and lora_scope against the Cohere ASR module names."
        )

    return model, {
        "target_modules": target_report,
        "lora_scope": scope,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def _build_training_arguments(config: TrainConfig, *, output_dir: Path):
    import inspect
    import torch
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "logging_steps": max(1, config.progress_log_interval_steps),
        "save_strategy": "no",
        "save_total_limit": 1,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "bf16": bool(torch.cuda.is_available()),
        "fp16": False,
        "gradient_checkpointing": config.gradient_checkpointing,
        "report_to": [],
        "load_best_model_at_end": False,
        "seed": config.seed,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "optim": config.optim,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if config.gradient_checkpointing and "gradient_checkpointing_kwargs" in signature:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kwargs)


class _ArtifactProgressCallback:
    def __init__(self, *, run_dir: Path, commit_every_steps: int):
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            def __init__(self, parent: "_ArtifactProgressCallback"):
                self.parent = parent

            def on_train_begin(self, args, state, control, **kwargs):
                self.parent.write(state, logs={}, status="running", commit=True)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not state.is_world_process_zero:
                    return
                should_commit = (
                    int(state.global_step) - self.parent.last_committed_step
                ) >= self.parent.commit_every_steps
                if should_commit:
                    self.parent.last_committed_step = int(state.global_step)
                self.parent.write(state, logs=dict(logs or {}), status="running", commit=should_commit)

            def on_train_end(self, args, state, control, **kwargs):
                if not state.is_world_process_zero:
                    return
                self.parent.write(state, logs={}, status="complete", commit=True)

        self.impl = _Impl(self)
        self.run_dir = run_dir
        self.commit_every_steps = max(1, commit_every_steps)
        self.last_committed_step = 0

    def write(self, state: Any, *, logs: dict[str, Any], status: str, commit: bool) -> None:
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        global_step = int(getattr(state, "global_step", 0) or 0)
        payload = {
            "stage": "train",
            "status": status,
            "global_step": global_step,
            "max_steps": max_steps,
            "percent_complete": (global_step / max_steps) if max_steps else None,
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "logs": _json_safe(logs),
        }
        _write_progress(self.run_dir, payload, commit=commit)


def _metadata_preview(dataset, *, audio_column: str, text_column: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    metadata_dataset = dataset.remove_columns([audio_column])
    for index in range(min(limit, len(dataset))):
        row = dict(metadata_dataset[index])
        row["index"] = index
        row["text_preview"] = str(row.get(text_column) or "")[:240]
        rows.append(_json_safe(row))
    return rows


def _text_normalizer():
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        return BasicTextNormalizer()
    except Exception:
        punctuation = re.compile(r"[^\w\s']+", flags=re.UNICODE)

        def normalize(text: str) -> str:
            return re.sub(r"\s+", " ", punctuation.sub(" ", str(text).lower())).strip()

        return normalize


def _move_batch_to_model(inputs: Any, *, device: Any, dtype: Any) -> Any:
    for key, value in list(inputs.items()):
        if not hasattr(value, "to"):
            continue
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            inputs[key] = value.to(device=device, dtype=dtype)
        else:
            inputs[key] = value.to(device=device)
    return inputs


def _generate_validation_preview(
    *,
    model: Any,
    processor: Any,
    dataset,
    audio_column: str,
    text_column: str,
    config: TrainConfig,
    sample_limit: int = 4,
) -> dict[str, Any]:
    import torch

    if not len(dataset):
        return {"samples": 0, "rows": []}
    normalizer = _text_normalizer()
    sample_count = min(sample_limit, len(dataset))
    rows = []
    model.eval()
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    for index in range(sample_count):
        row = dataset[index]
        array, _ = _audio_to_array_and_rate(row[audio_column])
        inputs = processor(
            array,
            sampling_rate=16_000,
            return_tensors="pt",
            language=config.language_code,
            punctuation=config.punctuation,
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = _move_batch_to_model(inputs, device=model_device, dtype=model_dtype)
        try:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
        except ValueError as exc:
            if "['length']" not in str(exc):
                raise
            inputs.pop("length", None)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
        decoded = processor.decode(
            generated_ids,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=config.language_code,
        )
        prediction = decoded if isinstance(decoded, str) else str(decoded[0] if decoded else "")
        reference = str(row[text_column])
        rows.append(
            {
                "index": index,
                "reference": reference,
                "prediction": prediction,
                "normalized_reference": normalizer(reference).strip(),
                "normalized_prediction": normalizer(prediction).strip(),
            }
        )
    model.train()
    return {"samples": sample_count, "rows": rows}


def _train_lora_impl(config: TrainConfig) -> dict[str, Any]:
    import torch
    from transformers import Trainer, set_seed

    config = _normalize_train_config(config)
    set_seed(config.seed)
    token = _get_hf_token()
    run_id = f"{_sanitize_artifact_component(config.experiment_name)}-{_now_utc()}"
    run_dir = ARTIFACTS_DIR / run_id
    output_dir = LOCAL_WORK_DIR / run_id / "trainer"
    adapter_dir = run_dir / "adapter"
    _ensure_dir(run_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _ensure_dir(output_dir)

    _write_json(run_dir / "train_config.json", asdict(config))
    _write_progress(
        run_dir,
        {
            "stage": "startup",
            "status": "running",
            "run_id": run_id,
            "base_model": config.base_model,
            "smoke": config.smoke,
            "gpu": os.environ.get("MODAL_GPU_LABEL", TRAIN_GPU),
        },
        commit=True,
    )

    started = time.monotonic()
    try:
        split_payload = _resolve_train_validation_splits(config, token=token, run_dir=run_dir)
        train_split = split_payload["train_split"]
        validation_split = split_payload["validation_split"]
        train_audio_column = split_payload["train_audio_column"]
        train_text_column = split_payload["train_text_column"]
        validation_audio_column = split_payload["validation_audio_column"]
        validation_text_column = split_payload["validation_text_column"]

        preview_rows = _metadata_preview(
            train_split,
            audio_column=train_audio_column,
            text_column=train_text_column,
        )
        _append_jsonl(run_dir / "train_preview.jsonl", preview_rows)

        _write_progress(
            run_dir,
            {
                "stage": "load_model",
                "status": "running",
                "train_rows": len(train_split),
                "validation_rows": len(validation_split),
            },
            commit=True,
        )
        processor, model = _load_processor_and_model(config, token=token)
        model, lora_report = _apply_lora(model, config)

        if torch.cuda.is_available():
            model.to("cuda")

        train_collator = CohereAsrTrainingCollator(
            processor=processor,
            audio_column=train_audio_column,
            text_column=train_text_column,
            language_code=config.language_code,
            punctuation=config.punctuation,
            audio_speed_perturb_factors=config.audio_speed_perturb_factors,
            audio_speed_perturb_probability=config.audio_speed_perturb_probability,
            audio_gain_jitter_db=config.audio_gain_jitter_db,
            audio_noise_snr_db=config.audio_noise_snr_db,
            augmentation_seed=config.seed + 17,
        )
        eval_collator = CohereAsrTrainingCollator(
            processor=processor,
            audio_column=validation_audio_column,
            text_column=validation_text_column,
            language_code=config.language_code,
            punctuation=config.punctuation,
        )

        args = _build_training_arguments(config, output_dir=output_dir)
        progress_callback = _ArtifactProgressCallback(
            run_dir=run_dir,
            commit_every_steps=max(1, config.progress_log_interval_steps),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_split,
            eval_dataset=validation_split if len(validation_split) else None,
            data_collator=train_collator,
            callbacks=[progress_callback.impl],
        )

        _write_progress(
            run_dir,
            {
                "stage": "train",
                "status": "running",
                "train_rows": len(train_split),
                "max_steps": config.max_steps,
                "effective_batch_size": config.per_device_train_batch_size
                * config.gradient_accumulation_steps,
                "lora": lora_report,
            },
            commit=True,
        )
        train_output = trainer.train()
        train_metrics = _json_safe(train_output.metrics)

        eval_metrics = None
        preview = None
        if config.run_validation and len(validation_split):
            _write_progress(
                run_dir,
                {"stage": "validation", "status": "running", "samples": len(validation_split)},
                commit=True,
            )
            trainer.data_collator = eval_collator
            eval_metrics = _json_safe(trainer.evaluate(eval_dataset=validation_split))
            preview = _generate_validation_preview(
                model=model,
                processor=processor,
                dataset=validation_split,
                audio_column=validation_audio_column,
                text_column=validation_text_column,
                config=config,
            )
            _write_json(run_dir / "validation_preview.json", preview)

        if config.save_adapter:
            _ensure_dir(adapter_dir)
            model.save_pretrained(str(adapter_dir))
            processor.save_pretrained(str(run_dir / "processor"))

        report = {
            "run_id": run_id,
            "created_at_utc": _now_iso(),
            "status": "complete",
            "config": asdict(config),
            "model": {
                "base_model": config.base_model,
                "dtype": "bfloat16",
                "lora": lora_report,
            },
            "dataset": {
                "train": {
                    "name": config.train_dataset.name,
                    "config": config.train_dataset.config,
                    "split": config.train_dataset.split,
                    "audio_column": train_audio_column,
                    "text_column": train_text_column,
                    "samples": len(train_split),
                },
                "validation": {
                    **split_payload["validation_source_summary"],
                    "audio_column": validation_audio_column,
                    "text_column": validation_text_column,
                    "samples": len(validation_split),
                },
            },
            "metrics": {
                "train": train_metrics,
                "validation": eval_metrics,
                "preview": preview,
            },
            "timing": {
                "wall_seconds": time.monotonic() - started,
            },
            "artifacts": {
                "run_dir": str(run_dir),
                "progress_path": str(run_dir / "progress.json"),
                "report_path": str(run_dir / "report.json"),
                "train_config": str(run_dir / "train_config.json"),
                "train_preview_jsonl": str(run_dir / "train_preview.jsonl"),
                "adapter_dir": str(adapter_dir) if config.save_adapter else None,
                "processor_dir": str(run_dir / "processor") if config.save_adapter else None,
                "validation_preview": str(run_dir / "validation_preview.json") if preview else None,
            },
        }
        _write_json(run_dir / "report.json", report)
        _write_progress(
            run_dir,
            {
                "stage": "complete",
                "status": "complete",
                "run_id": run_id,
                "report_path": str(run_dir / "report.json"),
                "adapter_dir": str(adapter_dir) if config.save_adapter else None,
                "train_metrics": train_metrics,
                "validation_metrics": eval_metrics,
            },
            commit=True,
        )
        artifacts_volume.commit()
        hf_cache_volume.commit()
        return report
    except Exception as exc:
        error_report = {
            "run_id": run_id,
            "created_at_utc": _now_iso(),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "config": asdict(config),
            "artifacts": {
                "run_dir": str(run_dir),
                "progress_path": str(run_dir / "progress.json"),
                "report_path": str(run_dir / "report.json"),
            },
        }
        _write_json(run_dir / "report.json", error_report)
        _write_progress(
            run_dir,
            {
                "stage": "failed",
                "status": "failed",
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "report_path": str(run_dir / "report.json"),
            },
            commit=True,
        )
        artifacts_volume.commit()
        raise


def _inspect_dataset_impl(config: DatasetConfig) -> dict[str, Any]:
    token = _get_hf_token()
    dataset, audio_column, text_column = _load_dataset_split(
        config,
        token=token,
        sample_max_hint=config.max_samples,
        seed=42,
    )
    dataset = _sample_dataset_rows(dataset, max_samples=config.max_samples, seed=42)
    samples = []
    for index in range(min(5, len(dataset))):
        row = dataset[index]
        duration = _row_duration_seconds(row, audio_column=audio_column)
        samples.append(
            {
                "index": index,
                "text": str(row[text_column])[:240],
                "duration_seconds": duration,
            }
        )
    return {
        "dataset": asdict(config),
        "rows": len(dataset),
        "columns": dataset.column_names,
        "audio_column": audio_column,
        "text_column": text_column,
        "samples": samples,
    }


@app.function(
    timeout=60 * 60 * 8,
    gpu=TRAIN_GPU,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def train_lora_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = TRAIN_GPU
    config = _normalize_train_config(config_payload)
    return _train_lora_impl(config)


@app.function(
    timeout=60 * 60,
    gpu=TRAIN_GPU,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def smoke_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = TRAIN_GPU
    config = _normalize_train_config({**config_payload, "smoke": True})
    return _train_lora_impl(config)


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
    if config is None:
        raise ValueError("dataset_payload is required")
    return _inspect_dataset_impl(config)


def _dataset_config_from_args(
    *,
    dataset: str,
    config_name: str,
    split: str,
    audio_column: str,
    text_column: str,
    max_samples: int,
    min_duration_seconds: float,
    max_duration_seconds: float,
    trust_remote_code: bool,
    require_text: bool,
) -> DatasetConfig:
    return DatasetConfig(
        name=dataset,
        config=config_name or None,
        split=split or None,
        audio_column=audio_column or None,
        text_column=text_column or None,
        max_samples=max_samples or None,
        min_duration_seconds=min_duration_seconds if min_duration_seconds > 0 else None,
        max_duration_seconds=max_duration_seconds if max_duration_seconds > 0 else None,
        trust_remote_code=trust_remote_code,
        require_text=require_text,
    )


@app.local_entrypoint()
def main(
    mode: str = "print_config",
    experiment_name: str = "cohere-asr-lora",
    train_dataset: str = "WillHeld/india_accent_cv",
    train_config_name: str = "",
    train_split: str = "train",
    train_audio_column: str = "",
    train_text_column: str = "",
    train_max_samples: int = 64,
    train_min_duration_seconds: float = 0.2,
    train_max_duration_seconds: float = 30.0,
    train_trust_remote_code: bool = False,
    train_require_text: bool = True,
    validation_dataset: str = "",
    validation_config_name: str = "",
    validation_split: str = "validation",
    validation_audio_column: str = "",
    validation_text_column: str = "",
    validation_max_samples: int = 16,
    validation_min_duration_seconds: float = 0.2,
    validation_max_duration_seconds: float = 30.0,
    validation_trust_remote_code: bool = False,
    validation_require_text: bool = True,
    max_steps: int = 20,
    num_train_epochs: float = 1.0,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: str = "",
    target_module_set: str = "attention",
    lora_scope: str = "encoder",
    gradient_checkpointing: bool = True,
    audio_speed_perturb_factors: str = "",
    audio_speed_perturb_probability: float = 0.0,
    audio_gain_jitter_db: float = 0.0,
    audio_noise_snr_db: float = 0.0,
    warmup_steps: int = 0,
    weight_decay: float = 0.0,
    optim: str = "adamw_torch",
    dataloader_num_workers: int = 0,
    language_code: str = "en",
    punctuation: bool = True,
    max_new_tokens: int = 128,
    save_adapter: bool = True,
    run_validation: bool = True,
    allow_large_run: bool = False,
    smoke_samples: int = 2,
) -> None:
    train_dataset_config = _dataset_config_from_args(
        dataset=train_dataset,
        config_name=train_config_name,
        split=train_split,
        audio_column=train_audio_column,
        text_column=train_text_column,
        max_samples=train_max_samples,
        min_duration_seconds=train_min_duration_seconds,
        max_duration_seconds=train_max_duration_seconds,
        trust_remote_code=train_trust_remote_code,
        require_text=train_require_text,
    )
    validation_dataset_config = None
    if validation_dataset:
        validation_dataset_config = _dataset_config_from_args(
            dataset=validation_dataset,
            config_name=validation_config_name,
            split=validation_split,
            audio_column=validation_audio_column,
            text_column=validation_text_column,
            max_samples=validation_max_samples,
            min_duration_seconds=validation_min_duration_seconds,
            max_duration_seconds=validation_max_duration_seconds,
            trust_remote_code=validation_trust_remote_code,
            require_text=validation_require_text,
        )

    config = TrainConfig(
        experiment_name=experiment_name,
        train_dataset=train_dataset_config,
        validation_dataset=validation_dataset_config,
        train_max_samples=train_max_samples or None,
        validation_max_samples=validation_max_samples or None,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=[item.strip() for item in target_modules.split(",") if item.strip()],
        target_module_set=target_module_set,
        lora_scope=lora_scope,
        gradient_checkpointing=gradient_checkpointing,
        audio_speed_perturb_factors=[
            float(item.strip()) for item in audio_speed_perturb_factors.split(",") if item.strip()
        ],
        audio_speed_perturb_probability=audio_speed_perturb_probability,
        audio_gain_jitter_db=audio_gain_jitter_db,
        audio_noise_snr_db=audio_noise_snr_db,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        optim=optim,
        dataloader_num_workers=dataloader_num_workers,
        language_code=language_code,
        punctuation=punctuation,
        max_new_tokens=max_new_tokens,
        save_adapter=save_adapter,
        run_validation=run_validation,
        allow_large_run=allow_large_run,
        smoke_samples=smoke_samples,
    )

    normalized_mode = mode.strip().lower()
    if normalized_mode == "inspect":
        payload = inspect_dataset_remote.remote(asdict(train_dataset_config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if normalized_mode == "smoke":
        payload = smoke_remote.remote(asdict(config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if normalized_mode in {"train", "train_lora"}:
        payload = train_lora_remote.remote(asdict(config))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if normalized_mode == "print_config":
        print(json.dumps(asdict(config), indent=2, sort_keys=True))
        return

    raise ValueError(
        "Unsupported mode. This local entrypoint intentionally supports only "
        "smoke, inspect, print_config, and explicit train/train_lora."
    )
