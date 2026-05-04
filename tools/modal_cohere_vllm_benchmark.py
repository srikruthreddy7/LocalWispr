"""Modal benchmark for ASR models served through vLLM.

This is separate from ``modal_asr_bakeoff.py`` because the vLLM path is a
newer stack benchmark: start an OpenAI-compatible transcription server or run
offline vLLM inference, then score the returned text.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LOCALWISPR_MODAL_COHERE_VLLM_APP_NAME", "localwispr-cohere-vllm-benchmark")
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME", "localwispr-whisper-lora-artifacts"
)
HF_CACHE_VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_LORA_HF_CACHE_VOLUME", "localwispr-hf-cache"
)
HF_SECRET_NAME = os.environ.get("LOCALWISPR_MODAL_LORA_HF_SECRET_NAME", "huggingface-secret")
GPU = os.environ.get("LOCALWISPR_MODAL_LORA_TRAIN_GPU", "H100!")

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
    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "64",
    "VLLM_USE_DEEP_GEMM": "0",
    "VLLM_USE_DEEP_GEMM_E8M0": "0",
}

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ffmpeg", "git", "libsndfile1", "wget")
    .pip_install("uv")
    .run_commands('uv pip install --system -U "vllm[audio]==0.20.0"')
    .run_commands(
        'uv pip install --system "datasets[audio]" "httpx" "jiwer" "librosa" '
        '"mistral-common[audio]>=1.8.1" "numpy" "requests" "soundfile" '
        '"transformers>=5.4.0" "websockets"'
    )
    .env(COMMON_ENV)
)

app = modal.App(APP_NAME)


@dataclass
class BenchmarkConfig:
    benchmark_name: str = "cohere-vllm-benchmark"
    model_name: str = "CohereLabs/cohere-transcribe-03-2026"
    model_label: str | None = None
    model_family: str | None = None
    eval_dataset: str = "ai4bharat/Svarah"
    eval_config_name: str | None = None
    eval_split: str = "test"
    eval_audio_column: str | None = None
    eval_text_column: str | None = None
    eval_trust_remote_code: bool = False
    max_samples: int | None = 64
    language_code: str = "en"
    punctuation: bool = True
    concurrency: int = 16
    request_timeout_seconds: float = 300.0
    port: int = 8000
    api_key: str = "localwispr-vllm"
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.90
    startup_timeout_seconds: float = 900.0
    progress_log_interval_samples: int = 8
    compare_report_run_id: str | None = None
    vllm_mode: str = "offline"
    max_tokens: int = 256
    offline_batch_size: int = 512
    max_model_len: int | None = None
    temperature: float = 0.0
    enable_lora: bool = False
    max_lora_rank: int = 64


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


def _model_label(config: BenchmarkConfig) -> str:
    if config.model_label:
        return _sanitize_artifact_component(config.model_label)
    return _sanitize_artifact_component(config.model_name.split("/")[-1]).lower()


def _model_family(config: BenchmarkConfig) -> str:
    if config.model_family:
        return config.model_family.strip().lower()
    model_name = config.model_name.lower()
    if "cohere-transcribe" in model_name:
        return "cohere_asr"
    if "voxtral" in model_name and "realtime" in model_name:
        return "voxtral_realtime"
    if "voxtral" in model_name:
        return "voxtral"
    if "granite" in model_name and "speech" in model_name:
        return "granite_speech"
    return "generic_transcription"


def _run_dir(run_id: str) -> Path:
    return ARTIFACTS_DIR / run_id


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
        if audio.get("array") is not None:
            return _resample_audio(
                _normalize_audio_array(audio["array"]),
                int(audio.get("sampling_rate") or 16_000),
            )
        if audio.get("bytes") is not None:
            return _decode_audio_bytes(audio["bytes"])
        if audio.get("path"):
            return _decode_audio_path(str(audio["path"]))
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        sample_rate = int(getattr(samples, "sample_rate", 16_000) or 16_000)
        return _resample_audio(_normalize_audio_array(samples.data), sample_rate)
    return _resample_audio(_normalize_audio_array(audio), 16_000)


def _resample_audio(array: Any, sample_rate: int, *, target_rate: int = 16_000) -> tuple[Any, int]:
    audio_array = _normalize_audio_array(array)
    if sample_rate and sample_rate != target_rate:
        import librosa

        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_rate)
        sample_rate = target_rate
    return _normalize_audio_array(audio_array), int(sample_rate or target_rate)


def _decode_audio_bytes(audio_bytes: bytes) -> tuple[Any, int]:
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(audio_bytes)) as sound_file:
        array = sound_file.read(dtype="float32", always_2d=False)
        sample_rate = int(sound_file.samplerate)
    return _resample_audio(array, sample_rate)


def _decode_audio_path(path: str) -> tuple[Any, int]:
    import soundfile as sf

    try:
        array, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa

        array, sample_rate = librosa.load(path, sr=None, mono=False)
    return _resample_audio(array, int(sample_rate or 16_000))


def _text_normalizer():
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        return BasicTextNormalizer()
    except Exception:
        punctuation = re.compile(r"[^\w\s']+", flags=re.UNICODE)

        def normalize(text: str) -> str:
            return re.sub(r"\s+", " ", punctuation.sub(" ", str(text).lower())).strip()

        return normalize


def _load_dataset_split(config: BenchmarkConfig, *, token: str | None):
    from datasets import Audio, load_dataset

    kwargs: dict[str, Any] = {
        "path": config.eval_dataset,
        "split": config.eval_split,
        "token": token,
        "trust_remote_code": config.eval_trust_remote_code,
    }
    if config.eval_config_name:
        kwargs["name"] = config.eval_config_name
    dataset = load_dataset(**kwargs)
    if config.max_samples and len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))

    audio_column = config.eval_audio_column or _infer_audio_column(dataset)
    text_column = config.eval_text_column or _infer_text_column(dataset)
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16_000, decode=False))
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


def _cohere_vllm_prompt(config: BenchmarkConfig) -> str:
    punctuation_token = "<|pnc|>" if config.punctuation else "<|nopnc|>"
    return (
        "<|startofcontext|><|startoftranscript|>"
        f"<|emo:undefined|><|{config.language_code}|><|{config.language_code}|>"
        f"{punctuation_token}<|noitn|><|notimestamp|><|nodiarize|>"
    )


def _cohere_vllm_prompt_token_ids(config: BenchmarkConfig) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    punctuation_token = "<|pnc|>" if config.punctuation else "<|nopnc|>"
    prompt_tokens = [
        "▁",
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        f"<|{config.language_code}|>",
        f"<|{config.language_code}|>",
        punctuation_token,
        "<|noitn|>",
        "<|notimestamp|>",
        "<|nodiarize|>",
    ]
    token_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
    if not isinstance(token_ids, list):
        token_ids = [token_ids]
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and any(token_id == unk_token_id for token_id in token_ids):
        raise ValueError("Failed to resolve Cohere ASR decoder control tokens.")
    return [int(token_id) for token_id in token_ids]


def _granite_prompt(config: BenchmarkConfig) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    chat = [
        {
            "role": "user",
            "content": "<|audio|>can you transcribe the speech into a written format?",
        }
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False)


def _voxtral_input(
    config: BenchmarkConfig,
    audio: tuple[Any, int],
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import RawAudio
    from mistral_common.protocol.transcription.request import TranscriptionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = tokenizer or MistralTokenizer.from_hf_hub(config.model_name)
    audio_array, sample_rate = audio
    mistral_audio = Audio(audio_array, int(sample_rate), format="wav")
    request = TranscriptionRequest(
        model=config.model_name,
        audio=RawAudio.from_audio(mistral_audio),
        language=config.language_code,
        temperature=config.temperature,
    )
    tokenized = tokenizer.encode_transcription(request)
    audios_and_sr = [
        (item.audio_array, int(item.sampling_rate))
        for item in tokenized.audios
    ]
    return {
        "prompt_token_ids": tokenized.tokens,
        "multi_modal_data": {"audio": audios_and_sr},
    }


def _offline_engine_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    family = _model_family(config)
    kwargs: dict[str, Any] = {
        "model": config.model_name,
        "dtype": "bfloat16",
        "max_num_seqs": config.max_num_seqs,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "limit_mm_per_prompt": {"audio": 1},
    }
    if config.max_model_len:
        kwargs["max_model_len"] = config.max_model_len
    if config.enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = config.max_lora_rank
    if family in {"cohere_asr", "generic_transcription"}:
        kwargs["trust_remote_code"] = True
    if family == "voxtral":
        kwargs.update(
            {
                "tokenizer_mode": "mistral",
                "config_format": "mistral",
                "load_format": "mistral",
                "enforce_eager": True,
                "enable_chunked_prefill": False,
            }
        )
    return kwargs


def _build_offline_inputs(
    config: BenchmarkConfig,
    audios: list[tuple[Any, int]],
) -> list[dict[str, Any]]:
    family = _model_family(config)
    if family == "cohere_asr":
        prompt_token_ids = _cohere_vllm_prompt_token_ids(config)
        return [
            {
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": {"audio": [audio]},
            }
            for audio in audios
        ]
    if family == "granite_speech":
        prompt = _granite_prompt(config)
        return [
            {
                "prompt": prompt,
                "multi_modal_data": {"audio": audio},
            }
            for audio in audios
        ]
    if family == "voxtral":
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tokenizer = MistralTokenizer.from_hf_hub(config.model_name)
        return [_voxtral_input(config, audio, tokenizer=tokenizer) for audio in audios]
    raise ValueError(f"Unsupported offline model_family={family!r}")


def _start_log_thread(process: subprocess.Popen[str], *, prefix: str) -> threading.Thread:
    def drain() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{prefix}] {line.rstrip()}", flush=True)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread


def _start_vllm_server(config: BenchmarkConfig) -> tuple[subprocess.Popen[str], float]:
    env = os.environ.copy()
    token = _get_hf_token()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token

    family = _model_family(config)
    if family == "voxtral_realtime":
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    command = [
        "vllm",
        "serve",
        config.model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--api-key",
        config.api_key,
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]
    if family in {"cohere_asr", "generic_transcription"}:
        command.append("--trust-remote-code")
    if config.max_model_len:
        command.extend(["--max-model-len", str(config.max_model_len)])
    if family == "voxtral":
        command.extend(
            [
                "--tokenizer-mode",
                "mistral",
                "--config-format",
                "mistral",
                "--load-format",
                "mistral",
            ]
        )
    if family == "voxtral_realtime":
        command.extend(
            [
                "--tokenizer-mode",
                "mistral",
                "--compilation-config",
                '{"cudagraph_mode":"PIECEWISE"}',
            ]
        )
    print("[vllm] starting: " + " ".join(command), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    _start_log_thread(process, prefix="vllm")

    import requests

    health_url = f"http://127.0.0.1:{config.port}/health"
    deadline = started + config.startup_timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}")
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                ready_seconds = time.monotonic() - started
                print(f"[vllm] server ready in {ready_seconds:.2f}s", flush=True)
                return process, ready_seconds
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server was not ready after {config.startup_timeout_seconds}s")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass


async def _transcribe_one(
    *,
    client: Any,
    config: BenchmarkConfig,
    audio_path: Path,
    index: int,
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{config.port}/v1/audio/transcriptions"
    started = time.monotonic()
    audio_bytes = audio_path.read_bytes()
    data = {
        "model": config.model_name,
        "language": config.language_code,
        "response_format": "json",
        "temperature": str(config.temperature),
    }
    if _model_family(config) == "cohere_asr":
        data["prompt"] = _cohere_vllm_prompt(config)
    response = await client.post(
        url,
        headers={"Authorization": f"Bearer {config.api_key}"},
        data=data,
        files={"file": (audio_path.name, audio_bytes, "audio/wav")},
    )
    ended = time.monotonic()
    if response.status_code >= 400:
        raise RuntimeError(
            f"vLLM request index={index} failed status={response.status_code}: {response.text[:1000]}"
        )
    try:
        payload = response.json()
    except Exception:
        payload = {"text": response.text}
    text = payload.get("text") if isinstance(payload, dict) else str(payload)
    return {
        "index": index,
        "prediction": str(text or ""),
        "request_seconds": ended - started,
        "request_started": started,
        "request_ended": ended,
        "response": _json_safe(payload),
    }


async def _transcribe_all(
    *,
    config: BenchmarkConfig,
    audio_paths: list[Path],
    audio_seconds: list[float],
    progress_path: Path,
    run_started: float,
) -> list[dict[str, Any]]:
    import httpx

    timeout = httpx.Timeout(config.request_timeout_seconds, connect=20.0)
    limits = httpx.Limits(max_connections=max(1, config.concurrency), max_keepalive_connections=max(1, config.concurrency))
    semaphore = asyncio.Semaphore(max(1, config.concurrency))
    results: list[dict[str, Any] | None] = [None] * len(audio_paths)
    completed = 0
    request_seconds_done = 0.0
    audio_seconds_done = 0.0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def run_one(index: int, path: Path) -> dict[str, Any]:
            async with semaphore:
                return await _transcribe_one(
                    client=client,
                    config=config,
                    audio_path=path,
                    index=index,
                )

        tasks = [asyncio.create_task(run_one(index, path)) for index, path in enumerate(audio_paths)]
        for task in asyncio.as_completed(tasks):
            result = await task
            index = int(result["index"])
            results[index] = result
            completed += 1
            request_seconds_done += float(result["request_seconds"])
            audio_seconds_done += float(audio_seconds[index])
            if (
                completed == len(audio_paths)
                or completed % max(1, config.progress_log_interval_samples) == 0
            ):
                elapsed = time.monotonic() - run_started
                eta_seconds = None
                if completed < len(audio_paths) and completed:
                    eta_seconds = (elapsed / completed) * (len(audio_paths) - completed)
                progress = {
                    "stage": "transcribing",
                    "status": "running" if completed < len(audio_paths) else "complete",
                    "updated_at_utc": _now_iso(),
                    "model": config.model_name,
                    "samples_done": completed,
                    "samples_total": len(audio_paths),
                    "percent_complete": completed / len(audio_paths) if audio_paths else 1.0,
                    "audio_seconds_done": audio_seconds_done,
                    "elapsed_seconds": elapsed,
                    "request_seconds_done": request_seconds_done,
                    "current_throughput_rtfx": (audio_seconds_done / elapsed) if elapsed > 0 else None,
                    "eta_seconds": eta_seconds,
                    "concurrency": config.concurrency,
                }
                _write_json(progress_path, progress)
                await artifacts_volume.commit.aio()
                print(
                    f"[{_model_label(config)}-vllm] samples {completed}/{len(audio_paths)} "
                    f"rtfx={progress['current_throughput_rtfx']:.2f} eta={eta_seconds}",
                    flush=True,
                )

    return [result for result in results if result is not None]


def _pcm16_chunks(audio: tuple[Any, int], *, chunk_size: int = 4096) -> list[str]:
    import numpy as np

    array, sample_rate = _resample_audio(audio[0], int(audio[1]), target_rate=16_000)
    clipped = np.clip(_normalize_audio_array(array), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2").tobytes()
    return [
        base64.b64encode(pcm16[index : index + chunk_size]).decode("ascii")
        for index in range(0, len(pcm16), chunk_size)
    ]


async def _transcribe_realtime_one(
    *,
    config: BenchmarkConfig,
    audio: tuple[Any, int],
    index: int,
) -> dict[str, Any]:
    import websockets

    url = f"ws://127.0.0.1:{config.port}/v1/realtime"
    chunks = _pcm16_chunks(audio)
    started = time.monotonic()
    text_parts: list[str] = []
    response_payload: dict[str, Any] = {}
    headers = {"Authorization": f"Bearer {config.api_key}"}
    async with websockets.connect(
        url,
        additional_headers=headers,
        max_size=None,
        ping_timeout=config.request_timeout_seconds,
    ) as ws:
        first_event = json.loads(await ws.recv())
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "model": config.model_name,
                    "temperature": config.temperature,
                }
            )
        )
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
        for chunk in chunks:
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": chunk,
                    }
                )
            )
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        while True:
            response = json.loads(
                await asyncio.wait_for(ws.recv(), timeout=config.request_timeout_seconds)
            )
            response_type = response.get("type")
            if response_type == "transcription.delta":
                text_parts.append(str(response.get("delta") or ""))
            elif response_type == "transcription.done":
                response_payload = response
                break
            elif response_type == "error":
                raise RuntimeError(f"vLLM realtime request index={index} failed: {response}")
    ended = time.monotonic()
    return {
        "index": index,
        "prediction": str(response_payload.get("text") or "".join(text_parts)),
        "request_seconds": ended - started,
        "request_started": started,
        "request_ended": ended,
        "response": _json_safe({"session_created": first_event, "final": response_payload}),
    }


async def _transcribe_realtime_all(
    *,
    config: BenchmarkConfig,
    audios: list[tuple[Any, int]],
    audio_seconds: list[float],
    progress_path: Path,
    run_started: float,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, config.concurrency))
    results: list[dict[str, Any] | None] = [None] * len(audios)
    completed = 0
    request_seconds_done = 0.0
    audio_seconds_done = 0.0

    async def run_one(index: int, audio: tuple[Any, int]) -> dict[str, Any]:
        async with semaphore:
            return await _transcribe_realtime_one(
                config=config,
                audio=audio,
                index=index,
            )

    tasks = [asyncio.create_task(run_one(index, audio)) for index, audio in enumerate(audios)]
    for task in asyncio.as_completed(tasks):
        result = await task
        index = int(result["index"])
        results[index] = result
        completed += 1
        request_seconds_done += float(result["request_seconds"])
        audio_seconds_done += float(audio_seconds[index])
        if (
            completed == len(audios)
            or completed % max(1, config.progress_log_interval_samples) == 0
        ):
            elapsed = time.monotonic() - run_started
            eta_seconds = None
            if completed < len(audios) and completed:
                eta_seconds = (elapsed / completed) * (len(audios) - completed)
            progress = {
                "stage": "realtime_transcribing",
                "status": "running" if completed < len(audios) else "complete",
                "updated_at_utc": _now_iso(),
                "model": config.model_name,
                "samples_done": completed,
                "samples_total": len(audios),
                "percent_complete": completed / len(audios) if audios else 1.0,
                "audio_seconds_done": audio_seconds_done,
                "audio_seconds_total": sum(audio_seconds),
                "elapsed_seconds": elapsed,
                "request_seconds_done": request_seconds_done,
                "current_throughput_rtfx": (audio_seconds_done / elapsed) if elapsed > 0 else None,
                "eta_seconds": eta_seconds,
                "concurrency": config.concurrency,
            }
            _write_json(progress_path, progress)
            await artifacts_volume.commit.aio()
            print(
                f"[{_model_label(config)}-vllm-realtime] samples {completed}/{len(audios)} "
                f"rtfx={progress['current_throughput_rtfx']:.2f} eta={eta_seconds}",
                flush=True,
            )

    return [result for result in results if result is not None]


def _stage_audio_files(
    *,
    dataset,
    audio_column: str,
    text_column: str,
    temp_dir: Path,
) -> tuple[list[Path], list[str], list[float]]:
    import soundfile as sf

    audio_paths: list[Path] = []
    references: list[str] = []
    audio_seconds: list[float] = []
    for index, row in enumerate(dataset):
        array, sample_rate = _audio_to_array_and_rate(row[audio_column])
        audio_path = temp_dir / f"sample-{index:06d}.wav"
        sf.write(audio_path, array, sample_rate)
        audio_paths.append(audio_path)
        references.append(str(row[text_column]))
        audio_seconds.append(float(len(array)) / float(sample_rate) if sample_rate else 0.0)
    return audio_paths, references, audio_seconds


def _collect_audio_arrays(
    *,
    dataset,
    audio_column: str,
    text_column: str,
) -> tuple[list[tuple[Any, int]], list[str], list[float]]:
    audios: list[tuple[Any, int]] = []
    references: list[str] = []
    audio_seconds: list[float] = []
    for row in dataset:
        array, sample_rate = _audio_to_array_and_rate(row[audio_column])
        audios.append((array, sample_rate))
        references.append(str(row[text_column]))
        audio_seconds.append(float(len(array)) / float(sample_rate) if sample_rate else 0.0)
    return audios, references, audio_seconds


def _load_compare_report(run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    path = _run_dir(run_id) / "report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _benchmark_offline_impl(
    *,
    config: BenchmarkConfig,
    run_id: str,
    run_dir: Path,
    progress_path: Path,
    dataset,
    audio_column: str,
    text_column: str,
    metadata: list[dict[str, Any]],
    dataset_load_seconds: float,
    run_started: float,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    _write_json(
        progress_path,
        {
            "stage": "collecting_audio",
            "status": "running",
            "updated_at_utc": _now_iso(),
            "run_id": run_id,
            "samples_total": len(dataset),
        },
    )
    artifacts_volume.commit()
    audio_stage_started = time.monotonic()
    audios, references, audio_seconds = _collect_audio_arrays(
        dataset=dataset,
        audio_column=audio_column,
        text_column=text_column,
    )
    audio_stage_seconds = time.monotonic() - audio_stage_started

    _write_json(
        progress_path,
        {
            "stage": "loading_vllm_offline",
            "status": "running",
            "updated_at_utc": _now_iso(),
            "run_id": run_id,
            "samples_total": len(audios),
            "audio_seconds_total": sum(audio_seconds),
            "max_num_seqs": config.max_num_seqs,
        },
    )
    artifacts_volume.commit()
    model_load_started = time.monotonic()
    llm = LLM(**_offline_engine_kwargs(config))
    model_load_seconds = time.monotonic() - model_load_started

    sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_tokens)
    batch_size = max(1, int(config.offline_batch_size or len(audios)))

    _write_json(
        progress_path,
        {
            "stage": "generating",
            "status": "running",
            "updated_at_utc": _now_iso(),
            "run_id": run_id,
            "samples_total": len(audios),
            "audio_seconds_total": sum(audio_seconds),
            "model_load_seconds": model_load_seconds,
            "offline_batch_size": batch_size,
        },
    )
    artifacts_volume.commit()
    generate_started = time.monotonic()
    predictions: list[str] = []
    for batch_start in range(0, len(audios), batch_size):
        batch_audios = audios[batch_start : batch_start + batch_size]
        inputs = _build_offline_inputs(config, batch_audios)
        if config.enable_lora:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("speech", 1, config.model_name)
            outputs = llm.generate(
                inputs,
                sampling_params=sampling_params,
                lora_request=[lora_request for _ in inputs],
            )
        else:
            outputs = llm.generate(inputs, sampling_params=sampling_params)
        predictions.extend(output.outputs[0].text if output.outputs else "" for output in outputs)
        completed = min(len(predictions), len(audios))
        elapsed = time.monotonic() - generate_started
        audio_seconds_done = sum(audio_seconds[:completed])
        eta_seconds = None
        if completed < len(audios) and completed and elapsed > 0:
            eta_seconds = (elapsed / completed) * (len(audios) - completed)
        progress = {
            "stage": "generating",
            "status": "running" if completed < len(audios) else "complete",
            "updated_at_utc": _now_iso(),
            "run_id": run_id,
            "samples_done": completed,
            "samples_total": len(audios),
            "percent_complete": completed / len(audios) if audios else 1.0,
            "audio_seconds_done": audio_seconds_done,
            "audio_seconds_total": sum(audio_seconds),
            "elapsed_seconds": elapsed,
            "current_throughput_rtfx": (audio_seconds_done / elapsed) if elapsed > 0 else None,
            "eta_seconds": eta_seconds,
            "offline_batch_size": batch_size,
        }
        _write_json(progress_path, progress)
        artifacts_volume.commit()
        print(
            f"[{_model_label(config)}-vllm-offline] samples {completed}/{len(audios)} "
            f"rtfx={progress['current_throughput_rtfx']:.2f} eta={eta_seconds}",
            flush=True,
        )
    generate_seconds = time.monotonic() - generate_started

    return _finalize_report(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        progress_path=progress_path,
        references=references,
        predictions=predictions,
        metadata=metadata,
        audio_seconds=audio_seconds,
        dataset_info={
            "name": config.eval_dataset,
            "config": config.eval_config_name,
            "split": config.eval_split,
            "audio_column": audio_column,
            "text_column": text_column,
            "samples": len(references),
            "audio_seconds_total": sum(audio_seconds),
        },
        timing={
            "dataset_load_seconds": dataset_load_seconds,
            "model_load_seconds": model_load_seconds,
            "audio_stage_seconds": audio_stage_seconds,
            "request_wall_seconds": generate_seconds,
            "request_seconds_sum": generate_seconds,
            "end_to_end_seconds": time.monotonic() - run_started,
            "throughput_rtfx": (sum(audio_seconds) / generate_seconds)
            if generate_seconds > 0
            else None,
            "end_to_end_rtfx": (sum(audio_seconds) / (time.monotonic() - run_started))
            if time.monotonic() > run_started
            else None,
            "request_latency_seconds": _summarize_numeric([]),
            "audio_seconds": _summarize_numeric(audio_seconds),
        },
    )


def _finalize_report(
    *,
    config: BenchmarkConfig,
    run_id: str,
    run_dir: Path,
    progress_path: Path,
    references: list[str],
    predictions: list[str],
    metadata: list[dict[str, Any]],
    audio_seconds: list[float],
    dataset_info: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    label = _model_label(config)
    normalizer = _text_normalizer()
    normalized_references = [normalizer(reference) for reference in references]
    normalized_predictions = [normalizer(prediction) for prediction in predictions]
    metrics = _compute_text_metrics(normalized_references, normalized_predictions)

    from jiwer import cer, wer

    pairwise_path = run_dir / "pairwise_predictions.jsonl"
    if pairwise_path.exists():
        pairwise_path.unlink()
    rows: list[dict[str, Any]] = []
    for index, reference in enumerate(normalized_references):
        row = {
            "index": index,
            "reference": references[index],
            "normalized_reference": reference,
            f"{label}__prediction": predictions[index],
            f"{label}__normalized_prediction": normalized_predictions[index],
            f"{label}__wer": wer(reference, normalized_predictions[index]) if reference else None,
            f"{label}__cer": cer(reference, normalized_predictions[index]) if reference else None,
            "audio_seconds": audio_seconds[index],
            "metadata": metadata[index] if index < len(metadata) else {},
        }
        rows.append(row)
        if len(rows) >= 500:
            _append_jsonl(pairwise_path, rows)
            rows = []
    if rows:
        _append_jsonl(pairwise_path, rows)

    compare_report = _load_compare_report(config.compare_report_run_id)
    report = {
        "run_id": run_id,
        "created_at_utc": _now_iso(),
        "config": asdict(config),
        "dataset": dataset_info,
        "model": {
            "label": label,
            "model_name": config.model_name,
            "backend": f"vllm_{config.vllm_mode}",
            "family": _model_family(config),
        },
        "metrics": metrics,
        "timing": timing,
        "compare_report": compare_report,
        "artifacts": {
            "run_dir": str(run_dir),
            "report_path": str(run_dir / "report.json"),
            "progress_path": str(progress_path),
            "pairwise_predictions_jsonl": str(pairwise_path),
        },
    }
    _write_json(run_dir / "report.json", report)
    _write_json(
        progress_path,
        {
            "stage": "complete",
            "status": "complete",
            "updated_at_utc": _now_iso(),
            "run_id": run_id,
            "report_path": str(run_dir / "report.json"),
            "samples_total": len(references),
            "metrics": metrics,
            "throughput_rtfx": report["timing"]["throughput_rtfx"],
        },
    )
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return report


def _benchmark_impl(config: BenchmarkConfig) -> dict[str, Any]:
    token = _get_hf_token()
    run_id = f"{_sanitize_artifact_component(config.benchmark_name)}-{_now_utc()}"
    run_dir = _run_dir(run_id)
    progress_path = run_dir / "progress.json"
    run_started = time.monotonic()
    _write_json(
        progress_path,
        {
            "stage": "starting",
            "status": "running",
            "updated_at_utc": _now_iso(),
            "run_id": run_id,
            "config": asdict(config),
        },
    )
    artifacts_volume.commit()

    dataset_load_started = time.monotonic()
    dataset, audio_column, text_column = _load_dataset_split(config, token=token)
    metadata = _metadata_rows(dataset, audio_column=audio_column)
    dataset_load_seconds = time.monotonic() - dataset_load_started

    if config.vllm_mode == "offline":
        return _benchmark_offline_impl(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            progress_path=progress_path,
            dataset=dataset,
            audio_column=audio_column,
            text_column=text_column,
            metadata=metadata,
            dataset_load_seconds=dataset_load_seconds,
            run_started=run_started,
        )
    if config.vllm_mode not in {"server", "realtime"}:
        raise ValueError("vllm_mode must be one of: offline, server, realtime")

    process: subprocess.Popen[str] | None = None
    try:
        process, server_ready_seconds = _start_vllm_server(config)
        if config.vllm_mode == "realtime":
            audio_stage_started = time.monotonic()
            audios, references, audio_seconds = _collect_audio_arrays(
                dataset=dataset,
                audio_column=audio_column,
                text_column=text_column,
            )
            audio_stage_seconds = time.monotonic() - audio_stage_started
            _write_json(
                progress_path,
                {
                    "stage": "realtime_transcribing",
                    "status": "running",
                    "updated_at_utc": _now_iso(),
                    "run_id": run_id,
                    "samples_total": len(audios),
                    "audio_seconds_total": sum(audio_seconds),
                    "server_ready_seconds": server_ready_seconds,
                    "dataset_load_seconds": dataset_load_seconds,
                    "audio_stage_seconds": audio_stage_seconds,
                    "concurrency": config.concurrency,
                },
            )
            artifacts_volume.commit()
            transcribe_started = time.monotonic()
            results = asyncio.run(
                _transcribe_realtime_all(
                    config=config,
                    audios=audios,
                    audio_seconds=audio_seconds,
                    progress_path=progress_path,
                    run_started=transcribe_started,
                )
            )
        else:
            temp_context = tempfile.TemporaryDirectory()
            temp = temp_context.__enter__()
            audio_stage_started = time.monotonic()
            audio_paths, references, audio_seconds = _stage_audio_files(
                dataset=dataset,
                audio_column=audio_column,
                text_column=text_column,
                temp_dir=Path(temp),
            )
            audio_stage_seconds = time.monotonic() - audio_stage_started
            _write_json(
                progress_path,
                {
                    "stage": "transcribing",
                    "status": "running",
                    "updated_at_utc": _now_iso(),
                    "run_id": run_id,
                    "samples_total": len(audio_paths),
                    "audio_seconds_total": sum(audio_seconds),
                    "server_ready_seconds": server_ready_seconds,
                    "dataset_load_seconds": dataset_load_seconds,
                    "audio_stage_seconds": audio_stage_seconds,
                    "concurrency": config.concurrency,
                },
            )
            artifacts_volume.commit()
            transcribe_started = time.monotonic()
            results = asyncio.run(
                _transcribe_all(
                    config=config,
                    audio_paths=audio_paths,
                    audio_seconds=audio_seconds,
                    progress_path=progress_path,
                    run_started=transcribe_started,
                )
            )
            temp_context.__exit__(None, None, None)
    finally:
        if process is not None:
            _stop_process(process)

    predictions = [str(result["prediction"]) for result in results]
    request_started_values = [float(result["request_started"]) for result in results]
    request_ended_values = [float(result["request_ended"]) for result in results]
    request_seconds_values = [float(result["request_seconds"]) for result in results]
    request_wall_seconds = (
        max(request_ended_values) - min(request_started_values)
        if request_started_values and request_ended_values
        else 0.0
    )
    audio_seconds_total = sum(audio_seconds)

    return _finalize_report(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        progress_path=progress_path,
        references=references,
        predictions=predictions,
        metadata=metadata,
        audio_seconds=audio_seconds,
        dataset_info={
            "name": config.eval_dataset,
            "config": config.eval_config_name,
            "split": config.eval_split,
            "audio_column": audio_column,
            "text_column": text_column,
            "samples": len(references),
            "audio_seconds_total": audio_seconds_total,
        },
        timing={
            "dataset_load_seconds": dataset_load_seconds,
            "server_ready_seconds": server_ready_seconds,
            "audio_stage_seconds": audio_stage_seconds,
            "request_wall_seconds": request_wall_seconds,
            "request_seconds_sum": sum(request_seconds_values),
            "end_to_end_seconds": time.monotonic() - run_started,
            "throughput_rtfx": (audio_seconds_total / request_wall_seconds)
            if request_wall_seconds > 0
            else None,
            "end_to_end_rtfx": (audio_seconds_total / (time.monotonic() - run_started))
            if time.monotonic() > run_started
            else None,
            "request_latency_seconds": _summarize_numeric(request_seconds_values),
            "audio_seconds": _summarize_numeric(audio_seconds),
        },
    )


@app.function(
    image=vllm_image,
    gpu=GPU,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        str(ARTIFACTS_DIR): artifacts_volume,
        str(HF_CACHE_DIR): hf_cache_volume,
    },
)
def cohere_vllm_benchmark_remote(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["MODAL_GPU_LABEL"] = GPU
    config = BenchmarkConfig(**payload)
    return _benchmark_impl(config)


@app.local_entrypoint()
def main(
    benchmark_name: str = "cohere-vllm-benchmark",
    model_name: str = "CohereLabs/cohere-transcribe-03-2026",
    model_label: str = "",
    model_family: str = "",
    eval_dataset: str = "ai4bharat/Svarah",
    eval_config_name: str = "",
    eval_split: str = "test",
    eval_audio_column: str = "",
    eval_text_column: str = "",
    eval_trust_remote_code: bool = False,
    max_samples: int = 64,
    language_code: str = "en",
    punctuation: bool = True,
    concurrency: int = 16,
    request_timeout_seconds: float = 300.0,
    port: int = 8000,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.90,
    startup_timeout_seconds: float = 900.0,
    progress_log_interval_samples: int = 8,
    compare_report_run_id: str = "",
    vllm_mode: str = "offline",
    max_tokens: int = 256,
    offline_batch_size: int = 512,
    max_model_len: int = 0,
    temperature: float = 0.0,
    enable_lora: bool = False,
    max_lora_rank: int = 64,
) -> None:
    config = BenchmarkConfig(
        benchmark_name=benchmark_name,
        model_name=model_name,
        model_label=model_label or None,
        model_family=model_family or None,
        eval_dataset=eval_dataset,
        eval_config_name=eval_config_name or None,
        eval_split=eval_split,
        eval_audio_column=eval_audio_column or None,
        eval_text_column=eval_text_column or None,
        eval_trust_remote_code=eval_trust_remote_code,
        max_samples=max_samples or None,
        language_code=language_code,
        punctuation=punctuation,
        concurrency=concurrency,
        request_timeout_seconds=request_timeout_seconds,
        port=port,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        startup_timeout_seconds=startup_timeout_seconds,
        progress_log_interval_samples=progress_log_interval_samples,
        compare_report_run_id=compare_report_run_id or None,
        vllm_mode=vllm_mode,
        max_tokens=max_tokens,
        offline_batch_size=offline_batch_size,
        max_model_len=max_model_len or None,
        temperature=temperature,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
    )
    report = cohere_vllm_benchmark_remote.remote(asdict(config))
    print(json.dumps(report, indent=2, sort_keys=True))
