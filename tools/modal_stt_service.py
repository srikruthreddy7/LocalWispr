"""Modal STT service for LocalWispr (OpenAI-compatible endpoint)."""

import io
import os
import time
from dataclasses import dataclass
from typing import Any

import modal

MODEL_NAME = os.environ.get("LOCALWISPR_MODAL_STT_MODEL", "openai/whisper-large-v3-turbo")
APP_NAME = os.environ.get("LOCALWISPR_MODAL_APP_NAME", "localwispr-modal-stt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi==0.116.1",
        "python-multipart==0.0.20",
        "torch==2.5.1",
        "transformers==4.47.1",
        "accelerate==1.2.1",
        "librosa==0.10.2",
        "soundfile==0.12.1",
    )
)

app = modal.App(APP_NAME, image=image)

_model = None


@dataclass
class TranscriptionResult:
    text: str
    segments: list[dict[str, Any]]
    decode_ms: int
    audio_prepare_ms: int
    audio_seconds: float | None


def _get_model():
    global _model
    if _model is None:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda",
        )
    return _model


def _transcribe_audio(
    *,
    audio_bytes: bytes,
    language: str | None,
    prompt: str | None,
    temperature: float,
) -> TranscriptionResult:
    import librosa
    import soundfile as sf

    pipe = _get_model()
    audio_prepare_started = time.perf_counter()
    audio, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = audio.mean(axis=1)

    target_sampling_rate = int(pipe.feature_extractor.sampling_rate)
    if int(sampling_rate) != target_sampling_rate:
        audio = librosa.resample(audio, orig_sr=int(sampling_rate), target_sr=target_sampling_rate)
        sampling_rate = target_sampling_rate

    audio_prepare_ms = int((time.perf_counter() - audio_prepare_started) * 1000)

    started = time.perf_counter()
    generate_kwargs = {
        "task": "transcribe",
        "language": language or "en",
        "temperature": temperature,
    }
    if prompt:
        # HF Whisper's generation path for this pipeline does not accept a raw
        # `prompt` kwarg; passing it causes a runtime ValueError and a 500.
        # We ignore client prompts here rather than fail the request.
        pass
    output = pipe(
        {"array": audio, "sampling_rate": int(sampling_rate)},
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
    )
    decode_ms = int((time.perf_counter() - started) * 1000)
    chunks = output.get("chunks", [])
    text = str(output.get("text", "")).strip()

    segments = []
    for chunk in chunks:
        ts = chunk.get("timestamp") if isinstance(chunk, dict) else None
        if not ts or len(ts) != 2:
            continue
        chunk_text = str(chunk.get("text", "")).strip()
        if not chunk_text:
            continue
        start, end = ts
        if start is None:
            start = 0.0
        if end is None:
            end = float(start)
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "text": chunk_text,
            }
        )

    audio_seconds = (len(audio) / float(sampling_rate)) if len(audio) else None
    return TranscriptionResult(
        text=text,
        segments=segments,
        decode_ms=decode_ms,
        audio_prepare_ms=audio_prepare_ms,
        audio_seconds=audio_seconds,
    )


@app.function(
    gpu="L40S",
    max_containers=1,
    scaledown_window=3_600,
    secrets=[modal.Secret.from_name("localwispr-modal-stt-secret")],
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    web_app = FastAPI(title="LocalWispr Modal STT")
    required_api_key = os.environ.get("LOCALWISPR_MODAL_STT_API_KEY", "").strip()

    @web_app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form(MODEL_NAME),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        temperature: float = Form(default=0.0),
        authorization: str | None = Header(default=None),
    ):
        if required_api_key:
            bearer = f"Bearer {required_api_key}"
            if authorization != bearer:
                raise HTTPException(status_code=401, detail="Unauthorized")

        if model != MODEL_NAME:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model '{model}'. Expected '{MODEL_NAME}'.",
            )

        request_read_started = time.perf_counter()
        audio_bytes = await file.read()
        request_read_ms = int((time.perf_counter() - request_read_started) * 1000)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty.")

        result = _transcribe_audio(
            audio_bytes=audio_bytes,
            language=language,
            prompt=prompt,
            temperature=temperature,
        )

        if not result.text:
            raise HTTPException(status_code=422, detail="Transcription returned empty text.")

        total_ms = request_read_ms + result.audio_prepare_ms + result.decode_ms
        payload = {
            "text": result.text,
            "segments": result.segments,
            "decode_ms": result.decode_ms,
            "model": MODEL_NAME,
            "metrics": {
                "request_read_ms": request_read_ms,
                "audio_prepare_ms": result.audio_prepare_ms,
                "decode_ms": result.decode_ms,
                "total_ms": total_ms,
                "audio_seconds": result.audio_seconds,
            },
        }
        return JSONResponse(payload)

    return web_app
