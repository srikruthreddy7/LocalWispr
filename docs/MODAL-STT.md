# Modal STT Runbook

This runbook covers deployment and operation of the Modal-hosted Whisper STT service used by LocalWispr.

## Scope

- STT runs on Modal (`tools/modal_stt_service.py`).
- Cleanup LLM remains on the existing cleanup provider path.
- Client UX is after-stop-only: one upload after stop, one final transcript response.

## Deploy Modal Service

```bash
modal deploy tools/modal_stt_service.py
```

For local development:

```bash
modal serve tools/modal_stt_service.py
```

## Required Service Secrets / Env

Set on Modal:

- `LOCALWISPR_MODAL_STT_API_KEY`: bearer token expected by endpoint auth.
- `LOCALWISPR_MODAL_STT_MODEL`: optional override (default `openai/whisper-large-v3-turbo`).
- `LOCALWISPR_MODAL_COMPUTE_TYPE`: optional override (default `float16`).
- `LOCALWISPR_MODAL_APP_NAME`: optional app name override.

## Required LocalWispr Client Env

Set in `.env` or process environment:

```bash
LOCALWISPR_MODAL_STT_ENDPOINT=https://<your-modal-url>/v1/audio/transcriptions
LOCALWISPR_MODAL_STT_API_KEY=<same-bearer-token>
LOCALWISPR_MODAL_STT_MODEL=openai/whisper-large-v3-turbo
LOCALWISPR_MODAL_STT_TIMEOUT_SECONDS=60
```

## Verification

### 1) Single-audio integration check

```bash
LOCALWISPR_TRANSCRIBER_AUDIO=/absolute/path/to/sample.wav \
LOCALWISPR_MODAL_STT_ENDPOINT=https://<endpoint>/v1/audio/transcriptions \
LOCALWISPR_MODAL_STT_API_KEY=<token> \
swift test --filter TranscriberIntegrationTests/testModalSTTTranscribesProvidedAudioFile
```

### 2) Eval set pass (WER + latency shape)

```bash
LOCALWISPR_EVAL_DIR=/absolute/path/to/eval/clips \
LOCALWISPR_MODAL_STT_ENDPOINT=https://<endpoint>/v1/audio/transcriptions \
LOCALWISPR_MODAL_STT_API_KEY=<token> \
swift test --filter ManualEvalTests/testModalSTTOnEvalSet
```

The eval output includes:
- audio duration per clip
- stop-to-transcript timing
- transcript text length
- WER against `_eng.txt` / `_eng.rtf` references

## Debug Metrics

Client logs in `/tmp/localwispr-debug.log` include:
- `audioSeconds`
- `uploadMs`
- `serverDecodeMs` (when service returns `decode_ms`)
- `textLength`

Service response includes:
- `text`
- `segments`
- `decode_ms`
- `metrics.decode_ms`
- `metrics.audio_seconds`

## Rollback Procedure

If you need to revert quickly to Groq STT during experimentation:

1. Restore previous Groq STT wiring in `Sources/LocalWispr/Transcriber.swift`.
2. Set `GROQ_API_KEY` and any previous STT model settings.
3. Remove `LOCALWISPR_MODAL_STT_*` variables from runtime environment.
4. Re-run `TranscriberIntegrationTests` to validate before team use.
