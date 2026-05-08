# ASR Model Benchmarks

This document records the speech-to-text model bakeoffs for LocalWispr/Wispr Flow-style dictation. The goal is practical product selection: high Indian-accent English accuracy, low stop-to-final latency, and predictable deployment behavior.

## Evaluation Target

- Primary benchmark: `ai4bharat/Svarah` test split.
- Primary metric: normalized WER/CER after Whisper basic English normalization.
- Product use case: short-to-medium dictation bursts, usually after-stop final transcript, with possible future streaming/partial transcript support.
- Current default recommendation: `openai/whisper-large-v3-turbo`.

## Important Methodology Note

Some accuracy runs were sharded across multiple GPUs to finish faster. That is valid for WER/CER, but it is not a deployment speed number.

For deployment speed, use:

- single request latency on the target serving stack;
- single-GPU equivalent RTFx;
- first-token/first-partial latency for streaming APIs;
- final transcript latency from audio stop to usable text.

Parallel RTFx is only useful for eval throughput.

## Current Svarah Results

| Model | Backend | Svarah WER | Svarah CER | Speed Notes |
|---|---|---:|---:|---|
| `openai/whisper-large-v3` | prior local benchmark | `0.0711` | `0.0344` | Best accuracy measured so far; likely slower than turbo. |
| `openai/whisper-large-v3-turbo` | prior local benchmark | `0.0821` | `0.0390` | Best current accuracy/speed tradeoff. |
| best non-Svarah Whisper LoRA | HF/PEFT | `0.0815` | `0.0386` | Small gain over turbo, not a decisive product win. |
| `ibm-granite/granite-speech-4.1-2b` | vLLM offline, 5 eval shards | `0.09048` | `0.04280` | Best non-Whisper base model in this run. Single-GPU equivalent inference RTFx was about `23.4` on H100-class vLLM shards. |
| `CohereLabs/cohere-transcribe-03-2026` | vLLM | `0.0933` | `0.0497` | Did not beat Granite or Whisper. |
| `nvidia/parakeet-unified-en-0.6b` | NeMo git backend, 5 eval shards | `0.09507` | `0.05125` | Fast and usable, but accuracy is behind Whisper and Granite AR. Single-GPU equivalent RTFx about `263.4`. |
| `ibm-granite/granite-speech-4.1-2b-nar` | Transformers + FlashAttention 2, 5 eval shards | `0.09718` | `0.05129` | Valid NAR stack now works. Single-GPU equivalent inference RTFx about `243.1`; accuracy behind Granite AR. |
| `nvidia/canary-qwen-2.5b` | prior bakeoff | `0.1054` | `0.0564` | Not competitive. |
| `nvidia/parakeet-tdt-0.6b-v2` | prior bakeoff | `0.1306` | `0.0742` | Not competitive on Svarah. |
| `nvidia/parakeet-tdt-0.6b-v3` | NeMo backend, 5 eval shards | `0.16676` | `0.10237` | Very fast, but accuracy is unacceptable for Indian-accent dictation. Single-GPU equivalent RTFx about `430.2`. |
| `Qwen/Qwen3-ASR-1.7B` | prior bakeoff | `0.1795` | `0.1054` | Not competitive. |

## May 8, 2026 Bakeoff Notes

Tested four new candidates:

- `ibm-granite/granite-speech-4.1-2b`
- `ibm-granite/granite-speech-4.1-2b-nar`
- `nvidia/parakeet-tdt-0.6b-v3`
- `nvidia/parakeet-unified-en-0.6b`

Outcome:

- Granite 4.1 2B AR was the best new accuracy candidate.
- Granite NAR required a valid FlashAttention 2 setup; SDPA fails by design because the model asserts `flash_attention_2`.
- Parakeet unified is the best fast non-Whisper candidate, but it does not beat Whisper turbo.
- Parakeet TDT v3 should be rejected for this product unless a future Indian-accent-specific checkpoint changes the result.

Local downloaded reports are under:

- `artifacts/benchmarks/granite_speech_4p1_2b_svarah_shards/`
- `artifacts/benchmarks/granite_speech_4p1_2b_nar_svarah_shards/`

Remote Modal artifacts are in the `localwispr-whisper-lora-artifacts` volume.

## Granite NAR Stack Fix

The first NAR attempts failed for three separate reasons:

1. vLLM could not load the NAR model config.
2. Transformers SDPA failed because the model requires `flash_attention_2`.
3. Building FlashAttention from source on Python 3.11 was too brittle.

The working stack:

- Python `3.12`
- Torch `2.9.1+cu128`
- `transformers==4.57.6`
- `ibm-granite/granite-speech-4.1-2b-nar`
- official `flash-attn==2.8.3+cu12torch2.9` wheel from the FlashAttention v2.8.3 release
- direct audio decoding with `Audio(decode=False)` plus soundfile/librosa, avoiding the torchcodec mismatch

Implementation lives in `tools/modal_granite_nar_benchmark.py`.

## Direct Base Model Recommendation

For LocalWispr dictation without LoRA:

1. Use `openai/whisper-large-v3-turbo` as the default.
2. Offer `openai/whisper-large-v3` as an accuracy-first mode if latency is acceptable.
3. Keep Granite 4.1 2B as the strongest non-Whisper fallback candidate.
4. Do not pick Parakeet v3 or Qwen3 ASR for Indian-accent English dictation right now.

## API Provider Benchmark Design

The API-provider question is different from the Modal offline model bakeoff. For a Wispr Flow-style product, accuracy alone is not enough. We need to measure the whole stop-to-usable-text path.

## Local Benchmark Corpus UI

LocalWispr now has a **Benchmarks** section in the macOS control panel for recording a reusable API test corpus.

Implementation:

- UI: `Sources/LocalWispr/BenchmarkRecorderView.swift`
- storage: `Sources/LocalWispr/BenchmarkCorpusStore.swift`
- app wiring: `Sources/LocalWispr/AppState.swift`
- sidebar routing: `Sources/LocalWispr/ControlPanelView.swift`
- tests: `Tests/LocalWisprTests/BenchmarkCorpusStoreTests.swift`

Workflow:

1. Open LocalWispr.
2. Go to **Benchmarks**.
3. Record a clip.
4. Enter the exact words spoken as the reference text.
5. Choose a category.
6. Save to corpus.

The corpus is written to:

```text
~/Documents/LocalWispr/BenchmarkCorpus
```

Each saved clip gets:

- a copied `.wav` file under `clips/<category>/`
- a same-basename `.txt` reference file
- a `manifest.json` entry with category, reference text, prompt text, duration, input device, and path

The app also includes **Recover Latest**, which points the UI at the newest debug recording under:

```text
/tmp/localwispr-debug-captures
```

This exists because debug/rebuild cycles can lose the in-memory pointer to the last recording even though the WAV file still exists on disk.

### Current Personal Clip

The current user-provided recording is:

```text
/tmp/localwispr-debug-captures/session-20260508-164457/audio.wav
```

Audio properties:

- WAV
- mono
- 16 kHz
- Float32
- duration: about `34.1s`

This is enough for an API smoke test: provider connectivity, final latency, rough transcript quality, streaming behavior, and cost per request. It is not enough for a final provider/model decision or a general Indian-accent robustness claim. Broader accuracy must come from public/gated datasets plus any future personal clips.

### Provider Key State

Current local project state:

- `.env` is gitignored.
- `GROQ_API_KEY` is present locally.
- No `CEREBRAS_API_KEY` was found locally.
- No `FIREWORKS_API_KEY`, `TOGETHER_API_KEY`, or `DEEPGRAM_API_KEY` was found locally.

Do not commit provider keys. The provider replay harness should read keys from environment variables or a local `.env` file loaded at runtime.

### Test Set

Use a fixed recorded corpus first, then add live manual tests.

Recommended corpus:

- 50 short commands: 2-5 seconds.
- 50 normal dictation snippets: 5-15 seconds.
- 30 long paragraphs: 20-45 seconds.
- 30 developer/code snippets with identifiers, package names, symbols, and punctuation.
- 20 noisy/real room clips: fan, keyboard, cafe, laptop mic, AirPods.
- At least 30 Indian-accent clips from known hard cases.

Keep the audio fixed across providers. Live testing alone is too noisy to rank models.

### Metrics

Accuracy:

- WER/CER after the same normalizer.
- Semantic preservation score: did it preserve the intended words and identifiers?
- Named entity / code token accuracy: package names, variables, function names, acronyms.
- Punctuation usability: useful, harmful, or missing.
- Hallucination rate: inserted words not present in speech.

Latency:

- Upload/open time.
- First partial latency for streaming APIs.
- Stable partial latency: when the text becomes unlikely to change.
- Final transcript latency after audio stop.
- End-to-end app latency including cleanup.

Throughput/cost:

- Audio seconds processed per wall second.
- Cost per audio hour.
- Cost per 1,000 dictations.
- Failure/retry rate.

### Test Modes

1. Offline fixed-audio batch: ranks raw model accuracy and final latency fairly.
2. Realtime replay: streams the same audio over WebSocket at 1x speed to test partials and finalization.
3. Live manual dictation: catches UX issues that corpus tests miss, but should not be the primary ranking metric.

### Provider Candidates

Candidates to wire into the harness:

- Groq Whisper variants, if available.
- Groq-hosted Parakeet or other ASR models, if exposed.
- Cerebras ASR options, if exposed.
- Together/Fireworks hosted Whisper/Parakeet-style models.
- OpenAI/Gemini/Deepgram only as external references if product scope allows.

### Product Decision Rule

For Wispr Flow-style dictation, choose the model/API that clears all three bars:

- Svarah-like Indian-accent WER close to or better than Whisper turbo.
- Final transcript latency comfortably below the user annoyance threshold for short utterances.
- Stable partials if using streaming; otherwise do not pay streaming complexity cost.

If an API model is faster but materially less accurate than Whisper turbo, it is not a win for this product.
