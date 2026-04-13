# Modal STT migration — engineering log

This document records **why** we moved speech-to-text to a private Modal deployment, **what** we shipped, **problems** encountered during integration, and **how** we resolved them. It complements the operational runbook in [`MODAL-STT.md`](MODAL-STT.md) (deploy commands, env vars, tests).

**Audience:** future maintainers deciding whether to extend, replace, or roll back this path.

---

## Goals and product decisions

- **Replace Groq STT** with a **self-hosted Whisper** model on Modal: `openai/whisper-large-v3-turbo`, with GPU sizing appropriate for the team (e.g. L40S class).
- **After-stop-only UX:** the client **does not** rely on streaming partial transcripts from the cloud service. Capture stops, then **one** upload runs and **one** final result comes back.
- **No STT fallbacks in the experiment:** simplify failure modes while validating Modal; cloud cleanup (LLM) remains on the existing provider path **separate** from STT.
- **Operational clarity:** one FastAPI-shaped endpoint, explicit env-based configuration on the client, and documentation for deploy, verification, and rollback.

---

## What we implemented

### Modal service (`tools/modal_stt_service.py`)

- HTTP **`POST /v1/audio/transcriptions`** compatible with the client’s multipart upload expectations.
- Loads Whisper via **Hugging Face `transformers`** pipeline on Modal GPU infrastructure.
- Returns JSON including **`text`**, optional **`segments`**, and **`decode_ms`** (and related timing fields) for client metrics and debugging.
- Auth: bearer token checked against Modal-stored secret (`LOCALWISPR_MODAL_STT_API_KEY` pattern as documented in the runbook).

### Swift client (`Sources/LocalWispr/Transcriber.swift`)

- **Buffered session:** on stop, audio is packaged and sent **once** to the Modal endpoint.
- Configuration via **`DotEnv.merged()`** (see `ContextAwareCleaning.swift`): process environment plus `.env` files, with documented **`LOCALWISPR_*`** keys for endpoint, API key, model override, timeout.
- Logging hooks for success/failure, upload latency, and server decode time where available.

### Tests and docs

- Integration and manual eval tests updated to exercise Modal STT behind env-gated filters (see `MODAL-STT.md`).
- **`docs/ARCHITECTURE.md`** and **`README.md`** updated to describe the new STT path at a high level.
- **`docs/MODAL-STT.md`** — deployment, env, verification commands, debug metrics.

### Git branching

- Feature work was consolidated on branch **`modal-initial`** (commit message: *Add Modal-hosted Whisper STT and client wiring* in the primary worktree).  
- **Note:** Git allows only **one** worktree to check out a given branch at a time; creating `modal-initial` in a secondary worktree required temporarily moving the other worktree back to its previous branch before checking out `modal-initial` in the main repo folder.

---

## Problems encountered and resolutions

### 1) Swift concurrency and locking around session state

**Symptom:** Unsafe or awkward use of **`NSLock`** across `async` boundaries risked deadlocks or actor isolation warnings during the transcriber refactor.

**Resolution:** Centralized lock usage in **synchronous** helpers so async entry points do not hold locks across suspension points; kept session lifecycle mutations predictable and main-actor-friendly where required.

### 2) Modal packaging, dependencies, and cold-start behavior

**Symptoms (iterative):** deploy failures, missing runtime dependencies, or containers that scaled down too aggressively for interactive dictation experiments.

**Resolution:** Tightened the Modal app definition: explicit image dependencies, sensible **`scaledown_window`**, and a loading strategy that avoids reloading the full model on every trivial request where possible. (Exact values live in `modal_stt_service.py` and should be treated as tunable ops parameters.)

### 3) Runtime500: `ValueError` — unused `model_kwargs: ['prompt']` (Hugging Face Whisper)

**Symptom:** Modal invocations failed when the client (or tests) sent a **`prompt`** form field. Server logs showed Hugging Face generation rejecting **`prompt`** passed into **`generate_kwargs`** for this pipeline.

**How we diagnosed it:** Opened **[modal.com](https://modal.com)** in the browser, navigated to the deployed **app**, and inspected **logs / trace output** for the failing requests. The stack trace pointed at **`generate_kwargs`** / unused **`prompt`**.

**Resolution:** Stop forwarding **`prompt`** into the Whisper **`generate_kwargs`** path (ignore or strip client prompt for this backend). After redeploy, **`curl`** and integration tests with a `prompt` field returned **HTTP 200** and valid JSON.

**Design note:** If we later need **hinting** (e.g. vocabulary biasing), we should implement an HF-supported mechanism explicitly rather than passing through OpenAI-style `prompt` blindly.

### 4) “Transcription failed” in the app despite Modal STT succeeding

**Symptom:** Debug log showed **Modal STT success**, then the pipeline still reported failure (“cloud speech recognition” / cleanup errors).

**Root cause:** **STT succeeded**, but **post-STT cleanup** uses the **cloud LLM** path (`GROQ_API_KEY` / configured cleanup provider). The worktree `.env` had Modal STT variables but **no** `GROQ_API_KEY`, so cleanup failed with **`provider=nil`** / missing API key — unrelated to Whisper.

**Resolution:** Ensure **both** are set locally when testing the full pipeline:

- Modal STT: `LOCALWISPR_MODAL_STT_*`
- Cleanup LLM: `GROQ_API_KEY` (and optional model / effort env vars as documented in code)

**Lesson:** Split metrics/logging clearly between **STT** and **cleanup** so a single user-facing error string does not conflate the two stages.

### 5) Multiple Git worktrees and divergent `.env` files

**Symptom:** Primary repo checkout had a **minimal** `.env` (e.g. only `GROQ_API_KEY`), while the Cursor worktree had the **full** Modal + Groq set — same Git repo, **different working directories**, so behavior differed by folder.

**What we tried:**

- **Symlink** each worktree’s `.env` to a single canonical file under the primary checkout (one edit updates all).
- Later, a **merge/dedupe** pass on the canonical file so each key appears once (e.g. a single `GROQ_API_KEY` line).

**Constraints:**

- `.env` remains **gitignored**; secrets never enter Git history.
- Each machine still needs a real `.env` (or exported env vars); copying keys is manual by design.

---

## Verification we used

- **Modal dashboard logs** for stack traces and request lifecycle.
- **HTTP smoke tests** against the deployed URL (multipart upload, optional `prompt` field).
- **`swift test`** filters documented in `MODAL-STT.md` for single-clip integration and optional eval-set WER/latency passes.
- **Local host debug log** (`/tmp/localwispr-debug.log` per runbook) for client-side timings.

---

## Known limitations / follow-ups

- **Hinting / `prompt`:** currently ignored server-side for HF Whisper compatibility; revisit with an explicit design if product needs biases or domain vocabulary.
- **Artifacts:** local smoke JSON/WAV files and Python **`__pycache__`** under `tools/` should stay **out of Git**; rely on `MODAL-STT.md` for reproducible checks.
- **Worktree workflow:** prefer one checkout on **`modal-initial`** for feature work, or merge to `main` when ready; remember the **one-branch-one-worktree** Git rule when switching.

---

## Related documents

- [`MODAL-STT.md`](MODAL-STT.md) — deploy, env, tests, metrics.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — overall system shape and pipeline.
