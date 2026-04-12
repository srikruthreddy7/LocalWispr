# Roadmap & future ideas

This file captures **non-binding** directions for future work. The shipped app may already implement some ideas partially; check the code and [ARCHITECTURE.md](ARCHITECTURE.md) for the source of truth.

Hard latency SLO (binding across roadmap items):
- **p90 < 1.0s** from user finishing dictation to final text inserted.
- **p99 <= 1.5s** for the same end-to-end path.

For detailed research behind the cloud pivot, see:
- **[CLOUD-LLM-RESEARCH.md](CLOUD-LLM-RESEARCH.md)** — provider comparison, transport decisions, prompt strategy
- **[CODEBASE-CONTEXT-RESEARCH.md](CODEBASE-CONTEXT-RESEARCH.md)** — editor detection, identifier extraction, ASR biasing, IDE extensions

---

## Cloud LLM exploration (researched March 2026)

Apple's on-device Foundation Model (3B params) has hit a quality ceiling: 2.5-3.8s latency for paragraph text, cannot fix phonetically-similar word substitutions, 4096 token limit. Research explored replacing the on-device LLM cleanup with a cloud LLM while keeping Apple's on-device ASR.

### Research areas explored

**Cloud LLM cleanup** — Cerebras and Groq emerged as the most promising providers for low-latency inference. OpenAI-compatible APIs, sub-500ms total latency, <$0.05/1K dictations. A `CloudTextCleaner` conforming to the existing `Cleaning` protocol could integrate cleanly. See [CLOUD-LLM-RESEARCH.md](CLOUD-LLM-RESEARCH.md).

**Codebase-aware dictation** (novel idea) — No existing tool combines project-wide identifier indexing, ASR vocabulary biasing, and post-recognition fuzzy matching. Research explored:
- Editor context detection via window title parsing (AX API)
- Identifier extraction via regex (MVP) and tree-sitter (production)
- ASR biasing via `AnalysisContext.contextualStrings` (not `contentHints` — that's an enum, not custom vocabulary)
- Post-processing fuzzy matching (Levenshtein + compound word joining)
- VS Code companion extension (~150 lines TypeScript) and JetBrains MCP client
- See [CODEBASE-CONTEXT-RESEARCH.md](CODEBASE-CONTEXT-RESEARCH.md).

---

## Latency & pipeline

- **Faster stop-to-visible-text** — Measure **stop -> transcription** vs **cleanup** vs **insert** separately (already surfaced in UI); consider:
  - Tighter finalization / grace windows for live session.
  - **Insert** immediately after transcript is ready, run **cleanup** in the background, then **optional safe replace** if focus and text context still match (requires careful AX/pasteboard safety).
- **Persistent analyzer/session** across runs — reduce repeated model/session startup where the Speech APIs allow.
- **Structured latency types** — Extend or split `PipelineLatency` if background cleanup or replace steps become first-class.

---

## Product & UX

- **Export / benchmark** — Optional debug export of run IDs and timings for local benchmarking.
- **Settings** — API key configuration UI for cloud LLM providers. Provider selection (Cerebras / Groq / custom endpoint).
- **Offline indicator** — Show when cloud cleanup is unavailable and the app is using local-only fallback.

---

## Engineering

- **Sandbox** — Currently the host target may use sandbox entitlements selectively; revisit before Mac App Store distribution. Note: Accessibility API (required for editor context detection) does NOT work in sandboxed apps.
- **CI** — Add GitHub Actions (or similar) for `swift test` on supported macOS runners when available.

---

## How this relates to old planning

Earlier internal notes explored a **sub-300ms stop-to-text** target with streaming STT, instant insert, and background cleanup. The **current** codebase already uses **live streaming** transcription with **batch fallback**; **pipeline ordering** and **replace semantics** may still evolve -- see `Pipeline.swift` and `AppState.stopDictation`.

The cloud LLM pivot targets **<1s total latency** (ASR ~200ms + cloud cleanup ~400-700ms + insert ~20ms) with dramatically better quality than the on-device model.
