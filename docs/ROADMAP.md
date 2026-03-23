# Roadmap & future ideas

This file captures **non-binding** directions for future work. The shipped app may already implement some ideas partially; check the code and [ARCHITECTURE.md](ARCHITECTURE.md) for the source of truth.

---

## Latency & pipeline

- **Faster stop-to-visible-text** — Measure **stop → transcription** vs **cleanup** vs **insert** separately (already surfaced in UI); consider:
  - Tighter finalization / grace windows for live session.
  - **Insert** immediately after transcript is ready, run **cleanup** in the background, then **optional safe replace** if focus and text context still match (requires careful AX/pasteboard safety).
- **Persistent analyzer/session** across runs — reduce repeated model/session startup where the Speech APIs allow.
- **Structured latency types** — Extend or split `PipelineLatency` if background cleanup or replace steps become first-class.

---

## Product & UX

- **Export / benchmark** — Optional debug export of run IDs and timings for local benchmarking.
- **Settings** — More visible diagnostics toggles if needed for support.

---

## Engineering

- **Sandbox** — Currently the host target may use sandbox entitlements selectively; revisit before Mac App Store distribution.
- **CI** — Add GitHub Actions (or similar) for `swift test` on supported macOS runners when available.

---

## How this relates to old planning

Earlier internal notes explored a **sub-300ms stop-to-text** target with streaming STT, instant insert, and background cleanup. The **current** codebase already uses **live streaming** transcription with **batch fallback**; **pipeline ordering** and **replace semantics** may still evolve—see `Pipeline.swift` and `AppState.stopDictation`.
