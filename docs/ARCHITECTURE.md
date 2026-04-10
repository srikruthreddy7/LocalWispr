# LocalWispr — Architecture & decisions

This document describes how the app is structured, how data flows through it, and the main engineering choices. It reflects the **current** codebase, not a hypothetical plan.

---

## High-level shape

```
┌─────────────────────────────────────────────────────────────────┐
│  LocalWisprHost (AppKit + SwiftUI)                                 │
│  MenuBarExtra + NSWindow (Control Panel)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  AppState (@MainActor)                                           │
│  Orchestrates: hotkey, audio, transcriber, pipeline, history, UI│
└────────────────────────────┬────────────────────────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌─────────────┐      ┌───────────────┐      ┌────────────────┐
│ HotkeyMonitor│      │ AudioCapture  │      │ Transcriber   │
│ (CGEvent tap)│      │ AVAudioEngine │      │ Parakeet EOU  │
└─────────────┘      └───────┬───────┘      │ + Speech fallback │
                             │      └───────┬───────────────┘
                             │              │
                             └──────┬───────┘
                                    ▼
                           ┌────────────────┐
                           │ Pipeline       │
                           │ TextCleaner +  │
                           │ TextInserter   │
                           └────────────────┘
```

---

## Runtime entry

- **`AppHost/LocalWisprHost/App.swift`** — `@main` SwiftUI `App` with a **`MenuBarExtra`** (menu bar UI) and **`NSApplicationDelegateAdaptor`** for the control panel window.
- **`NSApplicationDelegate`** (`ControlPanelWindowController`) creates an **`NSWindow`** hosting **`ControlPanelView`** (SwiftUI). The app is an **`LSUIElement`** (no Dock icon by default).
- **`AppState.shared.bootstrap()`** runs on launch: hotkey wiring, permission checks, optional accessibility alert, async history load, model prewarm, speech/mic permission refresh.

---

## Dictation lifecycle

### Start

1. User triggers **`toggleDictation()`** (hotkey or UI).
2. **`ensurePermissionsForDictation()`** checks microphone + speech (and accessibility if hotkey requires it).
3. **`Transcriber.startSession`** builds a **live** session (Parakeet EOU streaming on Apple Silicon when enabled; otherwise Apple speech live path).
4. **`AudioCapture.start()`** begins capture; each buffer is forwarded to **`session.append(buffer)`**.
5. State becomes **`.listening`**.

### Stop

1. **Mic stops** and buffers are **drained** (for diagnostics and fallback).
2. **Live session** is **`finish()`**-ed (with timeout). Result is the primary transcript string.
3. If **no live** result but buffers exist, **`transcribe(buffers:)`** batch path runs.
4. **`Pipeline.process`** runs (see below).
5. **`handlePipelineResult`** updates UI, last latency, transcript history, status line.

---

## Pipeline (post-transcription)

**`Pipeline`** is an actor that:

1. **Normalizes** raw text; empty → **`.noSpeech`** with latency metrics.
2. **Cleanup** — **`TextCleaner`** (`Cleaning` protocol) using **Foundation Models / Apple Intelligence** when available. On failure or unavailability, **raw** text is used and optional warnings are recorded.
3. **Insertion** — **`TextInserter`** (`Inserting` protocol):
   - Prefer **Accessibility** direct insertion into the focused element when possible.
   - Else **pasteboard** + **simulated Cmd+V** to the frontmost app (with pasteboard restore).

**Important:** Today, **cleanup completes before insertion** in a single `process` call. The pipeline records **stop-to-transcript**, **cleanup**, and **insertion** durations separately.

---

## Transcription

- **`Transcriber`** prefers **Parakeet Flash via FluidAudio** for low-latency local streaming on Apple Silicon.
- Live path uses incremental streaming decode during capture and final flush on stop.
- Fallback path remains Apple Speech (`SpeechAnalyzer` / legacy recognizer), then optional cloud STT fallback when enabled.
- **Prewarm** (`prewarmModels`) warms both modes on boot to reduce first-session latency.
- **Modes** (`TranscriberMode`): **dictationLong** vs **speechTranscription** — different analyzer presets for long dictation vs speech-style transcription.
- **Fallback:** If live finalization fails or times out, buffered audio can be processed with a **batch** transcribe path.

---

## Text cleanup

- **`TextCleaner`** (`FoundationModels`) applies an LLM prompt for formatting/punctuation; includes **regex sanitization** (code fences, filler words, bullet handling, etc.) when needed.
- **Availability** is checked at runtime; if Apple Intelligence is unavailable, cleanup may be skipped with a clear reason.

---

## Hotkeys

- **`HotkeyMonitor`** + **`HotkeyFactory`** install a **CGEvent tap** (or equivalent) for global shortcuts.
- **`GlobalHotkeyBinding`** is persisted in **UserDefaults**; changing it recreates the monitor.
- Some bindings require **Accessibility**; others may use different event paths — see `HotkeyFactory` and `HotkeyMonitor`.

---

## UI

- **Menu bar** — `MenuBarView` + status icon.
- **Control panel** — `ControlPanelView`: Dashboard (grid of cards), History (`TranscriptHistoryView`), Settings.
- **Layout** — Sidebar + main area; GeometryReader-based height for short windows; **custom button styles** (`solidProminent`, `subtleGlass`) replace system glass styles so controls **do not dim** when the window is not key.
- **Hosting** — `NSHostingView` with `intrinsicContentSize` set to no intrinsic metric and low content hugging so the root fills the window (avoids sidebar height shrinking with tab content).

---

## Persistence

- **`TranscriptHistoryStore`** — local transcript records (paths and format in that source file).
- **UserDefaults** — hotkey binding, transcriber mode, etc.

---

## Testing

- **`Tests/LocalWisprTests/`** — unit tests for `TextCleaner`, pipeline assembly, transcript assembly, latency stats, hotkey detector, etc.
- `AppState` is injectable in tests where needed (see initializers).

---

## Product decisions (summary)

| Decision | Rationale |
|----------|-----------|
| **Local-only default path** | Privacy and no cloud dependency for core dictation. |
| **Menu bar + optional control panel** | Always-available control without cluttering the Dock (`LSUIElement`). |
| **Live session + batch fallback** | Best latency when possible; robustness when live finalization fails. |
| **Cleanup before insert** | Simpler correctness: user sees inserted text match “final” cleaned output when insertion succeeds. |
| **AX + paste fallback** | Maximizes compatibility across apps that don’t expose AX text APIs. |
| **Custom button styles** | System `.glass` styles dim when the window is inactive; custom styles keep contrast stable. |

---

## Latency SLO

Hard product target for end-to-end latency from user stop action to final text inserted into target app:

- **p90 < 1.0s**
- **p99 <= 1.5s**

This SLO includes transcription, cleanup, and insertion.

---

## Related

- **[ROADMAP.md](ROADMAP.md)** — Future ideas (e.g. sub-300ms stop-to-text explorations).
