# LocalWispr

**LocalWispr** is a macOS app for voice dictation: it captures speech with the microphone, transcribes with a **Modal-hosted Whisper large-v3-turbo** service (after-stop transcription), cleans text with a cloud LLM cleanup step, and inserts the result at the cursor in the frontmost app.

---

## Install

### Homebrew (recommended)

```bash
brew tap srikruthreddy7/localwispr
brew install --cask localwispr
```

### Manual download (DMG)

Grab the latest **`.dmg`** from [GitHub Releases](https://github.com/srikruthreddy7/LocalWispr/releases).

> **Note:** The DMG is unsigned. On first launch, right-click the app → Open → click Open. Or strip the quarantine flag:
> ```bash
> xattr -cr /Applications/LocalWisprHost.app
> ```

To **build the DMG yourself** or **cut a release**, see **[docs/RELEASE.md](docs/RELEASE.md)**.

---

## Features

- **Global shortcut** to start/stop dictation (configurable; default includes options such as Right ⌘ double-tap).
- **After-stop transcription UX**: one utterance upload after stop, one final transcript response.
- **Modal STT backend** built around Whisper large-v3-turbo on a dedicated L40S-class GPU.
- **Two transcriber modes**: Dictation (long-form) and Transcription (speech-oriented), user-selectable in Settings.
- **Text cleanup** via `TextCleaner` (Apple Intelligence when available), with deterministic fallbacks and guardrail handling.
- **Insertion** via Accessibility (direct text insertion when possible) or pasteboard + simulated paste.
- **Dashboard UI** with control panel (Dashboard, History, Settings).
- **Local transcript history** persisted on disk for review and copy.
- **Latency metrics** (stop → transcript, cleanup, insert) surfaced in the UI.

---

## Latency SLO (hard target)

LocalWispr should keep end-to-end latency from **user finishing dictation** to **final transcript appearing in the target text box** at:

- **p90 < 1.0s**
- **p99 <= 1.5s**

This is the primary performance requirement for transcription + cleanup + insertion.

---

## Design principles

- Build **general mechanisms**, not narrow fixes for one observed failure.
- Do not hardcode product behavior around a specific phrase, site, browser title pattern, app, or one-off example.
- When a test case exposes a bug, fix the underlying class of errors at the right abstraction layer.
- Any accuracy improvement must preserve the latency target above unless there is a deliberate, explicit tradeoff.
- Treat **parallelization and concurrency as a default engineering rule**: when work is independent and correctness is preserved, batch it, pipeline it, or run it concurrently instead of serializing it. New work should justify serialized execution rather than accidental parallel slack.

---

## Requirements

- **macOS 26** (project targets `.v26` in `Package.swift`; Xcode 26.x toolchain).
- Microphone, Speech Recognition, and (for global hotkey + paste) **Accessibility** permissions.
- **Apple Intelligence / Foundation Models** for full cleanup behavior when available; cleanup degrades gracefully if not.

---

## Repository layout

| Path | Purpose |
|------|--------|
| `Package.swift` | Swift Package for the **`LocalWispr`** library (core logic + UI). |
| `Sources/LocalWispr/` | App logic: `AppState`, `Transcriber`, `AudioCapture`, `Pipeline`, `TextCleaner`, `TextInserter`, `ControlPanelView`, etc. |
| `Sources/LocalWispr/Resources/` | Bundled fonts and assets. |
| `Resources/Info.plist` | Shared Info.plist (host app references this via Xcode). |
| `AppHost/LocalWisprHost.xcodeproj` | Thin **macOS app** target that links the Swift package and hosts the SwiftUI app. |
| `Tests/LocalWisprTests/` | Unit tests. |
| `tools/` | Modal service code and experiment scripts. |

The shipping app product is **`LocalWisprHost.app`** (bundle id `com.localwispr.host` in the Xcode project). The library is reusable for tests and tooling; the host is the runnable app.

---

## Build & run

### App (recommended)

1. Open `AppHost/LocalWisprHost.xcodeproj` in Xcode.
2. Select the **LocalWisprHost** scheme, **Run**.
3. Grant Microphone, Speech Recognition, and Accessibility when prompted.

### Swift Package only

```bash
swift build
swift test
```

---

## STT backend configuration

Set these environment variables (or `.env`) for STT:

- `LOCALWISPR_MODAL_STT_ENDPOINT` — full URL to your Modal transcription endpoint (`/v1/audio/transcriptions`).
- `LOCALWISPR_MODAL_STT_API_KEY` — bearer token expected by the Modal service.
- `LOCALWISPR_MODAL_STT_MODEL` — optional; defaults to `openai/whisper-large-v3-turbo`.
- `LOCALWISPR_MODAL_STT_TIMEOUT_SECONDS` — optional request timeout; defaults to `60`.

---

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Components, data flow, permissions, and product/technical decisions.
- **[docs/MODAL-STT.md](docs/MODAL-STT.md)** — Modal deployment, secrets, benchmark workflow, and rollback path.
- **[docs/MODAL-WHISPER-LORA.md](docs/MODAL-WHISPER-LORA.md)** — Modal-native Whisper LoRA experiment runbook for `india_accent_cv` → `Svarah`.
- **[docs/RELIABLE-SPEECH-DATA.md](docs/RELIABLE-SPEECH-DATA.md)** — Reliable-data acquisition checklist, 500-sample audit rubric, and the gate for the next full accent training run.
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — Future ideas (e.g. latency optimization directions), not a commitment.
- **[docs/RELEASE.md](docs/RELEASE.md)** — Building the downloadable DMG, checksums, and GitHub Releases.

---

## Privacy & security

- **Microphone** and **Speech Recognition** are used only for dictation; see `Info.plist` usage strings.
- **Accessibility** is used to register global hotkeys and to insert or paste text into other applications.
- Transcript history is stored **locally**. Captured utterance audio is uploaded to your configured Modal STT endpoint for transcription.

Review entitlements in `AppHost/LocalWisprHost/LocalWisprHost.entitlements` before shipping your own build.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

Issues and pull requests are welcome. Run the test target before submitting changes:

```bash
swift test
# or Xcode: Product → Test
```
