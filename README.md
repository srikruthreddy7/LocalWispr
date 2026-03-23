# LocalWispr

**LocalWispr** is a macOS app for **local voice dictation**: it captures speech with the microphone, transcribes on-device with Apple’s speech stack, optionally **cleans** text with **Apple Intelligence** (Foundation Models), and **inserts** the result at the cursor in the frontmost app.

There are no cloud services in the default path—audio and models stay on your Mac, subject to Apple’s speech and system APIs.

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
- **Live transcription session** using `SpeechAnalyzer` with buffer streaming; **batch fallback** if live finalization fails or times out.
- **Two transcriber modes**: Dictation (long-form) and Transcription (speech-oriented), user-selectable in Settings.
- **Text cleanup** via `TextCleaner` (Apple Intelligence when available), with deterministic fallbacks and guardrail handling.
- **Insertion** via Accessibility (direct text insertion when possible) or pasteboard + simulated paste.
- **Dashboard UI** with control panel (Dashboard, History, Settings).
- **Local transcript history** persisted on disk for review and copy.
- **Latency metrics** (stop → transcript, cleanup, insert) surfaced in the UI.

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

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Components, data flow, permissions, and product/technical decisions.
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — Future ideas (e.g. latency optimization directions), not a commitment.
- **[docs/RELEASE.md](docs/RELEASE.md)** — Building the downloadable DMG, checksums, and GitHub Releases.

---

## Privacy & security

- **Microphone** and **Speech Recognition** are used only for dictation; see `Info.plist` usage strings.
- **Accessibility** is used to register global hotkeys and to insert or paste text into other applications.
- Transcript history is stored **locally**; no network upload is implemented in the core paths described here.

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
