# Codebase Context Indexing — Research Findings

*Date: March 2026*

Research into the feasibility of making LocalWispr codebase-aware: detecting the developer's active project, extracting identifiers, and using them to improve speech-to-text accuracy at multiple levels (ASR biasing, LLM prompt injection, and post-processing fuzzy matching). These are exploratory findings, not a committed plan.

---

## The Problem

When developers dictate, speech recognition gets technical terms wrong:

| Developer says | Transcribed as | Should be |
|---|---|---|
| "update the fetchUsers function" | "update the fetch users function" | "update the fetchUsers function" |
| "import the useState hook" | "import the use state hook" | "import the useState hook" |
| "check the CLAUDE_MD variable" | "check the cloud MD variable" | "check the CLAUDE_MD variable" |
| "run pytest" | "run pie test" | "run pytest" |

The fix: extract identifiers from the project and use them to bias recognition + correct output.

---

## Part 1: Editor Detection via macOS Accessibility API

### Strategy: Window Title Parsing

Window title parsing is the most reliable, fastest (~2-5ms), and universally supported approach. It works across all editors using only the Accessibility permission LocalWispr already requests.

**Why not read editor text via AX API?**
- Electron apps (VS Code, Cursor, Windsurf) do NOT reliably expose editor text content via the AX tree
- JetBrains (Java/Swing) has partial, unreliable AX text support
- Only native macOS apps (Xcode, TextEdit) work reliably for text extraction
- Window title works everywhere

### Window Title Formats

| Editor | Format | Bundle ID |
|---|---|---|
| VS Code | `{filename} -- {projectName} -- Visual Studio Code` | `com.microsoft.VSCode` |
| Cursor | `{filename} -- {projectName} -- Cursor` | `com.todesktop.230313mzl4w4u92` |
| Windsurf | `{filename} -- {projectName} -- Windsurf` | (check with `osascript -e 'id of app "Windsurf"'`) |
| JetBrains | `{projectName} -- [{projectPath}] -- {fileName}` | `com.jetbrains.intellij`, `.pycharm`, etc. |
| Xcode | `{fileName} -- {targetName}` | `com.apple.dt.Xcode` |
| Terminal.app | Shell-set title (often CWD) | `com.apple.Terminal` |
| iTerm2 | Session name or CWD | `com.googlecode.iterm2` |
| Warp | CWD or Warp-specific | `dev.warp.Warp-Stable` |
| Ghostty | Shell-set title | `com.mitchellh.ghostty` |
| Claude Code (in terminal) | `claude - {projectDir}` | (parent terminal's bundle ID) |

### Implementation Approach

```swift
// Capture context at dictation start (~2-5ms)
let context = DictationContext.captureFromFrontmostApp()
// context.bundleIdentifier -> "com.microsoft.VSCode"
// context.windowTitle -> "TextCleaner.swift -- LocalWispr -- Visual Studio Code"
// context.parsedProjectName -> "LocalWispr"
// context.parsedFileName -> "TextCleaner.swift"
```

**Key APIs:**
- `NSWorkspace.shared.frontmostApplication` — app detection (already used in `TextInserter.swift`)
- `AXUIElementCopyAttributeValue(_, kAXFocusedWindowAttribute, _)` — focused window
- `AXUIElementCopyAttributeValue(_, kAXTitleAttribute, _)` — window title
- `AXObserver` with `kAXTitleChangedNotification` — monitor tab/file changes

### Terminal CWD Detection

For terminal apps, use `proc_pidinfo` with `PROC_PIDVNODEPATHINFO` on the shell child process to get the current working directory. Alternatively, parse the window title (which usually contains CWD).

### Project Root Detection

From the file path extracted from the window title, walk up to find markers:
- `.git/` (most reliable)
- `Package.swift`, `package.json`, `Cargo.toml`, `pyproject.toml`, `go.mod`

---

## Part 2: Identifier Extraction

### Phase 1: Regex (MVP, 1-2 days)

Simple regex to extract common identifier patterns:

```swift
// CamelCase: fetchUsers, UIViewController, getElementById
let camelCase = /\b[A-Z][a-zA-Z0-9]+|[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b/

// snake_case: fetch_users, my_variable
let snakeCase = /\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/

// SCREAMING_CASE: CLAUDE_MD, API_KEY
let screamingCase = /\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/
```

**Accuracy**: ~70-80%. False positives from comments/strings are mostly harmless for vocabulary biasing.
**Speed**: Microseconds per file.

### Phase 2: tree-sitter (3-5 days)

**SwiftTreeSitter** (v0.25.0, SPM-compatible, production-ready):
- Used by Chime editor in production
- Swift 6 compatible (strict concurrency)
- `github.com/tree-sitter/swift-tree-sitter`

**Performance:**
| File size | Parse time |
|---|---|
| 10 lines | < 1ms |
| 100 lines | ~1-2ms |
| 1,000 lines | ~5-10ms |
| 10,000 lines | ~50-100ms |
| Incremental re-parse | < 0.5ms |

**1,000 files concurrently: ~0.5-1.5 seconds**

**Binary size impact**: ~2-5MB for 10 language grammars (statically compiled).

**Key grammars (SPM packages):**
- `tree-sitter-swift` (alex-pinkus)
- `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`
- `tree-sitter-rust`, `tree-sitter-go`, `tree-sitter-java`

**Unified identifier query** (works across most languages):
```scheme
(identifier) @id
(type_identifier) @type_id
;; For Swift specifically:
(simple_identifier) @id
```

**Why tree-sitter over regex:**
- Distinguishes code from comments/strings (avoids polluting vocabulary)
- Aider (AI coding tool) switched from ctags to tree-sitter for this reason
- Incremental re-parsing is sub-millisecond
- No external binary dependency (unlike ctags)
- Sandboxable (can ship in App Store if needed)

### Phase 3: Persistent Index with File Watching (3-5 days)

- Use FSEvents (`FSEventStreamCreate` with `kFSEventStreamCreateFlagFileEvents`) for recursive directory monitoring
- Anchor to Git commit hash; on subsequent dictation starts, only re-scan changed files (`git diff --name-only`)
- LRU cache for parsed identifier sets (~100 most recently accessed files)
- Skip `.gitignore`'d paths, files >10K lines, binary files

### Indexing Tiers

| Tier | What | When | Latency |
|---|---|---|---|
| 1 | Current file (from window title) | Every dictation start | < 10ms |
| 2 | Sibling files in same directory | Async at dictation start | < 100ms |
| 3 | Full project index | Background, FSEvents-driven | Seconds (amortized) |

---

## Part 3: Using Context for Accuracy

### Layer 1: ASR Vocabulary Biasing

**Critical correction**: `contentHints` on `DictationTranscriber` is NOT custom vocabulary — it's an enum describing audio content characteristics.

**The real API**: `AnalysisContext.contextualStrings`

```swift
let context = AnalysisContext()
context.contextualStrings[.general] = ["fetchUsers", "useState", "CLAUDE_MD", "pytest"]
try await analyzer.setContext(context)
```

- Works with `SpeechAnalyzer` (macOS 26)
- Biases recognition at the ASR level — fixes errors before they reach the LLM
- No hard documented limit on count; works best with tens to low hundreds
- Provide both the identifier ("fetchUsers") and possibly the spoken form ("fetch users")

**Note**: The older `SFSpeechRecognizer.contextualStrings` (macOS 14+) is more documented and feature-rich (supports `SFCustomLanguageModelData` with custom pronunciation via X-SAMPA). The new `SpeechAnalyzer` version does NOT support custom pronunciation.

### Layer 2: LLM Prompt Injection

Include project identifiers in the cloud LLM cleanup prompt:

```
Known project identifiers (prefer these spellings):
fetchUsers, useState, CLAUDE_MD, TextCleaner, Pipeline, AppState
```

- Keep under ~150 tokens (~50-80 identifiers)
- Prioritize: current file identifiers > frequent project identifiers
- With prompt caching, the identifier list doesn't add TTFT cost

### Layer 3: Post-Processing Fuzzy Match

After LLM cleanup, run a final pass matching words against known identifiers:

**Compound word joining**:
- "fetch users" → try joining → "fetchusers" → case-insensitive match → `fetchUsers`
- "use state" → "usestate" → `useState`

**Levenshtein distance** (normalized):
- < 0.3: strong match → auto-replace
- 0.3-0.5: possible match → require phonetic confirmation

**Double Metaphone** as secondary signal:
- "cloud MD" and "CLAUDE_MD" have similar phonetic codes
- Metaphone match boosts confidence by ~30%

**Swift libraries:**
- **StringZilla** (v4.6.0) — high-performance Levenshtein, actively maintained
- **Fuzzywuzzy_swift** — port of Python's fuzzywuzzy, no external deps
- **DoubleMetaphoneSwift** — Double Metaphone encoding

**Latency**: < 1ms for the entire post-processing pass.

---

## Part 4: IDE Companion Extension

### VS Code Extension (Phase 2, ~3-4 hours)

A minimal extension (~150 lines of TypeScript) that exposes context to LocalWispr:

```
LocalWispr (macOS app)
    |  HTTP GET http://127.0.0.1:48221/context
    v
LocalWispr VS Code Extension
    |  vscode.window.activeTextEditor
    |  vscode.commands.executeCommand(documentSymbolProvider)
    v
VS Code / Cursor / Windsurf
```

**IPC**: Localhost HTTP server (1-5ms latency, debuggable with curl).

**Response payload**:
```json
{
  "projectRoot": "/Users/foo/myproject",
  "activeFile": {
    "path": "src/components/Button.tsx",
    "languageId": "typescriptreact",
    "cursorLine": 42
  },
  "currentFileSymbols": ["Button", "handleClick", "ButtonProps"],
  "workspaceSymbols": ["Button", "App", "fetchData", "UserContext"]
}
```

**Compatibility**: One `.vsix` works on VS Code, Cursor, AND Windsurf (all VS Code forks). Publish to VS Code Marketplace + Open VSX (for Windsurf).

### JetBrains (Phase 2, ~4 hours)

**Don't write a custom plugin.** JetBrains IDEs ship with a built-in MCP server since 2025.2 that exposes file access, code analysis, symbol retrieval, and project structure over HTTP. Connect as an MCP client from LocalWispr.

### Prior Art

- **Talon's command-server**: File-based IPC (`/tmp/vscode-command-server/`), writes `request.json`/`response.json`. Proven but slow (10-100ms).
- **Cursorless**: Uses tree-sitter for structural understanding, communicates via command-server.
- No general-purpose "context bridge" extension exists — this is a gap we'd fill.

---

## Part 5: Possible Implementation Phases

These are rough ideas for how implementation could be staged, not a committed plan.

### Phase 1 Ideas: Core Cloud + Context

| Idea | What | Est. Effort | Expected Impact |
|---|---|---|---|
| 1a | `CloudTextCleaner` — URLSession to Cerebras/Groq, SSE streaming, `Cleaning` protocol | 2-3 days | Replace on-device model |
| 1b | Window title parsing + project root detection | 1 day | Know active project/file |
| 1c | Regex identifier extraction from project files | 1-2 days | ~80% identifier coverage |
| 1d | `AnalysisContext.contextualStrings` wiring | 1 day | Bias ASR toward project terms |
| 1e | Fuzzy post-processing pass (Levenshtein + compound joining) | 1 day | "fetch users" -> fetchUsers |

### Phase 2 Ideas: Polish

| Idea | What | Est. Effort |
|---|---|---|
| 2a | tree-sitter integration (replace regex) | 3-5 days |
| 2b | VS Code companion extension | 3-4 hours |
| 2c | JetBrains via MCP client | 4 hours |
| 2d | FSEvents file watching + persistent index | 3-5 days |

### Phase 3 Ideas: Advanced

- Per-project vocabulary learning (track corrections)
- Phonetic index (pre-computed Soundex/Metaphone for O(1) lookup)
- Embedding-based contextual selection for very large projects
- VS Code extension with SSE push for real-time cursor/file changes

---

## Key Technical References

### Apple APIs
- `AnalysisContext.contextualStrings` — [Apple Developer Docs](https://developer.apple.com/documentation/speech/analysiscontext)
- WWDC 2025 Session 277 — [SpeechAnalyzer intro](https://developer.apple.com/videos/play/wwdc2025/277/)
- WWDC 2023 Session 10101 — [Custom on-device speech recognition](https://developer.apple.com/videos/play/wwdc2023/10101/)
- [Apple Developer Forums: improving SpeechAnalyzer transcription](https://developer.apple.com/forums/thread/801877)

### tree-sitter
- [swift-tree-sitter (SPM)](https://github.com/tree-sitter/swift-tree-sitter) — v0.25.0
- [tree-sitter-swift grammar](https://github.com/alex-pinkus/tree-sitter-swift)
- [tree-sitter code navigation (tags.scm)](https://tree-sitter.github.io/tree-sitter/4-code-navigation.html)
- [Aider: better repo map with tree-sitter](https://aider.chat/2023/10/22/repomap.html)

### Cursor / Copilot Indexing
- [How Cursor Indexes Codebases Fast](https://read.engineerscodex.com/p/how-cursor-indexes-codebases-fast)
- [Cursor Secure Codebase Indexing](https://cursor.com/blog/secure-codebase-indexing)
- [Cursor Fast Regex Search (sparse n-grams)](https://cursor.com/blog/fast-regex-search)
- [Copilot New Embedding Model](https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/)
- [Copilot @workspace Deep Dive](https://learn.microsoft.com/en-us/shows/visual-studio-code/github-copilots-workspace-deep-dive)

### Voice Coding Prior Art
- [Talon Voice community repo](https://github.com/talonhub/community)
- [Talon .talon-list files](https://talon.wiki/Customization/talon_lists/)
- [Cursorless](https://marketplace.visualstudio.com/items?itemName=pokey.cursorless)
- [Wispr Flow variable recognition](https://docs.wisprflow.ai/articles/8554805225-variable-recognition)
- [Onit Dictate](https://www.getonit.ai/)

### IDE Extensions
- [Talon command-server](https://github.com/pokey/command-server)
- [JetBrains built-in MCP server](https://www.jetbrains.com/help/idea/mcp-server.html)
- [VS Code Extension Runtime Security](https://code.visualstudio.com/docs/configure/extensions/extension-runtime-security)

### Inference Providers
- [Groq Pricing](https://groq.com/pricing)
- [Cerebras Pricing](https://www.cerebras.ai/pricing)
- [Artificial Analysis — Nemotron 3 Super](https://artificialanalysis.ai/models/nvidia-nemotron-3-super-120b-a12b)
- [Nemotron 3 Super NVIDIA Blog](https://blogs.nvidia.com/blog/nemotron-3-super-agentic-ai/)

### Phonetic Matching
- [StringZilla (Swift)](https://swiftpackageindex.com/ashvardanian/StringZilla)
- [Fuzzywuzzy_swift](https://github.com/lxian/Fuzzywuzzy_swift)
- [DoubleMetaphoneSwift](https://github.com/ZebulonRouseFrantzich/DoubleMetaphoneSwift)
- [IEEE: Recognizing Words from Source Code Identifiers](https://ieeexplore.ieee.org/document/5714421/)
