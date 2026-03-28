# Cloud LLM Cleanup — Research Findings

*Date: March 2026*

Research into the feasibility of replacing Apple's on-device Foundation Model (3B params) with a cloud LLM for text cleanup, and adding codebase-aware context to improve developer dictation accuracy. These are research findings and ideas for exploration, not a committed execution plan.

---

## Background

Apple's on-device Foundation Model has known limitations (documented in prior testing):

- **2.5-3.8 seconds** latency for paragraph-length text
- Cannot fix phonetically-similar word substitutions without few-shot examples
- 4096 combined token limit (input + output)
- Frequently adds unwanted preamble despite instructions
- Currently restricted to short utterances (<=160 chars, <=2 sentences) with 450ms timeout

A cloud LLM could potentially deliver **<500ms cleanup latency** with dramatically better quality, while keeping Apple's on-device ASR (which is fast and free).

---

## Part 1: Cloud LLM Provider Comparison

### Latency & Pricing (as of March 2026)

| Provider | Model | Est. Total Latency | Input $/M | Output $/M | Cost/1K dictations | API Format |
|---|---|---|---|---|---|---|
| **Groq** | Llama 3.1 8B | 200-300ms | $0.05 | $0.08 | $0.013 | OpenAI-compat |
| **Groq** | GPT-OSS 20B | 250-400ms | $0.075 | $0.30 | $0.038 | OpenAI-compat |
| **Groq** | Llama 4 Scout (17Bx16E) | 350-500ms | $0.11 | $0.34 | $0.045 | OpenAI-compat |
| **Groq** | Llama 3.3 70B | 400-700ms | $0.59 | $0.79 | $0.14 | OpenAI-compat |
| **Cerebras** | Llama 3.1 8B | 150-250ms | $0.10 | $0.10 | $0.02 | OpenAI-compat |
| **Cerebras** | GPT-OSS 120B | 300-400ms | $0.35 | $0.75 | $0.11 | OpenAI-compat |
| **Fireworks** | Llama 3.1 8B | 300-600ms | $0.10 | $0.10 | $0.02 | OpenAI-compat |
| **OpenAI** | GPT-5.4 Nano | 300-600ms | $0.20 | $1.25 | $0.145 | OpenAI |
| **Google** | Gemini 3.1 Flash Lite | 232-380 t/s but 7s TTFT (preview!) | $0.25 | $1.50 | $0.175 | Google |
| **Anthropic** | Claude 3.5 Haiku | 600-1200ms | $0.80 | $4.00 | $0.48 | Anthropic |
| **Nemotron 3 Super** | 120B/12B active (MoE) | 700-900ms (Baseten best) | $0.10 | $0.50 | $0.06 | OpenAI-compat |

*Assumes ~100 input tokens + ~100 output tokens per dictation cleanup.*

### Provider Notes

**Groq** (now NVIDIA-owned, $20B acquisition Dec 2025):
- Custom LPU silicon, purpose-built for fast inference
- Lowest TTFT in the industry (~100-200ms)
- New Groq 3 LPX chip announced (35x throughput/watt vs Blackwell)
- Prompt caching: 50% discount on cached inputs

**Cerebras**:
- Wafer-Scale Engine — fastest raw throughput (2,600 t/s on Llama 4 Scout)
- **Zero data retention by default** — strongest privacy posture
- No opt-in needed; data is never stored, logged, or reused
- Smaller model selection than Groq

**Nemotron 3 Super** (released March 11, 2026):
- Hybrid Mamba-Transformer MoE, 120B total / 12B active params
- Multi-token prediction (native speculative decoding)
- `enable_thinking: False` toggle for latency-sensitive tasks
- Best TTFT is ~0.51s (Baseten) — borderline for dictation
- Not yet available on Groq or Cerebras

### Most Promising Options

**Cerebras with Llama 3.1 8B** — fastest throughput (2,222 t/s), zero data retention by default, $0.02/1K dictations. Worth benchmarking first.

**Groq with Llama 3.1 8B or GPT-OSS 20B** — more model variety, prompt caching, proven at scale. Good alternative.

**Nemotron 3 Super on Groq (future)** — when Groq adds it, the 12B-active MoE with thinking disabled could be interesting. The 120B total / 12B active architecture is well-suited for quality-at-speed.

### Approaches That Seem Less Viable

- **Self-hosting** (Modal, BaseTen, Fly.io): $400-1,800/month for always-on GPU vs $0.40/month on Groq. Need ~1M req/day to break even. Not viable.
- **WebSockets / gRPC**: HTTP/2 connection reuse in URLSession already provides persistent connection benefits. Transport is ~5% of latency; model inference is ~90%. Not worth the complexity.
- **Custom relay servers**: Adds a hop, adds infrastructure. Negligible latency benefit.
- **Anthropic Haiku**: TTFT alone (300-600ms) eats the budget. Too slow for dictation cleanup.
- **Gemini 3.1 Flash Lite**: 7s TTFT in preview — non-starter until GA.

---

## Part 2: Transport & Streaming

### Finding: REST + SSE Streaming Appears Optimal

All research points toward simple REST POST with `stream: true` as the best approach:

- `URLSession` with HTTP/2 handles connection pooling automatically
- After the first request, subsequent requests reuse TCP+TLS connections (~0ms overhead)
- SSE parsing is ~100-200 lines of Swift — straightforward
- Both Groq and Cerebras use OpenAI-compatible endpoints

### Latency Breakdown

| Component | First request | Subsequent requests |
|---|---|---|
| TCP + TLS handshake | ~100-150ms | 0ms (HTTP/2 reuse) |
| Request transmission | ~5-10ms | ~5-10ms |
| TTFT (model) | 100-400ms | 100-400ms |
| Token streaming (~100 tok) | 50-500ms | 50-500ms |

### Prompt Caching

Both Groq and OpenAI offer prompt caching where repeated system prompts are served from cache, reducing TTFT by up to 80%. This is more impactful than any transport optimization.

---

## Part 3: Possible Implementation Architecture

### Integration with Existing Code

The `Cleaning` protocol in `Models.swift` would make integration straightforward:

```
Pipeline.swift
  └── CloudTextCleaner (new, conforms to Cleaning)
        ├── URLSession POST to Cerebras/Groq
        ├── SSE streaming parser
        ├── sanitizeModelOutput (existing post-processing)
        └── Fallback to fastClean on timeout/error
```

### Ideas for Design

1. A `CloudTextCleaner` could conform to the existing `Cleaning` protocol for clean integration
2. An 800ms timeout (vs current 450ms for on-device) would account for network variance
3. A fallback chain (Cloud LLM -> fastClean -> raw text) would handle failures gracefully
4. API key storage via macOS Keychain
5. Offline detection with fallback to on-device cleaning

### Prompt Strategy

```
You are a speech-to-text post-processor. Fix all errors and return ONLY the corrected text.

Rules:
- Fix misheard words using context
- Fix grammar, punctuation, and capitalization
- Remove filler words (um, uh, like, you know)
- Remove duplicate words from speech hesitation
- Do NOT rephrase, summarize, or add commentary

Known project identifiers: {dynamicIdentifiers}

Examples:
"tricking strategies" -> "recruiting strategies"
"get hub" -> "GitHub"
"fetch users function" -> "fetchUsers function"
```

---

## Part 4: Competitive Landscape

| Tool | Approach | Codebase Awareness | Pricing |
|---|---|---|---|
| **Wispr Flow** | Cloud AI, screen reader for active file | Active file only (screen reader hack) | $8-15/mo |
| **SuperWhisper** | Local whisper.cpp | Manual custom vocabulary | $8/mo |
| **Onit Dictate** | Local MLX on Apple Silicon | None | Free |
| **Talon** | Custom Conformer engine | Manual .talon-list files | Free (open source) |
| **Cursorless** | Structural (hats on tokens) | tree-sitter AST (no dictation) | Free (open source) |
| **GitHub Copilot Voice** | Discontinued (April 2024) | Was IDE-native | N/A |
| **LocalWispr (planned)** | Cloud LLM + project indexing | Full project + ASR biasing | Free (open source) |

**No existing tool combines**: project-wide codebase indexing + ASR vocabulary biasing + post-recognition fuzzy matching. This is genuinely novel.
