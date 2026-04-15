# Modal Whisper LoRA Runbook

This runbook covers the experiment path for:

1. LoRA fine-tuning `openai/whisper-large-v3-turbo` on Indian-accent English.
2. Evaluating the resulting adapter on AI4Bharat `Svarah`.

The implementation lives in [`tools/modal_whisper_lora_experiment.py`](../tools/modal_whisper_lora_experiment.py).

## Why this setup

The baseline training/eval pairing is:

- Train: [`WillHeld/india_accent_cv`](https://huggingface.co/datasets/WillHeld/india_accent_cv)
- Eval: [`ai4bharat/Svarah`](https://huggingface.co/datasets/ai4bharat/Svarah)

This keeps Svarah as the external benchmark instead of training directly on the benchmark itself.

After running the first full pure-accent experiment, the adapter overfit:

- base Whisper large-v3-turbo beat the adapter on full Svarah
- the adapter improved the held-out split from the training corpus
- but transfer to Svarah got worse

So the current recommended path is no longer the pure `india_accent_cv` run. The recommended recipe is now a mixed recipe:

- primary domain data: `WillHeld/india_accent_cv`
- anchor data: `openslr/librispeech_asr` (`clean`, `train.100`)
- transcript normalization enabled
- shorter training by default

The point of the anchor set is simple: keep the adapter close to competent English ASR while still nudging it toward Indian-accent speech.

The default LoRA choices follow the practical guidance from the Thinking Machines LoRA write-up:

- prefer LoRA over full fine-tuning for a small post-training dataset
- use a higher LR than full fine-tuning
- adapt both attention and MLP projections, not attention-only
- use a simple schedule first

Reference: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)

The default runtime baseline is intentionally conservative:

- GPU: Modal `H100!` for deterministic H100 allocation
- attention backend: `sdpa`
- framework stack: `transformers` + `peft` + `accelerate`

The default run does **not** enable FlashAttention 3 or FlashAttention 4.

## Prerequisites

Install the Modal CLI locally and authenticate it:

```bash
pip install modal
modal token new
```

Create a Modal secret containing a Hugging Face token with access to gated datasets such as Svarah:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

The script defaults to the secret name `huggingface-secret`. Override with:

```bash
export LOCALWISPR_MODAL_LORA_HF_SECRET_NAME=my-hf-secret
```

Override the default training GPU or attention backend only if you are deliberately experimenting:

```bash
export LOCALWISPR_MODAL_LORA_TRAIN_GPU=L40S
export LOCALWISPR_MODAL_LORA_ATTN_IMPLEMENTATION=eager
```

## Modal resources used

The script creates or reuses two Modal volumes:

- `localwispr-whisper-lora-artifacts`
- `localwispr-hf-cache`

Override names if needed:

```bash
export LOCALWISPR_MODAL_LORA_ARTIFACTS_VOLUME=my-artifacts
export LOCALWISPR_MODAL_LORA_HF_CACHE_VOLUME=my-hf-cache
```

## Step 1: Inspect the datasets

Before training, inspect the training and eval datasets from Modal. This is especially useful for gated datasets where you want to confirm split names and inferred columns.

Inspect the training dataset:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode inspect_train
```

Inspect Svarah:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode inspect_eval
```

If the inferred `audio_column` or `text_column` is wrong, pass explicit overrides in the train/eval run:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --eval-audio-column audio \
  --eval-text-column text
```

## Step 2: Run a smoke experiment

Start with a short run to verify dataset access, training, adapter save, and Svarah evaluation wiring:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --experiment-name whisper-turbo-india-accent-smoke \
  --train-max-samples 512 \
  --validation-max-samples 128 \
  --svarah-max-samples 128 \
  --num-train-epochs 1 \
  --rank 32 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4
```

That smoke run already uses the conservative baseline: `H100!` + `sdpa`.

## Step 3: Run the actual experiment

Once the smoke run is clean, use the mixed recipe:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --experiment-name whisper-turbo-india-accent-mixed-v1 \
  --recipe mixed-anchor-v1 \
  --rank 64 \
  --alpha 32 \
  --dropout 0.05 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4
```

Recipe `mixed-anchor-v1` applies these defaults unless you explicitly override them:

- `train_max_samples=40000`
- `anchor_max_samples=20000`
- `validation_max_samples=2000`
- `num_train_epochs=1`
- `learning_rate=5e-5`
- `normalize_transcripts=true`

You can still override any of those from the CLI.

## Outputs

Each run writes a timestamped directory into the artifacts volume:

```text
/artifacts/<experiment-name>-<timestamp>/
  adapter/
  report.json
  train_config.json
```

The terminal output from `modal run` includes the final JSON report. `report.json` contains:

- training dataset metadata
- eval dataset metadata
- LoRA hyperparameters
- trainer metrics
- validation metrics
- Svarah base-model WER/CER
- Svarah adapter WER/CER
- delta between the adapter and the base model

## Executed changes

The workflow in [`tools/modal_whisper_lora_experiment.py`](../tools/modal_whisper_lora_experiment.py) was changed during the first two experiment days:

- fixed the bf16 generation path by moving validation and Svarah scoring out of trainer-side generation eval
- kept the safe runtime baseline at `H100!` + `sdpa`
- added transcript normalization for training text
- added random row sampling instead of always taking the first `N`
- added the `mixed-anchor-v1` recipe
- added a detached Svarah analysis mode that compares finished adapters against the base model and breaks results down by metadata slices

The analysis mode is invoked with:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode analyze_svarah \
  --analysis-name svarah-compare-base-pure-mixed-v1 \
  --compare-run-ids whisper-turbo-india-accent-v1-20260414-175501,whisper-turbo-india-accent-mixed-v1-20260415-170305 \
  --compare-labels pure_lora,mixed_lora
```

## Experiment log

### 2026-04-14: Pure accent run

Run:

```bash
modal run --detach tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --experiment-name whisper-turbo-india-accent-v1 \
  --num-train-epochs 3 \
  --rank 64 \
  --alpha 32 \
  --dropout 0.05 \
  --learning-rate 1e-4 \
  --attn-implementation sdpa \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4
```

Artifacts:

- `/artifacts/whisper-turbo-india-accent-v1-20260414-175501/adapter`
- `/artifacts/whisper-turbo-india-accent-v1-20260414-175501/report.json`

Result summary:

| Model | Svarah WER | Svarah CER |
| --- | ---: | ---: |
| Base `whisper-large-v3-turbo` | 0.0816 | 0.0388 |
| Pure accent LoRA | 0.1281 | 0.0783 |
| Delta | +0.0464 | +0.0395 |

Additional notes:

- train runtime: about `4h 24m`
- train-domain validation WER: `0.0778`
- conclusion: this run overfit to the narrow train domain and hurt transfer to Svarah

### 2026-04-15: Mixed anchor run

Run:

```bash
modal run --detach tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --experiment-name whisper-turbo-india-accent-mixed-v1 \
  --recipe mixed-anchor-v1 \
  --rank 64 \
  --alpha 32 \
  --dropout 0.05 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4
```

Resolved recipe defaults:

- primary: `WillHeld/india_accent_cv`
- anchor: `openslr/librispeech_asr` (`clean`, `train.100`)
- `train_max_samples=40000`
- `anchor_max_samples=20000`
- `validation_max_samples=2000`
- `num_train_epochs=1`
- `learning_rate=5e-5`
- `normalize_transcripts=true`

Artifacts:

- `/artifacts/whisper-turbo-india-accent-mixed-v1-20260415-170305/adapter`
- `/artifacts/whisper-turbo-india-accent-mixed-v1-20260415-170305/report.json`

Result summary:

| Model | Svarah WER | Svarah CER |
| --- | ---: | ---: |
| Base `whisper-large-v3-turbo` | 0.0816 | 0.0388 |
| Mixed anchor LoRA | 0.1070 | 0.0617 |
| Delta | +0.0254 | +0.0228 |

Additional notes:

- train runtime: about `1h 06m`
- train-domain validation WER: `0.0917`
- conclusion: the anchor recipe recovered a large fraction of the damage from the pure accent run, but still did not beat the base model

## Analysis log

### 2026-04-15: Base vs pure vs mixed on full Svarah

Artifacts:

- `/artifacts/svarah-compare-base-pure-mixed-v1-20260415-191008/report.json`

Overall:

| Model | WER | CER |
| --- | ---: | ---: |
| Base | 0.0816 | 0.0388 |
| Mixed LoRA | 0.1070 | 0.0617 |
| Pure LoRA | 0.1281 | 0.0783 |

Per-utterance counts versus base:

| Adapter | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: |
| Mixed LoRA | 695 | 1284 | 4677 |
| Pure LoRA | 752 | 1645 | 4259 |

Key findings:

- the mixed recipe was better than the pure recipe on every subgroup slice that was analyzed
- the mixed recipe still lost to the base model on every subgroup slice with at least 100 samples
- the remaining error increased with duration, especially on `6-10s` and `10s+` utterances
- the worst mixed-recipe deltas showed up in `Bengali`, `Konkani`, `Hindi`, `Dogri`, `Malayalam` and in states such as `Goa`, `Gujarat`, `Assam`, `Telangana`, and `Jammu Kashmir`
- domain-heavy slices such as `Government`, `Healthcare`, and `Financial Services` were hit harder than `Technology and Services`
- top regressions were dominated by numeric and form-style utterances such as `7123`, `9522`, `9523`, `9515`, and date-like phrases, where the adapters expanded digits into words or repeated fragments

Interpretation:

- the anchor idea is directionally correct
- the remaining gap is not just accent adaptation; it is also a formatting and domain-preservation problem
- the next analysis pass should explicitly bucket digit-heavy, date-like, currency-like, and very short utterances before another training run is considered

### 2026-04-15: Format-sensitive follow-up analysis

Artifacts:

- `/artifacts/svarah-format-analysis-v1-20260415-202015/report.json`
- `/artifacts/svarah-format-analysis-v2-low-threshold-20260415-204349/report.json`

Key findings:

- digit-bearing utterances are the single biggest remaining failure mode
- very short utterances are much less stable than long utterances
- date-like utterances and currency/amount-style utterances are also disproportionately bad, even though they are a smaller slice of the benchmark

Derived buckets:

| Bucket | Samples | Base WER | Mixed WER | Mixed delta | Pure delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `contains_digit=yes` | 769 | 0.1348 | 0.2910 | +0.1561 | +0.2672 |
| `contains_digit=no` | 5887 | 0.0728 | 0.0765 | +0.0037 | +0.0098 |
| `contains_date_like=yes` | 60 | 0.0851 | 0.1324 | +0.0473 | +0.2279 |
| `contains_currency_or_amount=yes` | 95 | 0.1959 | 0.3396 | +0.1437 | +0.1886 |

Word-count buckets:

| Bucket | Samples | Base WER | Mixed WER | Mixed delta | Pure delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1-2 words` | 1429 | 0.4111 | 0.4868 | +0.0757 | +0.1378 |
| `3-5 words` | 662 | 0.1235 | 0.2062 | +0.0827 | +0.1328 |
| `6-10 words` | 1723 | 0.0916 | 0.1172 | +0.0256 | +0.0551 |
| `11+ words` | 2842 | 0.0663 | 0.0869 | +0.0207 | +0.0366 |

Concrete interpretation:

- the mixed recipe is close to base on non-digit utterances
- the large remaining gap is concentrated in digit normalization, form filling, banking/government phrasing, and very short commands
- the next small experiment should explicitly protect numeric and transactional formatting instead of just mixing more generic English audio

### 2026-04-15: Source dataset profiling for format coverage

The next small experiment should not be launched blind. The source datasets were profiled on the same sample budgets used by `mixed-anchor-v1`.

Primary dataset sample:

- dataset: `WillHeld/india_accent_cv`
- sampled rows: `40000`
- `contains_digit=yes`: `0`
- `contains_date_like=yes`: `542`
- `contains_currency_or_amount=yes`: `131`
- focus rows under the current format heuristic: `2712` (`6.78%`)

Anchor dataset sample:

- dataset: `openslr/librispeech_asr` (`clean`, `train.100`)
- sampled rows: `20000`
- `contains_digit=yes`: `0`
- `contains_date_like=yes`: `854`
- `contains_currency_or_amount=yes`: `297`
- focus rows under the current format heuristic: `1253` (`6.27%`)

Interpretation:

- neither current source dataset supplies literal digit-form transcripts at meaningful scale
- that means the remaining Svarah numeric gap cannot be fixed by oversampling the current data mix alone
- the repo now contains a scaffold for a `mixed-format-v1` experiment recipe, but that recipe should be treated as incomplete until a true numeric/transactional source slice is added

## Important caveats

- This workflow has been executed from this environment, but rerunning it still requires:
  - a working Modal account/session
  - a Hugging Face token with Svarah access
- `WillHeld/india_accent_cv` is the default train set because it is the most directly relevant public Indian-accent English corpus we found.
- Svarah remains a better **evaluation** set than a **training** set.
- The defaults are intentionally conservative and meant to be a first pass, not the final hyperparameter sweep.
- The safe baseline is `sdpa`; do not switch kernels until the baseline run is complete and measured.
- The first time you use `openslr/librispeech_asr` on a fresh Modal cache, the upstream dataset builder may spend several minutes preparing the clean config before training starts. Later runs reuse the cache and start much faster.

## Useful variants

Try the mixed recipe with a smaller anchor budget first:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --experiment-name whisper-turbo-india-accent-mixed-lite \
  --recipe mixed-anchor-v1 \
  --anchor-max-samples 10000
```

Try a smaller batch if you hit memory pressure:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --experiment-name whisper-turbo-india-accent-mixed-lowmem \
  --recipe mixed-anchor-v1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

## Related references

- [Thinking Machines: LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [OpenAI Whisper turbo on Hugging Face](https://huggingface.co/openai/whisper-large-v3-turbo)
- [WillHeld/india_accent_cv](https://huggingface.co/datasets/WillHeld/india_accent_cv)
- [AI4Bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
