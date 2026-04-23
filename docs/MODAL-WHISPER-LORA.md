# Modal Whisper LoRA Runbook

This runbook covers the experiment path for:

1. LoRA fine-tuning `openai/whisper-large-v3-turbo` on Indian-accent English.
2. Evaluating the resulting adapter on AI4Bharat `Svarah`.

The implementation lives in [`tools/modal_whisper_lora_experiment.py`](../tools/modal_whisper_lora_experiment.py).

The current reliable-data follow-up plan, including the April 22 shortlist for Indic-TIMIT, Common Voice Indian-accent slices, and NPTEL, lives in [`docs/RELIABLE-SPEECH-DATA.md`](./RELIABLE-SPEECH-DATA.md).

Indic-TIMIT archive download and extraction inside Modal now use the lightweight helpers in [`tools/modal_volume_downloader.py`](../tools/modal_volume_downloader.py) and [`tools/modal_indic_timit_volume_tools.py`](../tools/modal_indic_timit_volume_tools.py).

The training script also supports local JSONL manifests mounted from the artifacts volume, plus an explicit validation dataset. This is the current path for Indic-TIMIT train/validation manifests generated inside Modal.

The training script now supports comma-separated config lists for dataset-backed supplements such as the confirmed Vaani district pool. This is the intended way to attach `Vaani supplement v1` as a second training source without pre-merging it offline.

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

### 2026-04-22: Indic-TIMIT acquisition, training, and failure

The Indic-TIMIT archives were downloaded directly into the Modal artifacts volume, extracted there, and converted into JSONL manifests so no local machine download was needed.

Core artifacts:

- `/artifacts/datasets/indic-timit-v2-splits/train.jsonl`
- `/artifacts/datasets/indic-timit-v2-splits/validation.jsonl`
- `/artifacts/datasets/indic-timit-v2-splits/test.jsonl`

The script was updated to support local JSONL manifests, explicit validation datasets, multiple dataset configs, multi-GPU DDP training, detached `train_only` / `evaluate_saved_run` phases, and progress JSON under each run directory.

Full Indic-TIMIT-only run:

- run id: `whisper-turbo-indic-timit-4gpu-v2-20260422-193019`
- hardware: `4x H100`
- result on full Svarah:
  - base WER: `0.0816364495`
  - adapter WER: `0.1044603122`
  - WER delta: `+0.0228238627`
  - base CER: `0.0388499588`
  - adapter CER: `0.0564148805`
  - CER delta: `+0.0175649217`

Failure analysis:

- numeric/formatted subset WER delta: `+0.0374042927`
- numeric/formatted subset CER delta: `+0.0307107053`
- non-numeric subset WER delta: `+0.0034944089`
- non-numeric subset CER delta: `-0.0002671199`

Interpretation:

- Indic-TIMIT is not safe as a broad primary source for this Svarah target.
- The model did not simply fail on accent; it damaged Whisper's formatting and language priors.
- The next experiments should reduce model movement and avoid broad Indic-TIMIT mixing.

### 2026-04-22/23: CV + Indic-TIMIT and encoder-only LoRA

The next test mixed Common Voice Indian-accent data with a small Indic-TIMIT anchor.

Run:

- run id: `whisper-turbo-accent-cv-indictimit-v1-20260422-222751`
- primary: `WillHeld/india_accent_cv`, `16384` rows, `1-6s`
- anchor: Indic-TIMIT train JSONL, `2048` rows, `1-6s`
- epochs: `0.5`
- learning rate: `2e-6`
- LoRA: rank `16`, alpha `32`, dropout `0.05`, attention targets
- hardware: `5x H100`

Result:

- adapter WER: `0.0859722679`
- WER delta vs base: `+0.0043358184`
- adapter CER: `0.0407357791`
- CER delta vs base: `+0.0018858203`

This was much less bad than full Indic-TIMIT-only, but still worse than base.

To isolate whether decoder LoRA was damaging Whisper's language/formatting behavior, the script added `--lora-scope encoder`. Encoder-only LoRA freezes adapter parameters outside the encoder after PEFT construction.

Encoder-only CV-only run:

- run id: `whisper-turbo-accent-encoder-cv-only-v1-20260423-183154`
- primary: `WillHeld/india_accent_cv`, `16384` rows, `1-6s`
- no anchor
- epochs: `0.5`
- learning rate: `2e-6`
- hardware: `5x H100`
- eval: stable single-H100 Svarah-only path

Result:

- adapter WER: `0.0822231444`
- WER delta vs base: `+0.0005866949`
- adapter CER: `0.0392647362`
- CER delta vs base: `+0.0004147774`

Encoder-only CV + Indic-TIMIT run:

- run id: `whisper-turbo-accent-encoder-cv-indictimit-v1-20260423-183156`
- primary: `WillHeld/india_accent_cv`, `16384` rows, `1-6s`
- anchor: Indic-TIMIT train JSONL, `2048` rows, `1-6s`
- epochs: `0.5`
- learning rate: `2e-6`
- hardware: `5x H100`
- eval: stable single-H100 Svarah-only path

Result:

- adapter WER: `0.0823805503`
- WER delta vs base: `+0.0007441009`
- adapter CER: `0.0396331410`
- CER delta vs base: `+0.0007831822`

Interpretation:

- Decoder LoRA was a major part of the earlier damage.
- Encoder-only LoRA reduces regression sharply.
- Indic-TIMIT still hurts even with decoder drift mostly removed.
- CV-only is better than CV + Indic-TIMIT for this benchmark.

### 2026-04-23: Per-sample Svarah analysis and data selection

The next analysis compared base, the old ultragentle winner, encoder-only CV-only, encoder-only CV + Indic-TIMIT, and all-attention CV + Indic-TIMIT on a 1024-sample Svarah probe.

Command shape:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode analyze_svarah \
  --analysis-name svarah-data-selection-probe-v1 \
  --compare-run-ids whisper-turbo-accent-probe-cv-ultragentle-v1-20260421-152853,whisper-turbo-accent-encoder-cv-only-v1-20260423-183154,whisper-turbo-accent-encoder-cv-indictimit-v1-20260423-183156,whisper-turbo-accent-cv-indictimit-v1-20260422-222751 \
  --compare-labels old_cv_ultragentle,encoder_cv_only,encoder_cv_indictimit,allattn_cv_indictimit \
  --svarah-max-samples 1024 \
  --per-device-eval-batch-size 8 \
  --analysis-top-examples 20
```

Artifacts:

- `/artifacts/svarah-data-selection-probe-v1-20260423-191333/report.json`
- `/artifacts/svarah-data-selection-probe-v1-20260423-191333/pairwise_predictions.jsonl`

Overall probe metrics:

| Model | WER | WER delta | CER | CER delta |
| --- | ---: | ---: | ---: | ---: |
| Base | `0.084092` | - | `0.038677` | - |
| `old_cv_ultragentle` | `0.083997` | `-0.000095` | `0.038711` | `+0.000034` |
| `encoder_cv_only` | `0.083997` | `-0.000095` | `0.039002` | `+0.000325` |
| `encoder_cv_indictimit` | `0.085419` | `+0.001327` | `0.039345` | `+0.000668` |
| `allattn_cv_indictimit` | `0.085703` | `+0.001612` | `0.039465` | `+0.000788` |

Pairwise counts versus base:

| Model | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: |
| `old_cv_ultragentle` | `5` | `5` | `1014` |
| `encoder_cv_only` | `37` | `32` | `955` |
| `encoder_cv_indictimit` | `38` | `39` | `947` |
| `allattn_cv_indictimit` | `37` | `43` | `944` |

Important slices:

- On `<3s` utterances, `encoder_cv_only` improved WER by `-0.010577`.
- On `3-6s` utterances, every newer larger run regressed; `old_cv_ultragentle` stayed slightly better than base.
- On digit-bearing utterances, `old_cv_ultragentle` preserved WER while encoder-only CV regressed slightly and encoder CV + Indic-TIMIT regressed more.
- On `1-2` word utterances, the stronger runs improved WER, but that benefit was offset by regressions on longer utterances.

Conclusion:

- The project should not chase broad adaptation yet.
- The next run should combine better row selection with the old ultragentle update size.
- Success criterion is still beating base Whisper on Svarah, not improving train-domain validation.

### 2026-04-23: Candidate data audit

`ishands/commonvoice-indian_accent` became the best next source after the public data hunt.

Profile result over 50000 rows:

- total source rows: `110088`
- sampled rows: `50000`
- mean duration: `5.35s`
- p50 duration: `5.256s`
- p90 duration: `7.728s`
- exact India/South Asia accent metadata: `49567/50000`
- downvoted rows: `12130/50000`
- rows over `8s`: `3903/50000`
- digit-bearing transcripts: `2/50000`

Selection probe:

- artifact: `/artifacts/cv-indian-accent-selection-probe-v3-20260423-191752`
- selected rows: `16384`
- unique speakers: `1953`
- speaker cap: `50`
- selected duration mean: `4.86s`
- selected duration p90: `6.79s`

Training-ready curated manifest:

- artifact: `/artifacts/cv-indian-accent-curated-4k-v2-20260423-192247`
- training JSONL: `/artifacts/cv-indian-accent-curated-4k-v2-20260423-192247/train.jsonl`
- selected rows: `4096`
- unique speakers: `1290`
- speaker cap: `8`
- selected duration mean: `4.54s`
- selected duration p90: `5.81s`
- audio directory: `/artifacts/cv-indian-accent-curated-4k-v2-20260423-192247/audio`

Rejected or deferred sources:

- `skit-ai/skit-s2i`: too template-heavy and narrow-domain for broad accent adaptation.
- `En1gma02/processed_indian_accent_english`: only `6765` rows, read-story style, weak metadata, too many long rows.
- `edinburghcstr/edacc`: has `1004` Indian-English rows across validation/test, useful as a side eval but not primary training data.
- Sarvam public Hugging Face datasets: useful benchmark/eval material, but no usable Indian-accent English ASR training corpus found.

New tooling added:

- `build_training_manifest`: metadata scoring, rejection reasons, speaker caps, progress JSON, optional audio export, training JSONL output.
- `profile_train`: duration, word count, duplicate text, metadata top-values, quality warnings, score distribution.
- `analyze_svarah`: per-sample prediction JSONL with base/adapters and CER/WER deltas.

Next exact run:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --experiment-name whisper-turbo-accent-curated-cv-1k-v1 \
  --train-dataset /artifacts/cv-indian-accent-curated-4k-v2-20260423-192247/train.jsonl \
  --train-split train \
  --train-audio-column audio \
  --train-text-column text \
  --train-max-samples 1024 \
  --num-train-epochs 0.5 \
  --learning-rate 5e-6 \
  --rank 16 \
  --alpha 32 \
  --dropout 0.05 \
  --target-module-set attention \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --skip-validation-eval
```

Completed result:

- run id: `whisper-turbo-accent-curated-cv-1k-v1-20260423-193101`
- adapter: `/artifacts/whisper-turbo-accent-curated-cv-1k-v1-20260423-193101/adapter`
- report: `/artifacts/whisper-turbo-accent-curated-cv-1k-v1-20260423-193101/report.json`
- train rows: `1024`
- optimizer steps: `16`
- preprocess runtime: `4m28s`
- train runtime: `37.9444s`
- train loss: `2.8712950945`

Full Svarah result:

| Model | WER | CER |
| --- | ---: | ---: |
| Base | `0.0816364495` | `0.0388499588` |
| Curated CV 1k adapter | `0.0816507591` | `0.0388267725` |
| Delta | `+0.0000143096` | `-0.0000231863` |

Decision:

- do not scale this exact recipe to `2048` rows yet
- the WER target did not beat base, even though CER improved slightly
- next step is pairwise diagnosis, not another blind data-size increase

Pairwise diagnostic:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode analyze_svarah \
  --analysis-name svarah-curated-1k-diagnostic-v1 \
  --compare-run-ids whisper-turbo-accent-probe-cv-ultragentle-v1-20260421-152853,whisper-turbo-accent-curated-cv-1k-v1-20260423-193101 \
  --compare-labels old_cv_ultragentle,curated_cv_1k \
  --svarah-max-samples 1024 \
  --per-device-eval-batch-size 8 \
  --analysis-top-examples 20 \
  --analysis-group-fields duration_bucket,word_count_bucket,contains_digit,contains_date_like,contains_currency_or_amount,gender,age-group,primary_language,native_place_state,occupation_domain
```

Completed diagnostic:

- analysis id: `svarah-curated-1k-diagnostic-v1-20260423-195029`
- report: `/artifacts/svarah-curated-1k-diagnostic-v1-20260423-195029/report.json`
- pairwise predictions: `/artifacts/svarah-curated-1k-diagnostic-v1-20260423-195029/pairwise_predictions.jsonl`
- Svarah sample count: `1024`

1024-sample Svarah comparison:

| Model | WER | CER | WER delta vs base |
| --- | ---: | ---: | ---: |
| Base | `0.0840917710` | `0.0386769668` | `0` |
| Old CV ultragentle | `0.0839969662` | `0.0387112245` | `-0.0000948047` |
| Curated CV 1k | `0.0841865757` | `0.0387626111` | `+0.0000948047` |

Pairwise movement vs base:

| Adapter | Improved | Unchanged | Worsened |
| --- | ---: | ---: | ---: |
| Old CV ultragentle | `5` | `1014` | `5` |
| Curated CV 1k | `4` | `1017` | `3` |

Useful diagnostic slices:

| Slice | Old CV ultragentle WER delta | Curated CV 1k WER delta | Read |
| --- | ---: | ---: | --- |
| `<3s` | `-0.0009615` | `0` | curated data did not help short Svarah utterances |
| `3-6s` | `-0.0002975` | `-0.0005951` | curated data helped this middle-duration band |
| `6-10s` | `0` | `+0.0003311` | curated data started hurting longer utterances |
| `10s+` | `+0.0003198` | `+0.0006396` | curated data hurt longest utterances more |
| digit-bearing | `0` | `+0.0006859` | curated data still does not protect numeric/form-sensitive cases |
| `6-10` words | `-0.0009042` | `-0.0009042` | both adapters help this band |
| `11+` words | `+0.0001302` | `+0.0003905` | curated data hurts longer textual contexts |

Observed curated regressions:

- inserted `that` in `Then throughout the story...`
- dropped a leading `And` in one sample
- changed `Perfume` to `Parfume`

Observed curated improvements:

- corrected one `we are`/`we or` confusion
- improved a TataCliq spacing/form issue
- improved several phonetically-close noisy hypotheses, though not always to the exact reference

Conclusion:

- the curated 1k recipe is safer than the larger CV+Indic runs, but it still does not beat base WER
- the selection was too conservative: it over-focused short, clean Common Voice rows and under-covered longer Svarah-style utterances, named entities, product-like words, and formatting-sensitive text
- do not run a `2048`/`4096` scale-up of this exact manifest
- the next useful step is to compare the old winning 1024-row source distribution against the curated manifest, then change the selection objective before training again

### April 23, 2026 selection-profile analysis

New tooling added:

- `profile_train_selection`: profiles the exact training rows selected by the same train/validation split and sampling path used by `train_eval`
- `build_training_manifest --manifest-selection-strategy bucketed_transfer`: keeps the existing quality/rejection filters but prioritizes longer contextual utterances before sorting by score
- `verify_audio_manifest`: checks every local JSONL audio path with `soundfile.info`, writes progress, and reports sample rate/channel/duration integrity

Exact selected-row profiles:

| Selection | Mean duration | p50 duration | p90 duration | `6-10s` rows | `11+` word rows | Unique speakers | Format-sensitive rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old CV ultragentle 1k | `5.642s` | `5.508s` | `7.812s` | `390` | `541` | `271` | `18` |
| Curated CV 1k | `4.535s` | `4.620s` | `5.753s` | `40` | `264` | `661` | `0` |
| Bucketed transfer probe 1k | `5.768s` | `6.120s` | `7.416s` | `579` | `598` | `657` | `18` |

Interpretation:

- the failed curated 1k run was too short and too clean relative to the only base-beating recipe
- the old winning recipe was not special because of Indian lexical text; both old and curated selected only `4` explicit Indian lexical-marker rows
- the likely useful signal was longer Indian-accent speech over richer English sentences, not domain text
- the new bucketed transfer recipe deliberately restores that length/context shape while keeping stronger speaker diversity than the old sample

Metadata-only bucketed probe:

- run id: `cv-indian-accent-bucketed-transfer-4k-probe-v1-20260423-200908`
- manifest: `/artifacts/cv-indian-accent-bucketed-transfer-4k-probe-v1-20260423-200908/manifest.jsonl`
- output rows: `4096`
- selection strategy: `bucketed_transfer`
- selected duration mean: `5.748s`
- selected duration p50: `6.144s`
- selected duration p90: `7.488s`
- unique speakers: `1313`

Audio-backed bucketed manifest:

- run id: `cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156`
- training JSONL: `/artifacts/cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156/train.jsonl`
- manifest JSONL: `/artifacts/cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156/manifest.jsonl`
- audio dir: `/artifacts/cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156/audio`
- output rows: `4096`
- selected duration mean: `5.748s`
- selected duration p50: `6.144s`
- selected duration p90: `7.488s`
- unique speakers: `1313`

Audio verification:

- verify id: `cv-indian-accent-bucketed-transfer-4k-v1-audio-verify-v2-20260423-201910`
- report: `/artifacts/cv-indian-accent-bucketed-transfer-4k-v1-audio-verify-v2-20260423-201910/report.json`
- checked rows: `4096`
- valid rows: `4096`
- failed rows: `0`
- sample rate counts: `16000: 4096`
- channel counts: `1: 4096`

Next training candidate:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --experiment-name whisper-turbo-accent-bucketed-transfer-1k-v1 \
  --train-dataset /artifacts/cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156/train.jsonl \
  --train-split train \
  --train-audio-column audio \
  --train-text-column text \
  --train-max-samples 1024 \
  --num-train-epochs 0.5 \
  --learning-rate 5e-6 \
  --rank 16 \
  --alpha 32 \
  --dropout 0.05 \
  --target-module-set attention \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --skip-validation-eval
```

Do not scale beyond `1024` rows unless this beats base WER on Svarah.

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
