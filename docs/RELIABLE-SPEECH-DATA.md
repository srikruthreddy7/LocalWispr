# Reliable Speech Data Plan

This document defines the next phase after the Whisper LoRA recipe search.

The short version is simple:

- hyperparameter tuning is no longer the main blocker
- the blocker is training data that does not match the target
- the next full run should wait until the data passes a manual audit

April 23 update:

- the curated Common Voice 1k run did not beat base Whisper on full Svarah
- the bucketed transfer 1k run also did not beat base Whisper on full Svarah
- the next step should be stricter data evidence, not a larger run from the same selection logic

## Current conclusion

The strongest verified recipe so far is:

- dataset: `WillHeld/india_accent_cv`
- LoRA: attention-only
- learning rate: `5e-6`
- rank: `16`
- alpha: `32`
- dropout: `0.05`
- epochs: `0.5`

That recipe beats base Whisper on full Svarah by a tiny margin, but it still misses the stricter accent-only target by a tiny margin.

That means the next improvement should come from better data, not from spending more runs on the same sources.

## Target data profile

For this project, a reliable training set must be:

1. Indian-accent English, not generic English.
2. Human-transcribed or strongly validated.
3. Mostly short-to-medium utterances, with the center of gravity in the `1-6s` band.
4. Phrase or sentence style speech, not lecture-style long paragraphs.
5. Speaker-diverse and geography-diverse.
6. Clean enough that transcript error is low on manual audit.
7. Split by speaker for train/validation/test.
8. Kept separate from Svarah, which remains evaluation-only.

## Candidate sources

### 1. Vaani

Primary target.

Why:

- India-representative data across districts and states
- spontaneous speech
- transcribed subset exists
- strong demographic and regional coverage

Sources:

- [ARTPARK-IISc/Vaani on Hugging Face](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)
- [Hugging Face + IISc Vaani article](https://huggingface.co/blog/iisc-huggingface-collab)
- [Vaani dataset portal](https://vaani.iisc.ac.in/dataset)

Important note:

- Vaani access now works from Modal after refreshing `huggingface-secret` with an approved Hugging Face token.
- Vaani is organized as many district-level configs such as `Karnataka_Bangalore`, `Telangana_Hyderabad`, `TamilNadu_Chennai`, and `Maharashtra_Pune`, not a single monolithic config.
- The raw district splits are not directly usable for English-only training. You must filter to non-empty transcripts and then filter metadata such as `language=English`.

### Vaani audit result on April 21, 2026

We fully audited four likely English-heavy district configs on the `1-6s` band with non-empty transcripts:

- `Maharashtra_Pune`: reject
- `TamilNadu_Chennai`: reject
- `Telangana_Hyderabad`: keep as supplement
- `Karnataka_Bangalore`: keep as supplement

Why:

- `Pune` had real transcribed speech after filtering, but the usable subset was `Hindi`, `Marathi`, and `Urdu`, not English.
- `Chennai` collapsed to `Tamil` only after filtering.
- `Hyderabad` had a real English subset with `618` transcribed `1-6s` rows.
- `Bangalore` had a real English subset with `1338` transcribed `1-6s` rows.

Estimated English audio scale from the audited districts:

- `Hyderabad English`: about `0.51h`
- `Bangalore English`: about `1.17h`
- combined: about `1.69h`

What this means:

- Vaani is useful, but not yet a primary source for the next training run.
- For our current target, Vaani is best treated as a curated supplemental source unless we mine many more districts.
- The English transcripts contain structured markup like `<noise>`, `[horn]`, and `{gym}`. This is manageable because it is regular enough to strip during preprocessing.

### Broader Vaani sweep on April 21, 2026

After the first district audit, we ran a broader search in two stages:

1. a stats sweep across all `166` Vaani configs to find districts with nontrivial raw English presence
2. exact filtered probes on the strongest candidates using:
   - `language=English`
   - `isTranscriptionAvailable=Yes`
   - non-empty transcripts
   - `1-6s` duration band

Confirmed exact districts so far:

- `Assam_KamrupMetropolitan`: `3881` rows, about `3.57h`
- `ArunachalPradesh_LowerDibangvalley`: `1254` rows, about `1.12h`
- `Karnataka_Bangalore`: `1338` rows, about `1.17h`
- `Telangana_Hyderabad`: `618` rows, about `0.51h`
- `Sikkim_Gangtok`: `299` rows, about `0.45h`
- `Gujarat_Valsad`: `306` rows, about `0.35h`

Confirmed exact subtotal:

- `7696` rows
- about `7.17h`

This changes the recommendation:

- Vaani is still **not** the main corpus for the next training run.
- But it is now a **meaningful supplement**, not just a tiny side source.
- The right move is to build `Vaani supplement v1` from these confirmed districts first, then optionally add more districts in descending order of expected yield.

Operationally, this means:

1. keep the confirmed six-district English slice
2. strip annotation markup during preprocessing
3. combine this Vaani slice with the best existing accent source instead of training on Vaani alone
4. avoid spending more time on obviously weak districts like `Pune` and `Chennai`

### 2. Common Voice

Supplement only.

Why:

- open and validated
- useful as backfill for speaker diversity

Why not primary:

- read speech
- weaker control over India-specific accent targeting

Source:

- [Mozilla Common Voice](https://www.mozillafoundation.org/en/common-voice/platform-and-dataset/)

### 3. FLEURS

Dev and sanity-check dataset, not core training data.

Why:

- clean
- useful as a stable secondary check

Why not primary:

- too small for the main fine-tune
- the Hugging Face builder configs do not expose an India-English config; in practice this is not an Indian-English training source

Source:

- [google/fleurs](https://huggingface.co/datasets/google/fleurs)

## Acquisition checklist

Use this checklist before any new full training run.

1. Refresh Hugging Face access in Modal.

   If Vaani still shows as gated from Modal after approval:

   ```bash
   modal secret create huggingface-secret HF_TOKEN=hf_xxx
   ```

   If you use a different secret name, also set:

   ```bash
   export LOCALWISPR_MODAL_LORA_HF_SECRET_NAME=my-hf-secret
   ```

2. Inspect the dataset from Modal.

   Vaani requires a concrete district config. Start with a likely English-heavy district:

   ```bash
   modal run tools/modal_whisper_lora_experiment.py \
     --mode inspect_train \
     --train-dataset ARTPARK-IISc/Vaani \
     --train-config-name Karnataka_Bangalore
   ```

   FLEURS generic English sanity check:

   ```bash
   modal run tools/modal_whisper_lora_experiment.py \
     --mode inspect_train \
     --train-dataset google/fleurs \
     --train-config-name en_us \
     --train-trust-remote-code
   ```

3. Identify the usable English or English-mixed subset.

   Required:

   - transcribed rows only
   - Indian English or English-mixed rows only
   - no empty transcripts

   In practice for Vaani, start with:

   - `--train-require-text`
   - `--train-metadata-filters language=English,isTranscriptionAvailable=Yes`

4. Apply the first-pass filters.

   Start with:

   - `1-6s` duration
   - optional transcript cap around `25` words
   - no obvious transcript corruption

5. Generate a 500-sample audit bundle.

   Generic command:

   ```bash
   modal run tools/modal_whisper_lora_experiment.py \
     --mode build_audit_manifest \
      --audit-name vaani-audit-v1 \
      --train-dataset ARTPARK-IISc/Vaani \
      --train-config-name Karnataka_Bangalore \
      --train-min-duration-seconds 1 \
      --train-max-duration-seconds 6 \
      --train-require-text \
      --train-metadata-filters language=English,isTranscriptionAvailable=Yes \
      --audit-sample-limit 500 \
      --audit-max-samples-per-group 0 \
      --audit-group-fields gender,state,district \
      --audit-export-audio \
      --audit-normalize-transcripts
   ```

   If the dataset uses nonstandard columns, pass explicit audio and text columns after `inspect_train`.

   Use `--audit-max-samples-per-group 0` once the slice is already narrowed to English-only rows and you want the full `500`-sample bundle. The default cap of `50` is useful for broad heterogeneous audits, but it will underfill narrow slices.

   For a broader config sweep before the exact audit, use:

   ```bash
   modal run tools/modal_whisper_lora_experiment.py \
     --mode survey_train_configs \
     --survey-name vaani-english-sweep-v1 \
     --survey-max-workers 8 \
     --train-dataset ARTPARK-IISc/Vaani \
     --train-min-duration-seconds 1 \
     --train-max-duration-seconds 6 \
     --train-require-text \
     --train-metadata-filters language=English,isTranscriptionAvailable=Yes
   ```

6. Review the audit bundle manually.

   The audit bundle writes:

   - `report.json`
   - `manifest.csv`
   - `manifest.jsonl`
   - `audio/*.wav`

7. Reject the dataset if it misses the thresholds below.

8. Only after it passes, create a speaker-separated train/validation split and launch the next full run.

## Current Vaani recommendation

Confirmed `Vaani supplement v1` right now:

1. `Assam_KamrupMetropolitan`
2. `ArunachalPradesh_LowerDibangvalley`
3. `Karnataka_Bangalore`
4. `Telangana_Hyderabad`
5. `Sikkim_Gangtok`
6. `Gujarat_Valsad`

Use it as a supplement, not the main train corpus.

Exact Modal training shape with `india_accent_cv` primary plus `Vaani supplement v1`:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --experiment-name whisper-turbo-accent-cv-plus-vaani-v1 \
  --train-dataset WillHeld/india_accent_cv \
  --train-split train \
  --anchor-dataset ARTPARK-IISc/Vaani \
  --anchor-config-name Assam_KamrupMetropolitan,ArunachalPradesh_LowerDibangvalley,Karnataka_Bangalore,Telangana_Hyderabad,Sikkim_Gangtok,Gujarat_Valsad \
  --anchor-min-duration-seconds 1 \
  --anchor-max-duration-seconds 6 \
  --anchor-require-text \
  --anchor-metadata-filters language=English,isTranscriptionAvailable=Yes \
  --num-train-epochs 0.5 \
  --learning-rate 5e-6 \
  --rank 16 \
  --alpha 32 \
  --dropout 0.05 \
  --target-module-set attention \
  --normalize-transcripts \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4
```

Important implementation detail:

- comma-separated `--anchor-config-name` values now load and concatenate multiple configs into one training source
- transcript markup is stripped during preprocessing, so Vaani tags like `<noise>` and `[horn]` do not leak into training labels

Verified on April 21, 2026:

- a tiny end-to-end training smoke with `india_accent_cv` primary plus `Telangana_Hyderabad,Karnataka_Bangalore` as the Vaani supplement completed successfully
- the smoke run id was `whisper-turbo-vaani-multiconfig-train-smoke-v1-20260421-214240`

### CV + Vaani supplement result on April 21, 2026

We tested the first real combined recipe against the earlier `CV-only ultragentle` winner.

Probe runs:

- `whisper-turbo-accent-probe-cv-vaani-balanced-v1-20260421-221324`
  - `1024` CV + `1024` Vaani supplement rows
- `whisper-turbo-accent-probe-cv-vaani-strong-v1-20260421-221346`
  - `1024` CV + `2048` Vaani supplement rows

Probe result on the `512`-sample Svarah slice:

- balanced Vaani: same WER as the old winner, slightly worse CER
- stronger Vaani: worse than the old winner

Full verification result:

- `CV-only ultragentle`
  - full Svarah WER `0.0814361`
  - accent-only WER `0.0718013`
- `CV + Vaani balanced`
  - full Svarah WER `0.0817366`
  - accent-only WER `0.0721124`

Conclusion:

- the current confirmed Vaani supplement does **not** improve the accent target
- the combined recipe is technically valid but behaviorally worse
- do **not** use `Vaani supplement v1` in the next main training run unless new districts materially change the mix

## Next data hunt on April 22, 2026

After ruling out the current Vaani mix, we searched for the next data sources that are both operationally usable and meaningfully aligned with Indian-accent English ASR.

### Tier 1: best next sources

#### 1. Indic-TIMIT

Best aligned source if we can get access.

Why it matters:

- a recent ASR paper describes Indic-TIMIT as a phonetically rich Indian English corpus with about `240h` of speech from `80` speakers
- the same paper says the corpus contains `18` L1 native accents of read English
- the paper's data-availability section says Indic-TIMIT is available **on request**, not through a turnkey Hugging Face dataset
- that same work trained on less than `25h` of the Hindi-accent subset and reported strong cross-accent generalization, which is exactly the kind of signal we care about

What this means:

- this is the strongest next target for the project
- it is much better aligned than Vaani English slices
- it is still read speech, but it is explicitly Indian English and accent-focused instead of generic English with weak accent metadata

Recommendation:

1. request Indic-TIMIT access from SPIRE
2. if granted, make it the next primary training source to evaluate

#### 2. `ishands/commonvoice-indian_accent`

Best immediately usable open-source source.

What we verified:

- dataset size: `110,088` rows
- mean duration: `5.36s`
- estimated total audio: about `163.9h`
- it is a Common Voice-style English corpus filtered to Indian/South Asian accent metadata
- the dominant accent string is the exact `India and South Asia (India, Pakistan, Sri Lanka)` label

Why it matters:

- it is larger than the current `WillHeld/india_accent_cv` source
- it stays in the short-utterance, transcribed, Common Voice regime that already gave us the only base-beating recipe so far
- unlike the Vaani supplement path, it does not introduce a large spontaneous-speech distribution jump

What this means:

- this is the best immediate open source dataset to test next
- if we stay fully open-source, this should be the next source we inspect and audit

Operational note:

- our current metadata filtering supports exact-value matches, so we can already filter the strict accent string
- if we later want stronger Common Voice quality gates like `up_votes >= 2` and `down_votes == 0`, that needs a small extension to numeric filter expressions

Suggested first inspection:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode inspect_train \
  --train-dataset ishands/commonvoice-indian_accent
```

Suggested first audit slice:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode build_audit_manifest \
  --audit-name cv-indian-accent-audit-v1 \
  --train-dataset ishands/commonvoice-indian_accent \
  --train-min-duration-seconds 1 \
  --train-max-duration-seconds 8 \
  --train-require-text \
  --train-metadata-filters "accents=India and South Asia (India, Pakistan, Sri Lanka)" \
  --audit-sample-limit 500 \
  --audit-group-fields gender,age,accents \
  --audit-max-samples-per-group 50 \
  --audit-export-audio \
  --audit-normalize-transcripts
```

### Tier 2: useful, but not the first next run

#### 3. AI4Bharat `NPTEL2020-Indian-English-Speech-Dataset`

Best open scale, but noisy and domain-shifted.

What we verified from the dataset repo:

- `15,700h` total
- `6,253,389` chunks
- average chunk length `3-10s`
- source is NPTEL YouTube educational content
- authors explicitly say they did **not** manually annotate the data and assume NPTEL used Google ASR plus corrections
- authors still reported that a quick fine-tune on a sample of the data substantially improved results on their manually cleaned pure set

Why it matters:

- it is clearly Indian-English and huge
- but it is lecture and education heavy, with transcript noise
- that profile is much closer to the TIE failure mode than to the CV-only ultragentle win

Recommendation:

- do **not** use this as the first next supervised LoRA run
- keep it for later as a weak auxiliary source or warmup source after we confirm a cleaner primary dataset

#### 4. `skit-ai/skit-s2i`

Small but real Indian-English telephony corpus.

What we verified:

- `11,845` total utterances
- `10,445` train rows
- mean duration `4.21s`
- about `12.2h` in train, about `13.8h` total by rough estimate
- `11` Indian speakers
- telephony audio
- banking intent domain
- text comes from utterance templates tied to intent classes

Why it matters:

- it is genuinely Indian English
- it is short-utterance and telephony-like
- but it is narrow-domain and small-speaker compared with what we want for the main accent adaptation run

Recommendation:

- use only as a narrow telephony supplement or a side evaluation source
- do **not** make it the next primary training corpus

Suggested inspection:

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode inspect_train \
  --train-dataset skit-ai/skit-s2i \
  --train-text-column template
```

### Rejected or low-value paths

#### `westbrook/English_Accent_DataSet`

Not enough Indian data.

What we verified:

- `50,382` train rows total
- only `1,351` train rows are tagged as `Indian`
- rough Indian subset size is only about `1.65h` by duration estimate

Recommendation:

- reject for the next run

#### `DarshanaS/IndicAccentDb`

Useful accent resource, not ASR training data.

What we verified:

- HF dataset has `8,116` rows and about `3.2 GB`
- recordings cover six Indian L1 accent groups speaking Harvard sentences
- the HF dataset exposes `audio` and `label`, but no transcript field

Recommendation:

- reject for direct ASR fine-tuning

#### `humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR`

Not an Indian-English source.

What we verified:

- the dataset is multilingual Indic speech
- sample rows are Assamese, Bhojpuri, Chhattisgarhi and other non-English languages

Recommendation:

- reject for this accent-only English objective

#### `DataoceanAI/Indian_English_Speech_Recognition_Corpus_Conversations`

Potentially useful commercial option, but not directly reproducible.

What we verified:

- the vendor page advertises `200h`, `200` speakers, mobile recordings, and free dialogue
- the Hugging Face repo for the same dataset is currently empty

Recommendation:

- treat this as a paid fallback only
- do not make it the default next data path

### Updated recommendation

The next data path should be:

1. request and inspect `Indic-TIMIT`
2. in parallel, audit `ishands/commonvoice-indian_accent`
3. hold `NPTEL2020` for later weak-supervision use
4. keep `skit-ai/skit-s2i` as a small telephony side source only
5. do not spend more time on `Vaani supplement v1`, `westbrook`, or `IndicAccentDb` for the next main run

## Indic-TIMIT on Modal on April 22, 2026

We now have a Modal-native path for handling Indic-TIMIT without downloading the archives to the local machine.

New tooling:

- [tools/modal_volume_downloader.py](/Volumes/SSD/temp/LocalWispr/tools/modal_volume_downloader.py)
  - downloads external archives straight into the Modal volume
  - validates final size against `Content-Length`
  - runs files in parallel through Modal task mapping
- [tools/modal_indic_timit_volume_tools.py](/Volumes/SSD/temp/LocalWispr/tools/modal_indic_timit_volume_tools.py)
  - extracts a chosen archive inside the same Modal volume
  - profiles the extracted tree

Current download target:

- volume: `localwispr-whisper-lora-artifacts`
- archive root: `/datasets/indic-timit-v2`

Current extraction target:

- `/datasets/indic-timit-v2-extracted`

### First extraction result: Punjabi

We selectively extracted the smallest ready shard first:

- archive: `IISc_IndicTIMIT_PUN.tar.gz`
- extracted root:
  - `/datasets/indic-timit-v2-extracted/IISc_IndicTIMIT_PUN/IISc_IndicTIMIT_PUN`

Profile result for the Punjabi shard:

- `4684` wav files
- `2` speaker directories:
  - `F37`
  - `F41`
- transcript index files:
  - `Transcripts_PUN`
  - `transcript_details.txt`

Observed transcript format:

- `Transcripts_PUN` maps file ids directly to text:
  - `IISc_IndicTIMIT_F37_1 He picked up the powder canister and ran out.`
  - `IISc_IndicTIMIT_F41_1 He picked up the powder canister and ran out.`
- `transcript_details.txt` stores prompt inventory metadata such as:
  - `Phoneme_L5_0047.WRD # He picked up the powder canister and ran out. # 1`

What this means:

- the corpus is straightforward to convert into a Hugging Face style ASR table
- the transcript mapping is not per-file sidecar text; it is a central language-level transcript index
- the speaker directories look clean and regular enough for scripted ingestion

### Recommended next ingestion step

Once the remaining language archives finish downloading:

1. extract each language shard into `/datasets/indic-timit-v2-extracted`
2. build a small converter that:
   - walks speaker directories
   - parses the language transcript index
   - emits rows of `{audio_path, transcript, speaker_id, language_shard}`
3. run a corpus profile over all extracted shards before launching training

### Completed on April 22, 2026

That ingestion step is now done.

Full extracted + indexed Indic-TIMIT result:

- full manifest:
  - `/datasets/indic-timit-v2-index/manifest.full.jsonl`
- manifest summary:
  - `/datasets/indic-timit-v2-index/manifest.full.summary.json`

Current full-manifest counts:

- total rows: `135,836`
- shard counts:
  - `IISc_IndicTIMIT_HIN`: `30,446`
  - `IISc_IndicTIMIT_KAN`: `18,736`
  - `IISc_IndicTIMIT_MAR`: `23,420`
  - `IISc_IndicTIMIT_MLY`: `18,736`
  - `IISc_IndicTIMIT_PUN`: `4,684`
  - `IISc_IndicTIMIT_TAM`: `21,078`
  - `IISc_IndicTIMIT_TEL`: `18,736`
- unique speaker directories across the current shard set: `58`

Speaker-separated split result:

- train manifest:
  - `/datasets/indic-timit-v2-splits/train.jsonl`
- validation manifest:
  - `/datasets/indic-timit-v2-splits/validation.jsonl`
- split summary:
  - `/datasets/indic-timit-v2-splits/split.summary.json`

Current split counts:

- train rows: `119,442`
- validation rows: `16,394`
- validation strategy:
  - one held-out speaker per shard for this first split

### Training wiring result

The Whisper LoRA training path now supports:

1. local JSONL manifests stored in the Modal artifacts volume
2. an explicit validation dataset, so Indic-TIMIT can use the speaker-separated split instead of an internal random split

Smoke verification already passed with:

- train manifest:
  - `/artifacts/datasets/indic-timit-v2-splits/train.jsonl`
- validation manifest:
  - `/artifacts/datasets/indic-timit-v2-splits/validation.jsonl`
- smoke run id:
  - `whisper-turbo-indic-timit-smoke-v1-20260422-162410`

The smoke run confirmed:

- local manifest loading works
- explicit validation loading works
- Whisper LoRA training completes
- validation scoring completes
- Svarah evaluation completes

## 500-sample audit rubric

Score each sampled clip across the following fields.

### Required labels

1. `language_fit`

- `pass`: Indian English or English-mixed speech that we would want the model to learn from
- `fail`: mostly non-English or otherwise off-target

2. `transcript_fit`

- `pass`: transcript matches the audio closely enough for ASR training
- `fail`: transcript is meaningfully wrong, truncated, or misaligned

3. `audio_fit`

- `pass`: speech is intelligible and clipped correctly
- `fail`: severe noise, overlap, truncation, or silence

4. `style_fit`

- `pass`: phrase or sentence style speech
- `fail`: long lecture-style, paragraph-style, or otherwise too language-model-heavy

5. `accent_fit`

- `pass`: clearly Indian-accent English
- `fail`: not the accent family we are targeting, or impossible to tell

6. `notes`

- free-text field for anything unusual

### Acceptance thresholds

Do not train if the 500-sample audit misses any of these:

- `language_fit pass` >= `90%`
- `transcript_fit pass` >= `95%`
- `audio_fit pass` >= `95%`
- `style_fit pass` >= `85%`
- `accent_fit pass` >= `85%`

If a candidate misses the thresholds, either:

- tighten the filters and re-audit
- or discard that source for the next run

## April 23, 2026 data-selection update

The current evidence says the next improvement is not "more Indian English audio" by itself. The only clearly base-beating result so far was the old `CV-only ultragentle` probe, and its `train_config.json` shows it used only `1024` rows from `WillHeld/india_accent_cv`:

- `num_train_epochs`: `0.5`
- `learning_rate`: `5e-6`
- `rank`: `16`
- `alpha`: `32`
- `dropout`: `0.05`
- `target_module_set`: `attention`

The later larger runs moved more predictions but did not beat base. A 1024-sample Svarah comparison on April 23 confirms the pattern:

| Model | Svarah probe WER | Delta vs base | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | `0.084092` | - | - | - | - |
| `old_cv_ultragentle` | `0.083997` | `-0.000095` | `5` | `5` | `1014` |
| `encoder_cv_only` | `0.083997` | `-0.000095` | `37` | `32` | `955` |
| `encoder_cv_indictimit` | `0.085419` | `+0.001327` | `38` | `39` | `947` |
| `allattn_cv_indictimit` | `0.085703` | `+0.001612` | `37` | `43` | `944` |

Artifact:

- `/artifacts/svarah-data-selection-probe-v1-20260423-191333/report.json`
- `/artifacts/svarah-data-selection-probe-v1-20260423-191333/pairwise_predictions.jsonl`

Interpretation:

- `Indic-TIMIT` is still harmful for Svarah transfer in these recipes.
- encoder-only LoRA reduces damage, but it does not beat the old ultragentle full-attention recipe on the full benchmark.
- the old recipe wins because it barely moves the model, not because it learned a broad new accent model.
- the strongest next move is better row selection plus the old ultragentle update size.

### Current source audit

`WillHeld/india_accent_cv` remains a good source, but it is smaller and already mostly consumed by the earlier probe. A 4000-row profile showed:

- mean duration `5.68s`
- `0` digit-bearing transcripts
- exact Common Voice India/South Asia accent metadata on `3972/4000` sampled rows
- `1005/4000` downvoted rows

`ishands/commonvoice-indian_accent` is now the best next source to exploit. A 50000-row profile showed:

- source rows loaded: `50000` from `110088` total rows
- mean duration `5.35s`, p50 `5.256s`, p90 `7.728s`
- exact India/South Asia accent metadata on `49567/50000` sampled rows
- `12130/50000` downvoted rows
- `3903/50000` rows over `8s`
- top duplicate texts include one-word numerals like `seven`, `six`, `zero`, `nine`

The first metadata-only selection probe over this source produced:

- artifact: `/artifacts/cv-indian-accent-selection-probe-v3-20260423-191752`
- selected rows: `16384`
- unique speakers: `1953`
- max samples per speaker: `50`
- selected duration mean: `4.86s`, p90 `6.79s`
- rejections: `84` rows for many downvotes, `20123` rows skipped by speaker cap

This is the right pool to use next, but not as a broad 16k-row training run first. The old best used `1024` rows, so the next run should test `1024`, `2048`, and `4096` curated rows before scaling.

A training-ready 4096-row manifest was then exported:

- artifact: `/artifacts/cv-indian-accent-curated-4k-v2-20260423-192247`
- training JSONL: `/artifacts/cv-indian-accent-curated-4k-v2-20260423-192247/train.jsonl`
- selected rows: `4096`
- unique speakers: `1290`
- max samples per speaker: `8`
- selected duration mean: `4.54s`, p90 `5.81s`
- audio exported to: `/artifacts/cv-indian-accent-curated-4k-v2-20260423-192247/audio`

`skit-ai/skit-s2i` should not be primary accent data. A 4000-row profile showed:

- mean duration `4.19s`
- narrow banking/IFSC/card domain
- heavy template duplication, for example `unauthorised transaction` appeared `81` times in the sample
- only `11` known speakers in the dataset description/path

Use this only as a small domain supplement if we explicitly target banking commands later.

`En1gma02/processed_indian_accent_english` is low priority. A 1000-row profile showed:

- only `6765` total rows
- no useful accent metadata
- read-story style transcripts
- p90 word count `23`, max word count `140`
- `114/1000` sampled rows over `8s`

`edinburghcstr/edacc` has an Indian-English subset, but only validation/test splits:

- validation Indian English rows: `373`
- test Indian English rows: `631`

Keep it as a side evaluation set or a tiny manual-analysis source, not as primary training data.

Sarvam does not currently solve the training-data problem. The public Sarvam datasets checked on Hugging Face are benchmarks/evals or non-English Indic tasks, not a usable Indian-accent English ASR training corpus.

### New data-selection tooling

`tools/modal_whisper_lora_experiment.py` now has:

- `build_training_manifest`: scores rows, rejects bad metadata, caps speakers, and exports a training-ready JSONL
- `analyze_svarah`: writes full pairwise per-sample predictions to `pairwise_predictions.jsonl`
- richer `profile_train`: duration, word count, metadata, duplicate text, warning, and quality-score summaries

The selection scoring intentionally favors:

- Indian/South Asian English accent metadata
- English locale/language metadata
- clean vote history
- `1-6s` audio, with soft tolerance to `8s`
- moderate word count
- speaker diversity

It penalizes:

- many downvotes
- very long audio/text
- very short repeated prompts
- markup/noise strings
- format-sensitive numeric/currency/date text
- non-Indian or non-English metadata

## Training gate

Do not schedule the next full LoRA run until we have all of this:

- `20-50h` of usable transcribed Indian-accent English
- `200+` speakers
- `10+` states or regions
- speaker-separated splits
- passing 500-sample audit

## Next small training experiment

Before a full run, use the curated Common Voice Indian-accent manifest and repeat the old ultragentle recipe at small row budgets:

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

Then repeat only if the `1024` row run beats base:

- `2048` rows
- `4096` rows

Stop as soon as the curve turns negative. The goal is accent transfer with minimum model movement.

### Curated 1024-row result

The first curated Common Voice Indian-accent run completed after this plan was written.

Run:

- run id: `whisper-turbo-accent-curated-cv-1k-v1-20260423-193101`
- training JSONL: `/artifacts/cv-indian-accent-curated-4k-v2-20260423-192247/train.jsonl`
- train rows: `1024`
- epochs: `0.5`
- learning rate: `5e-6`
- LoRA: rank `16`, alpha `32`, dropout `0.05`, attention targets, all scope
- GPU: `H100!`

Training:

- preprocess runtime: `4m28s`
- optimizer steps: `16`
- train runtime: `37.9444s`
- train loss: `2.8712950945`

Full Svarah result:

| Model | WER | CER |
| --- | ---: | ---: |
| Base | `0.0816364495` | `0.0388499588` |
| Curated CV 1k adapter | `0.0816507591` | `0.0388267725` |
| Delta | `+0.0000143096` | `-0.0000231863` |

Decision:

- do **not** scale this exact recipe to `2048` rows yet
- WER did not beat base, even though CER improved slightly
- treat this as a near-tie, not a win
- pairwise diagnostics are required before changing the data recipe

### Curated 1024-row pairwise diagnostic

The diagnostic compared base Whisper, the old best `WillHeld/india_accent_cv` ultragentle adapter, and the new curated Common Voice 1k adapter on the same 1024 Svarah examples.

Run:

- analysis id: `svarah-curated-1k-diagnostic-v1-20260423-195029`
- report: `/artifacts/svarah-curated-1k-diagnostic-v1-20260423-195029/report.json`
- pairwise predictions: `/artifacts/svarah-curated-1k-diagnostic-v1-20260423-195029/pairwise_predictions.jsonl`
- compared adapters:
  - `old_cv_ultragentle`: `whisper-turbo-accent-probe-cv-ultragentle-v1-20260421-152853`
  - `curated_cv_1k`: `whisper-turbo-accent-curated-cv-1k-v1-20260423-193101`

Overall on the 1024-sample Svarah probe:

| Model | WER | CER | WER delta vs base |
| --- | ---: | ---: | ---: |
| Base | `0.0840917710` | `0.0386769668` | `0` |
| Old CV ultragentle | `0.0839969662` | `0.0387112245` | `-0.0000948047` |
| Curated CV 1k | `0.0841865757` | `0.0387626111` | `+0.0000948047` |

Per-sample movement vs base:

| Adapter | Improved | Unchanged | Worsened |
| --- | ---: | ---: | ---: |
| Old CV ultragentle | `5` | `1014` | `5` |
| Curated CV 1k | `4` | `1017` | `3` |

Important slices:

| Slice | Old CV ultragentle WER delta | Curated CV 1k WER delta | Interpretation |
| --- | ---: | ---: | --- |
| `<3s` duration | `-0.0009615` | `0` | curated selection did not help the shortest Svarah examples |
| `3-6s` duration | `-0.0002975` | `-0.0005951` | curated selection helped this band more than the old adapter |
| `6-10s` duration | `0` | `+0.0003311` | curated selection started to hurt longer utterances |
| `10s+` duration | `+0.0003198` | `+0.0006396` | curated selection hurt the longest utterances more |
| digit-bearing examples | `0` | `+0.0006859` | curated selection still does not cover numeric/form-sensitive behavior |
| `6-10` word examples | `-0.0009042` | `-0.0009042` | both adapters helped the medium word-count band |
| `11+` word examples | `+0.0001302` | `+0.0003905` | curated selection hurt longer textual contexts |

Concrete curated regressions:

- inserted an extra `that` into one `Then throughout the story...` sample
- dropped a leading `And` from one reference
- changed `Perfume` to `Parfume`

Concrete curated improvements:

- improved a `we are`/`we or` confusion
- improved a TataCliq spacing/form issue
- improved several phonetically-close hypotheses, even where the final reference still was not perfect

Interpretation:

- The curated 1k run was not a disaster, but it also did not beat base WER.
- It moved fewer samples than the old and encoder-only probes, which is why the full result is nearly tied.
- The data selection over-corrected toward short, clean, high-agreement Common Voice rows.
- Svarah needs some messier but reliable examples: longer utterances, named entities, product-like words, and Indian-English lexical patterns.
- Numeric/form-sensitive examples remain fragile and should be protected during evaluation, not blindly optimized with weak synthetic text.

Decision after the diagnostic:

- do **not** scale this exact curated 1k recipe
- reproduce/profile the old winning 1024-row `WillHeld/india_accent_cv` distribution: done
- compare old-vs-curated training rows by duration, transcript shape, entity-like text, duplicate text, and speaker diversity: done
- build a revised manifest that keeps the current quality filters but deliberately restores longer and entity-heavy rows before the next training run: done

### Old-vs-curated training-row profile

The analysis used the new `profile_train_selection` mode, so the profiles below are not generic dataset summaries. They reflect the exact rows selected by the training path after the same train/validation split and sampling logic used by `train_eval`.

Artifacts:

- old profile: `/artifacts/old-cv-ultragentle-selection-profile-train-selection-profile-20260423-200041/report.json`
- curated profile: `/artifacts/curated-cv-1k-selection-profile-metadata-v3-train-selection-profile-20260423-200619/report.json`
- bucketed probe profile: `/artifacts/bucketed-transfer-4k-probe-selection-profile-v1-train-selection-profile-20260423-201014/report.json`

Key comparison:

| Selection | Mean duration | p50 duration | p90 duration | `6-10s` rows | `11+` word rows | Unique speakers | Format-sensitive rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old CV ultragentle 1k | `5.642s` | `5.508s` | `7.812s` | `390` | `541` | `271` | `18` |
| Curated CV 1k | `4.535s` | `4.620s` | `5.753s` | `40` | `264` | `661` | `0` |
| Bucketed transfer probe 1k | `5.768s` | `6.120s` | `7.416s` | `579` | `598` | `657` | `18` |

What this means:

- the curated 1k manifest was too short and too clean
- the old base-beating run exposed the model to many more `6-10s` utterances and `11+` word transcripts
- explicit Indian lexical text was not the differentiator: old and curated both had only `4` Indian lexical-marker rows
- the next recipe should preserve clean accent metadata but match the old length/context shape much more closely

### Bucketed transfer manifest

The new manifest strategy is `bucketed_transfer`. It keeps the `accent_safe` rejection rules, keeps the speaker cap, and changes selection away from pure top-score rows toward long-context rows.

Metadata-only probe:

- run id: `cv-indian-accent-bucketed-transfer-4k-probe-v1-20260423-200908`
- selected rows: `4096`
- unique speakers: `1313`
- selected duration mean: `5.748s`
- selected duration p50: `6.144s`
- selected duration p90: `7.488s`

Audio-backed training-ready manifest:

- run id: `cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156`
- training JSONL: `/artifacts/cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156/train.jsonl`
- manifest JSONL: `/artifacts/cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156/manifest.jsonl`
- audio dir: `/artifacts/cv-indian-accent-bucketed-transfer-4k-v1-20260423-201156/audio`
- selected rows: `4096`
- unique speakers: `1313`
- selected duration mean: `5.748s`
- selected duration p50: `6.144s`
- selected duration p90: `7.488s`

Audio integrity check:

- verify id: `cv-indian-accent-bucketed-transfer-4k-v1-audio-verify-v2-20260423-201910`
- report: `/artifacts/cv-indian-accent-bucketed-transfer-4k-v1-audio-verify-v2-20260423-201910/report.json`
- checked rows: `4096`
- valid rows: `4096`
- failed rows: `0`
- sample rate counts: `16000: 4096`
- channel counts: `1: 4096`

Executed bucketed transfer experiment:

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
  --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --preprocess-num-workers 16 \
  --preprocess-batch-size 16 \
  --distributed-gpu-count 5 \
  --skip-validation-eval
```

Run result:

- run id: `whisper-turbo-accent-bucketed-transfer-1k-v1-20260423-204820`
- adapter: `/artifacts/whisper-turbo-accent-bucketed-transfer-1k-v1-20260423-204820/adapter`
- report: `/artifacts/whisper-turbo-accent-bucketed-transfer-1k-v1-20260423-204820/report.json`
- Modal GPU: `H100!:5`
- distributed world size: `5`
- preprocessing: `1024` rows with `num_proc=16`
- training runtime: `42.2915s`
- training steps: `13`
- train loss: `2.6227066333477316`
- Svarah eval samples: `6656`

Full Svarah metrics:

| Model | WER | CER |
| --- | ---: | ---: |
| Base `openai/whisper-large-v3-turbo` | `0.0813502569` | `0.0387159934` |
| Bucketed transfer LoRA | `0.0819083325` | `0.0388757214` |
| Delta, adapter minus base | `+0.0005580756` | `+0.0001597279` |

Read:

- the bucketed transfer run failed the stop rule
- restoring the old duration/context distribution was not enough to reproduce the old tiny base-beating result
- do not scale this recipe to `2048` or `4096`
- the next improvement needs better evidence that selected rows are genuinely useful Indian-accent supervision, not just longer Common Voice utterances

## Legacy full-run template

This was the original full-run template. Do not use it before the `1024` row bucketed transfer run beats base WER.

```bash
modal run tools/modal_whisper_lora_experiment.py \
  --mode train_eval \
  --experiment-name whisper-turbo-accent-curated-v1 \
  --train-dataset <CURATED_DATASET> \
  --train-split train \
  --train-audio-column <audio_col> \
  --train-text-column <text_col> \
  --train-min-duration-seconds 1 \
  --train-max-duration-seconds 6 \
  --train-max-samples 30000 \
  --validation-max-samples 2000 \
  --num-train-epochs 0.5 \
  --learning-rate 5e-6 \
  --rank 16 \
  --alpha 32 \
  --dropout 0.05 \
  --target-module-set attention \
  --normalize-transcripts \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4
```

## What not to do next

Do not spend more runs on:

- `TIE_shorts` mixing
- long-utterance-only CV slices
- broadening the same CV data without changing source quality

Those branches were already tested and did not beat base on the strict accent-only objective.
