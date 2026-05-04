#!/usr/bin/env python3
"""Generate the Indian-accent English data pipeline registry and runbook.

This is intentionally a planning/orchestration helper. It does not download
datasets or launch training by itself. It writes the exact source registry and
Modal command plan that should be run once access is available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "accent-data-pipeline"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    name: str
    status: str
    priority: str
    role: str
    access: str
    scale: str
    fit: str
    risks: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    modal_dataset: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CommandSpec:
    command_id: str
    phase: str
    source_id: str
    parallel_group: str
    purpose: str
    command: str
    requires_access: bool = False
    blocking_for_training: bool = False


def build_sources() -> list[SourceSpec]:
    return [
        SourceSpec(
            source_id="indic_timit_modal",
            name="Indic-TIMIT in Modal volume",
            status="available",
            priority="tier_1",
            role="primary_candidate",
            access="Already downloaded, extracted, indexed, and split in Modal artifacts volume.",
            scale="135,836 rows across 7 language shards; speaker-separated train/validation split exists.",
            fit="High. Indian English read speech with L1 accent coverage and phonetically rich prompts.",
            risks=[
                "Read-speech distribution may not match Svarah perfectly.",
                "Prior broad Indic-TIMIT training hurt full Svarah unless kept narrow and encoder-only.",
            ],
            links=["https://spiredatasets.ee.iisc.ac.in/"],
            notes=[
                "Train manifest: /artifacts/datasets/indic-timit-v2-splits/train.jsonl",
                "Validation manifest: /artifacts/datasets/indic-timit-v2-splits/validation.jsonl",
                "Full index: /artifacts/datasets/indic-timit-v2-index/manifest.full.jsonl",
            ],
        ),
        SourceSpec(
            source_id="cv_ishands",
            name="ishands/commonvoice-indian_accent",
            status="open",
            priority="tier_1",
            role="open_pool",
            access="Open Hugging Face dataset.",
            scale="110,088 rows, about 163.9 hours, Common Voice v21 Indian-accent English slice.",
            fit="Medium-high. Large short-utterance pool with explicit Indian/South Asian accent metadata.",
            risks=[
                "Read-prompt Common Voice style may not transfer enough by itself.",
                "Accent label includes India, Pakistan, and Sri Lanka, so audit before using as pure Indian English.",
            ],
            links=["https://huggingface.co/datasets/ishands/commonvoice-indian_accent"],
            modal_dataset="ishands/commonvoice-indian_accent",
            notes=[
                "Best open pool for audit, hard mining, and speaker-diverse backfill.",
                "Use accent-only transcript filters for the next manifest, not formatting-heavy rows.",
            ],
        ),
        SourceSpec(
            source_id="cv_willheld",
            name="WillHeld/india_accent_cv",
            status="open",
            priority="tier_2",
            role="known_baseline_pool",
            access="Open Hugging Face dataset.",
            scale="91.5k train rows in the HF viewer.",
            fit="Medium. Proven as a source but no longer enough to beat current turbo baseline.",
            risks=[
                "Overlaps conceptually with the larger Common Voice pool.",
                "Previous full-Svarah sweeps no longer show a base-beating adapter.",
            ],
            links=["https://huggingface.co/datasets/WillHeld/india_accent_cv"],
            modal_dataset="WillHeld/india_accent_cv",
            notes=["Keep for dedupe comparisons and regression checks."],
        ),
        SourceSpec(
            source_id="vaani_transcription",
            name="ARTPARK-IISc/Vaani-transcription-part",
            status="access_confirmed",
            priority="tier_2",
            role="gated_supplement",
            access="Gated Hugging Face dataset. Modal huggingface-secret access confirmed for the English config.",
            scale="English config has 15,075 train rows, 1,787 validation rows, and 1,519 test rows in Vaani-transcription-part.",
            fit="Medium. India-representative, but English slices must be mined carefully.",
            risks=[
                "Many non-English configs exist; target the English config directly.",
                "Spontaneous/image-prompted style can shift away from target dictation speech.",
                "Markup/noise tags must be stripped.",
            ],
            links=["https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part"],
            modal_dataset="ARTPARK-IISc/Vaani-transcription-part",
            notes=[
                "Use metadata-only probe first, then target the English config directly.",
                "Existing Vaani six-district supplement did not improve Svarah, so treat as supplement only.",
            ],
        ),
        SourceSpec(
            source_id="vaani_full",
            name="ARTPARK-IISc/Vaani",
            status="gated",
            priority="tier_2",
            role="gated_survey_pool",
            access="Accept Hugging Face terms and confirm token access from Modal.",
            scale="About 31,255 hours raw audio, 156k speakers, 165 districts, 2,043 transcribed hours.",
            fit="Medium. Strong regional coverage, weak English concentration.",
            risks=[
                "Full corpus is huge and mostly not English.",
                "English filtering must happen before any training manifest is built.",
            ],
            links=["https://huggingface.co/datasets/ARTPARK-IISc/Vaani", "https://vaani.iisc.ac.in/"],
            modal_dataset="ARTPARK-IISc/Vaani",
            notes=["Use only when the transcription-part repo is insufficient."],
        ),
        SourceSpec(
            source_id="iitm_indian_english_asr",
            name="IITM / NLTM Indian English ASR Challenge data",
            status="licensed_request",
            priority="tier_1_request",
            role="primary_request_candidate",
            access="Register/request via IITM/NLTM/TDIL flow and use only under the research license.",
            scale="302 hours on TDIL product page; challenge pages describe 280h English train and 190.3h Indian-English in the later multilingual release.",
            fit="High. Indian English read, lecture, and challenge ASR data with transcripts.",
            risks=[
                "License and redistribution terms must be checked before using in training artifacts.",
                "Part of the data is lecture style, so split IITM read speech from NPTEL lecture speech if files allow.",
            ],
            links=[
                "https://sites.google.com/view/englishasrchallenge/home",
                "https://sites.google.com/view/indian-language-asrchallenge/home",
                "https://tdil-dc.in/nplt/index.php?product_id=2250&route=product%2Fproduct",
            ],
            notes=[
                "This is one of the most important gated/request sources to pursue.",
                "Once access exists, download directly into Modal using the external archive path.",
            ],
        ),
        SourceSpec(
            source_id="nptel2020",
            name="AI4Bharat NPTEL2020 Indian-English Speech Dataset",
            status="open_external",
            priority="tier_3",
            role="weak_auxiliary_or_pretraining",
            access="Open GitHub/Zenodo/torrent scripts, not a normal HF ASR dataset.",
            scale="15,700 hours, 6.25M chunks, 3-10s average chunk length.",
            fit="Low-medium for supervised LoRA; high scale but noisy lecture domain.",
            risks=[
                "Authors state they did not manually annotate it.",
                "Education/lecture style is far from short dictation.",
                "Very large download, so sample and audit before any use.",
            ],
            links=[
                "https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset",
                "https://zenodo.org/records/4590121",
            ],
            notes=["Use only after cleaner sources fail, or as weak auxiliary data with strict teacher filtering."],
        ),
        SourceSpec(
            source_id="tie_shorts",
            name="TIE_shorts / Technical Indian English subset",
            status="open",
            priority="tier_2",
            role="technical_lecture_pool",
            access="Open Hugging Face subset; full TIE corpus requires separate request.",
            scale="Public subset is roughly tens of hours; full Technical Indian English corpus is much larger and request-based.",
            fit="Medium. Indian English lecture/technical speech can help accent coverage but may overfit technical/lecture style.",
            risks=[
                "Domain is technical lecture speech, not dictation.",
                "License metadata must be verified before commercial use.",
                "Use source-heldout evaluation so lecture gains do not masquerade as general accent gains.",
            ],
            links=[
                "https://huggingface.co/datasets/raianand/TIE_shorts",
                "https://github.com/raianand1991/TIE",
            ],
            modal_dataset="raianand/TIE_shorts",
            notes=[
                "Use Normalised_Transcript as the text column.",
                "Useful metadata fields include Speaker_ID, Gender, Native_Region, and Discipline_Group.",
            ],
        ),
        SourceSpec(
            source_id="l2_arctic_hindi",
            name="L2-ARCTIC Hindi L1 speakers",
            status="gated_or_request",
            priority="tier_3",
            role="small_phonetic_probe",
            access="Use KoelLabs/L2Arctic if access is approved, or request/download from the official PSI Lab source.",
            scale="Full corpus is about 27 hours across 24 non-native English speakers; Hindi L1 subset is small.",
            fit="High quality but too small. Useful for phonetic probes, not primary training.",
            risks=[
                "CC BY-NC license limits commercial use.",
                "Hindi L1 only, not broad Indian accent coverage.",
            ],
            links=[
                "https://huggingface.co/datasets/KoelLabs/L2Arctic",
                "https://psi.engr.tamu.edu/l2-arctic-corpus/",
            ],
            modal_dataset="KoelLabs/L2Arctic",
        ),
        SourceSpec(
            source_id="skit_s2i",
            name="skit-ai/skit-s2i",
            status="open",
            priority="tier_3",
            role="telephony_side_pool",
            access="Open Hugging Face dataset under CC BY-NC 4.0.",
            scale="About 13.8 hours total, 11 Indian speakers, banking intent telephony utterances.",
            fit="Narrow but real Indian-English telephony.",
            risks=[
                "Tiny speaker pool.",
                "Intent template/domain bias.",
                "Non-commercial license.",
            ],
            links=["https://huggingface.co/datasets/skit-ai/skit-s2i"],
            modal_dataset="skit-ai/skit-s2i",
        ),
        SourceSpec(
            source_id="tannmayy_indian_asr",
            name="tannmayy14 Indian English ASR repos",
            status="gated_unknown",
            priority="request_probe",
            role="gated_probe",
            access="Request access on Hugging Face, then inspect columns and provenance.",
            scale="HF tags say 10k-100k rows for each gated repo.",
            fit="Unknown until access/provenance is checked.",
            risks=[
                "No useful public README visible.",
                "Could be repackaged Common Voice or synthetic/unknown provenance.",
            ],
            links=[
                "https://huggingface.co/datasets/tannmayy14/indian_accent_english_asr",
                "https://huggingface.co/datasets/tannmayy14/Quickserve_Indian_english_asr",
            ],
            notes=["Do not train until provenance, license, and duplicate overlap are audited."],
        ),
        SourceSpec(
            source_id="dataocean_indian_english_mobile",
            name="DataoceanAI Indian English Speech Recognition Corpus - Conversations Mobile",
            status="commercial_request",
            priority="paid_fallback",
            role="commercial_candidate",
            access="Contact vendor / purchase license; HF repo is metadata only.",
            scale="Vendor page says 224.4 hours, 204 speakers, 16 kHz mobile recordings.",
            fit="Potentially high if license allows model training.",
            risks=[
                "Commercial licensing and reproducibility constraints.",
                "Need sample audit before buying or training.",
            ],
            links=[
                "https://huggingface.co/datasets/DataoceanAI/Indian_English_Speech_Recognition_Corpus_Conversations",
                "https://dataoceanai.com/datasets/asr/indian-english-speech-recognition-corpus-conversations-mobile/",
            ],
        ),
        SourceSpec(
            source_id="accentdatasets_com",
            name="AccentDatasets Indian to US/UK English collection",
            status="commercial_early_partner",
            priority="paid_probe",
            role="custom_collection_candidate",
            access="Contact provider with desired ASR training terms and transcript requirements.",
            scale="Custom/early partner collection; public page does not list fixed hours.",
            fit="Potentially high if we can specify Indian-accent ASR labels and license terms.",
            risks=[
                "Not immediately available.",
                "May be accent correction oriented rather than ASR transcription oriented.",
            ],
            links=["https://accentdatasets.com/"],
        ),
        SourceSpec(
            source_id="reject_indicaccentdb",
            name="DarshanaS/IndicAccentDb",
            status="reject_for_asr_training",
            priority="reject",
            role="not_training_data",
            access="Open/gated status is not the blocker.",
            scale="8k-ish rows; accent classification style.",
            fit="Low for ASR because the HF table does not expose usable transcripts.",
            risks=["No transcript column for direct ASR LoRA training."],
            links=["https://huggingface.co/datasets/DarshanaS/IndicAccentDb"],
        ),
        SourceSpec(
            source_id="reject_humyn_indic_asr",
            name="humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR",
            status="reject_for_english_accent",
            priority="reject",
            role="not_target_language",
            access="Open HF dataset.",
            scale="Small multilingual Indic-language ASR dataset.",
            fit="Low for this project because it is not Indian-accent English.",
            risks=["Training on non-English Indic speech targets the wrong objective."],
            links=["https://huggingface.co/datasets/humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR"],
        ),
    ]


def build_commands() -> list[CommandSpec]:
    return [
        CommandSpec(
            command_id="indic_timit_verify_train",
            phase="verify_available",
            source_id="indic_timit_modal",
            parallel_group="audit_A",
            purpose="Verify a quick sample of the existing Indic-TIMIT train manifest audio paths in Modal.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode verify_audio_manifest \\
  --audio-verify-name indic-timit-train-verify-quick-v1 \\
  --train-dataset /artifacts/datasets/indic-timit-v2-splits/train.jsonl \\
  --train-audio-column audio_path \\
  --train-text-column transcript \\
  --profile-sample-limit 200""",
            blocking_for_training=True,
        ),
        CommandSpec(
            command_id="indic_timit_verify_validation",
            phase="verify_available",
            source_id="indic_timit_modal",
            parallel_group="audit_A",
            purpose="Verify a quick sample of the existing Indic-TIMIT validation manifest audio paths in Modal.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode verify_audio_manifest \\
  --audio-verify-name indic-timit-validation-verify-quick-v1 \\
  --train-dataset /artifacts/datasets/indic-timit-v2-splits/validation.jsonl \\
  --train-audio-column audio_path \\
  --train-text-column transcript \\
  --profile-sample-limit 200""",
            blocking_for_training=True,
        ),
        CommandSpec(
            command_id="cv_ishands_inspect",
            phase="inspect_open_pool",
            source_id="cv_ishands",
            parallel_group="audit_A",
            purpose="Confirm columns and row count for the open Common Voice Indian-accent pool.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode inspect_train \\
  --train-dataset ishands/commonvoice-indian_accent \\
  --train-split train""",
        ),
        CommandSpec(
            command_id="cv_ishands_audit",
            phase="audit_open_pool",
            source_id="cv_ishands",
            parallel_group="audit_A",
            purpose="Build a 500-row audio audit bundle for the strict Common Voice Indian/South Asian accent label.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode build_audit_manifest \\
  --audit-name cv-indian-accent-audit-v2 \\
  --train-dataset ishands/commonvoice-indian_accent \\
  --train-split train \\
  --train-audio-column audio \\
  --train-text-column sentence \\
  --train-min-duration-seconds 1.5 \\
  --train-max-duration-seconds 8 \\
  --train-require-text \\
  --train-metadata-filters "accents=India and South Asia (India, Pakistan, Sri Lanka)" \\
  --audit-sample-limit 500 \\
  --audit-group-fields gender,age,accents \\
  --audit-max-samples-per-group 0 \\
  --audit-export-audio \\
  --audit-normalize-transcripts""",
            blocking_for_training=True,
        ),
        CommandSpec(
            command_id="cv_ishands_accent_only_manifest",
            phase="build_manifest_after_audit",
            source_id="cv_ishands",
            parallel_group="manifest_B",
            purpose="Build a Common Voice training manifest that rejects formatting-heavy text and caps speakers.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode build_training_manifest \\
  --manifest-name cv-indian-accent-accent-only-v2 \\
  --train-dataset ishands/commonvoice-indian_accent \\
  --train-split train \\
  --train-audio-column audio \\
  --train-text-column sentence \\
  --train-min-duration-seconds 1.5 \\
  --train-max-duration-seconds 8 \\
  --train-require-text \\
  --train-metadata-filters "accents=India and South Asia (India, Pakistan, Sri Lanka)" \\
  --manifest-sample-limit 50000 \\
  --manifest-output-limit 8192 \\
  --manifest-quality-preset accent_only \\
  --manifest-selection-strategy score \\
  --manifest-max-samples-per-speaker 8 \\
  --manifest-export-audio \\
  --manifest-normalize-transcripts""",
            blocking_for_training=True,
        ),
        CommandSpec(
            command_id="cv_ishands_hard_mine",
            phase="build_manifest_after_audit",
            source_id="cv_ishands",
            parallel_group="manifest_B",
            purpose="Mine rows where turbo fails but large-v3 agrees with the transcript.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode build_hard_example_manifest \\
  --manifest-name hard-mine-cv-indian-largev3-v2 \\
  --train-dataset ishands/commonvoice-indian_accent \\
  --train-split train \\
  --train-audio-column audio \\
  --train-text-column sentence \\
  --manifest-sample-limit 50000 \\
  --manifest-output-limit 4096 \\
  --manifest-max-samples-per-speaker 8 \\
  --hard-min-word-count 4 \\
  --hard-max-word-count 16 \\
  --hard-min-duration-seconds 1.5 \\
  --hard-max-duration-seconds 8.0 \\
  --hard-turbo-min-cer 0.04 \\
  --hard-turbo-max-cer 0.35 \\
  --hard-teacher-max-cer 0.02 \\
  --hard-min-selection-score 0.03 \\
  --hard-reject-format-sensitive \\
  --hard-dedupe-text \\
  --per-device-eval-batch-size 8 \\
  --distributed-gpu-count 5""",
            blocking_for_training=True,
        ),
        CommandSpec(
            command_id="tie_shorts_audit_quick",
            phase="audit_open_pool",
            source_id="tie_shorts",
            parallel_group="audit_A",
            purpose="Audit a bounded Technical Indian English subset before deciding whether to mine lecture-style speech.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode build_audit_manifest \\
  --audit-name tie-shorts-audit-quick-v1 \\
  --train-dataset raianand/TIE_shorts \\
  --train-split 'train[:2000]' \\
  --train-audio-column audio \\
  --train-text-column Normalised_Transcript \\
  --train-min-duration-seconds 1.5 \\
  --train-max-duration-seconds 10 \\
  --train-require-text \\
  --audit-sample-limit 500 \\
  --audit-group-fields Speaker_ID,Gender,Native_Region,Discipline_Group \\
  --audit-max-samples-per-group 0 \\
  --audit-export-audio \\
  --audit-normalize-transcripts""",
        ),
        CommandSpec(
            command_id="vaani_transcription_metadata_probe",
            phase="inspect_gated",
            source_id="vaani_transcription",
            parallel_group="gated_A",
            purpose="Metadata-only probe for Vaani transcription English shards before loading audio parquet.",
            command="""modal run tools/modal_hf_dataset_probe.py \\
  --dataset ARTPARK-IISc/Vaani-transcription-part \\
  --pattern "^audio/English/" \\
  --limit 20""",
            requires_access=True,
        ),
        CommandSpec(
            command_id="vaani_transcription_english_audit",
            phase="audit_gated",
            source_id="vaani_transcription",
            parallel_group="audit_A",
            purpose="Audit Vaani English rows with transcript normalization and audio export.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode build_audit_manifest \\
  --audit-name vaani-transcription-english-audit-v2 \\
  --train-dataset ARTPARK-IISc/Vaani-transcription-part \\
  --train-config-name English \\
  --train-split train \\
  --train-audio-column audio \\
  --train-text-column transcript \\
  --train-require-text \\
  --audit-sample-limit 500 \\
  --audit-group-fields gender,state,district \\
  --audit-max-samples-per-group 0 \\
  --audit-export-audio \\
  --audit-normalize-transcripts""",
            requires_access=True,
        ),
        CommandSpec(
            command_id="vaani_transcription_english_manifest",
            phase="build_manifest_after_audit",
            source_id="vaani_transcription",
            parallel_group="manifest_B",
            purpose="Build the Vaani English normalized accent supplement manifest after audit pass.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode build_training_manifest \\
  --manifest-name vaani-transcription-english-accent-only-v2 \\
  --train-dataset ARTPARK-IISc/Vaani-transcription-part \\
  --train-config-name English \\
  --train-split train \\
  --train-audio-column audio \\
  --train-text-column transcript \\
  --train-min-duration-seconds 1 \\
  --train-max-duration-seconds 8 \\
  --train-require-text \\
  --manifest-sample-limit 15075 \\
  --manifest-output-limit 4096 \\
  --manifest-quality-preset accent_only \\
  --manifest-selection-strategy score \\
  --manifest-max-samples-per-speaker 8 \\
  --manifest-export-audio \\
  --manifest-normalize-transcripts""",
            requires_access=True,
            blocking_for_training=True,
        ),
        CommandSpec(
            command_id="vaani_full_survey",
            phase="inspect_gated",
            source_id="vaani_full",
            parallel_group="gated_A",
            purpose="Fallback survey on full Vaani if transcription-part is missing required configs.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode survey_train_configs \\
  --survey-name vaani-full-english-sweep-v2 \\
  --survey-max-workers 8 \\
  --train-dataset ARTPARK-IISc/Vaani \\
  --train-min-duration-seconds 1 \\
  --train-max-duration-seconds 6 \\
  --train-require-text \\
  --train-metadata-filters language=English,isTranscriptionAvailable=Yes""",
            requires_access=True,
        ),
        CommandSpec(
            command_id="l2_arctic_inspect",
            phase="inspect_gated",
            source_id="l2_arctic_hindi",
            parallel_group="gated_A",
            purpose="Inspect L2-ARCTIC columns and splits after access. Only Hindi L1 rows are useful here.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode inspect_train \\
  --train-dataset KoelLabs/L2Arctic""",
            requires_access=True,
        ),
        CommandSpec(
            command_id="skit_s2i_audit",
            phase="audit_side_pool",
            source_id="skit_s2i",
            parallel_group="audit_A",
            purpose="Audit the small Indian-English telephony side pool.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode build_audit_manifest \\
  --audit-name skit-s2i-telephony-audit-v1 \\
  --train-dataset skit-ai/skit-s2i \\
  --train-split train \\
  --train-text-column template \\
  --train-min-duration-seconds 1 \\
  --train-max-duration-seconds 8 \\
  --train-require-text \\
  --audit-sample-limit 500 \\
  --audit-group-fields gender,native_language,languages \\
  --audit-max-samples-per-group 50 \\
  --audit-export-audio \\
  --audit-normalize-transcripts""",
        ),
        CommandSpec(
            command_id="tannmayy_probe_one",
            phase="inspect_gated",
            source_id="tannmayy_indian_asr",
            parallel_group="gated_A",
            purpose="Probe gated Indian English ASR repo after access is granted.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode inspect_train \\
  --train-dataset tannmayy14/indian_accent_english_asr""",
            requires_access=True,
        ),
        CommandSpec(
            command_id="tannmayy_probe_two",
            phase="inspect_gated",
            source_id="tannmayy_indian_asr",
            parallel_group="gated_A",
            purpose="Probe second gated Indian English ASR repo after access is granted.",
            command="""modal run tools/modal_whisper_lora_experiment.py \\
  --mode inspect_train \\
  --train-dataset tannmayy14/Quickserve_Indian_english_asr""",
            requires_access=True,
        ),
    ]


def _markdown_list(values: Iterable[str]) -> str:
    values = list(values)
    if not values:
        return "- none"
    return "\n".join(f"- {value}" for value in values)


def render_sources_markdown(sources: list[SourceSpec]) -> str:
    lines = [
        "# Indian-Accent English Data Source Registry",
        "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "This registry is focused on one objective: data that can improve Indian-accent English ASR.",
        "Formatting, punctuation, and numeric normalization are not the selection goal here.",
        "",
    ]
    for source in sources:
        lines.extend(
            [
                f"## {source.source_id}: {source.name}",
                "",
                f"- status: `{source.status}`",
                f"- priority: `{source.priority}`",
                f"- role: `{source.role}`",
                f"- access: {source.access}",
                f"- scale: {source.scale}",
                f"- fit: {source.fit}",
                "",
                "Risks:",
                _markdown_list(source.risks),
                "",
                "Links:",
                _markdown_list(source.links),
            ]
        )
        if source.modal_dataset:
            lines.append(f"- modal dataset id: `{source.modal_dataset}`")
        if source.notes:
            lines.extend(["", "Notes:", _markdown_list(source.notes)])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_commands_markdown(commands: list[CommandSpec]) -> str:
    lines = [
        "# Accent Data Pipeline Commands",
        "",
        "Run commands in the same parallel group concurrently when Modal quota allows.",
        "Do not run `build_manifest_after_audit` commands until the matching audit has passed.",
        "",
        "Parallel groups:",
        "- `audit_A`: independent verification/audit work; safe to run together.",
        "- `gated_A`: access probes and gated surveys; safe to run together after access is granted.",
        "- `manifest_B`: manifest builds after audit; safe to run per source, but these are expensive.",
        "",
    ]
    for command in commands:
        lines.extend(
            [
                f"## {command.command_id}",
                "",
                f"- phase: `{command.phase}`",
                f"- source: `{command.source_id}`",
                f"- parallel group: `{command.parallel_group}`",
                f"- requires gated access: `{str(command.requires_access).lower()}`",
                f"- blocks training: `{str(command.blocking_for_training).lower()}`",
                f"- purpose: {command.purpose}",
                "",
                "```bash",
                command.command,
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_shell_commands(commands: list[CommandSpec]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# This file is generated by tools/accent_data_pipeline.py.",
        "# Run independent commands in the same parallel group concurrently when quota allows.",
        "",
    ]
    for command in commands:
        lines.extend(
            [
                f"# {command.command_id}",
                f"# phase={command.phase} source={command.source_id} parallel_group={command.parallel_group}",
                f"# purpose: {command.purpose}",
                command.command,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(output_dir: Path, *, sources: list[SourceSpec], commands: list[CommandSpec]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_json_path = output_dir / "sources.json"
    command_json_path = output_dir / "commands.json"
    source_markdown_path = output_dir / "sources.md"
    command_markdown_path = output_dir / "commands.md"
    shell_path = output_dir / "commands.sh"

    source_json_path.write_text(
        json.dumps([asdict(source) for source in sources], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command_json_path.write_text(
        json.dumps([asdict(command) for command in commands], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_markdown_path.write_text(render_sources_markdown(sources), encoding="utf-8")
    command_markdown_path.write_text(render_commands_markdown(commands), encoding="utf-8")
    shell_path.write_text(render_shell_commands(commands), encoding="utf-8")

    return {
        "sources_json": str(source_json_path),
        "commands_json": str(command_json_path),
        "sources_markdown": str(source_markdown_path),
        "commands_markdown": str(command_markdown_path),
        "commands_shell": str(shell_path),
    }


def check_repo_wiring() -> dict[str, object]:
    required_paths = [
        REPO_ROOT / "tools" / "modal_whisper_lora_experiment.py",
        REPO_ROOT / "tools" / "modal_indic_timit_volume_tools.py",
        REPO_ROOT / "tools" / "modal_volume_downloader.py",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    return {
        "repo_root": str(REPO_ROOT),
        "missing_required_paths": missing,
        "ok": not missing,
        "available_sources": len(build_sources()),
        "planned_commands": len(build_commands()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated registry/runbook files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check local repo wiring and print JSON.",
    )
    parser.add_argument(
        "--print",
        choices=("sources", "commands", "all"),
        default="",
        help="Print generated markdown to stdout instead of writing files.",
    )
    args = parser.parse_args()

    if args.check:
        print(json.dumps(check_repo_wiring(), indent=2, sort_keys=True))
        return

    sources = build_sources()
    commands = build_commands()
    if args.print:
        if args.print in {"sources", "all"}:
            print(render_sources_markdown(sources))
        if args.print in {"commands", "all"}:
            print(render_commands_markdown(commands))
        return

    outputs = write_outputs(args.output_dir, sources=sources, commands=commands)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
