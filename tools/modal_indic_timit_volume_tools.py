from __future__ import annotations

import json
import os
import tarfile
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LOCALWISPR_MODAL_INDIC_TIMIT_APP_NAME", "localwispr-indic-timit-tools")
VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_DOWNLOAD_VOLUME", "localwispr-whisper-lora-artifacts"
)
VOLUME_DIR = Path("/volume")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=modal.Image.debian_slim(python_version="3.11"))

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".sph", ".m4a"}
TEXT_SUFFIXES = {".txt", ".wrd", ".phn", ".lab", ".trans", ".tsv", ".csv", ".json"}


@dataclass
class ExtractConfig:
    archive_path: str
    extract_root: str = "datasets/indic-timit-v2-extracted"
    overwrite: bool = False


@dataclass
class ProfileConfig:
    path: str
    sample_limit: int = 10


@dataclass
class ConvertConfig:
    input_root: str
    output_path: str = "datasets/indic-timit-v2-index/manifest.jsonl"


@dataclass
class SplitConfig:
    input_manifest: str
    output_root: str = "datasets/indic-timit-v2-splits"
    validation_fraction: float = 0.1
    seed: int = 42


def _resolve_volume_path(value: str) -> Path:
    normalized = value.strip().lstrip("/")
    return VOLUME_DIR / normalized


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _is_text_like_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return True
    return not suffix and path.name.lower().startswith("transcripts")


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> list[str]:
    extracted_paths: list[str] = []
    destination_resolved = destination.resolve()
    for member in archive.getmembers():
        member_path = destination / member.name
        resolved = member_path.resolve()
        if not str(resolved).startswith(str(destination_resolved)):
            raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
        extracted_paths.append(str(member_path))
    archive.extractall(destination)
    return extracted_paths


def _extract_archive_impl(config: ExtractConfig) -> dict[str, Any]:
    archive_path = _resolve_volume_path(config.archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    extract_root = _resolve_volume_path(config.extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    stem = archive_path.name
    if stem.endswith(".tar.gz"):
        stem = stem[: -len(".tar.gz")]
    target_dir = extract_root / stem

    if target_dir.exists() and any(target_dir.iterdir()):
        if not config.overwrite:
            return {
                "status": "skipped",
                "archive_path": str(archive_path),
                "extract_dir": str(target_dir),
                "reason": "existing_nonempty_directory",
            }
        for child in sorted(target_dir.iterdir(), reverse=True):
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        extracted_paths = _safe_extract(archive, target_dir)

    top_level_entries = sorted(
        {
            Path(member.name).parts[0]
            for member in members
            if member.name and not member.name.startswith("./")
        }
    )
    report = {
        "status": "extracted",
        "archive_path": str(archive_path),
        "extract_dir": str(target_dir),
        "member_count": len(members),
        "top_level_entries": top_level_entries,
        "sample_members": [member.name for member in members[:20]],
        "sample_extracted_paths": extracted_paths[:20],
    }
    _write_json(target_dir / "_extract_report.json", report)
    volume.commit()
    return report


def _profile_tree_impl(config: ProfileConfig) -> dict[str, Any]:
    root = _resolve_volume_path(config.path)
    if not root.exists():
        raise FileNotFoundError(f"Profile path not found: {root}")

    total_files = 0
    total_dirs = 0
    total_bytes = 0
    suffix_counter: Counter[str] = Counter()
    audio_files: list[str] = []
    text_files: list[str] = []
    files_by_stem: defaultdict[str, set[str]] = defaultdict(set)
    sample_texts: list[dict[str, str]] = []
    top_level_entries: list[str] = []
    immediate_dirs: list[str] = []

    if root.is_dir():
        top_level_entries = sorted(item.name for item in root.iterdir())[:50]
        immediate_dirs = sorted(item.name for item in root.iterdir() if item.is_dir())

    for path in root.rglob("*"):
        if path.is_dir():
            total_dirs += 1
            continue
        total_files += 1
        stat = path.stat()
        total_bytes += stat.st_size
        suffix = path.suffix.lower()
        suffix_counter[suffix] += 1
        relative = str(path.relative_to(root))
        if suffix in AUDIO_SUFFIXES:
            audio_files.append(relative)
        if _is_text_like_file(path):
            text_files.append(relative)
            if len(sample_texts) < config.sample_limit and stat.st_size <= 8192:
                try:
                    snippet = path.read_text(encoding="utf-8", errors="replace").strip()
                    sample_texts.append(
                        {
                            "path": relative,
                            "snippet": snippet[:500],
                        }
                    )
                except Exception:
                    pass
        files_by_stem[str(path.with_suffix("").relative_to(root))].add(suffix)

    paired_stems = {
        stem: sorted(list(suffixes))
        for stem, suffixes in files_by_stem.items()
        if suffixes & AUDIO_SUFFIXES and suffixes & TEXT_SUFFIXES
    }

    report = {
        "root": str(root),
        "total_files": total_files,
        "total_dirs": total_dirs,
        "total_bytes": total_bytes,
        "top_level_entries": top_level_entries,
        "speaker_dir_count": len(immediate_dirs),
        "speaker_dirs": immediate_dirs[:100],
        "suffix_counts": dict(sorted(suffix_counter.items())),
        "audio_file_count": len(audio_files),
        "text_file_count": len(text_files),
        "sample_audio_files": audio_files[: config.sample_limit],
        "sample_text_files": text_files[: config.sample_limit],
        "sample_text_snippets": sample_texts,
        "paired_audio_text_stem_count": len(paired_stems),
        "sample_paired_stems": dict(list(sorted(paired_stems.items()))[: config.sample_limit]),
    }
    _write_json(root / "_profile_report.json", report)
    volume.commit()
    return report


def _find_shard_root(extracted_root: Path) -> Path:
    transcript_candidates = sorted(extracted_root.rglob("Transcripts*"))
    if not transcript_candidates:
        raise FileNotFoundError(f"No transcript index found under {extracted_root}")
    return transcript_candidates[0].parent


def _parse_transcript_index(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        utterance_id, transcript = parts
        transcript = transcript.strip()
        if transcript:
            mapping[utterance_id] = transcript
    return mapping


def _convert_tree_impl(config: ConvertConfig) -> dict[str, Any]:
    input_root = _resolve_volume_path(config.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Conversion root not found: {input_root}")

    extracted_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    per_shard_counts: dict[str, int] = {}
    skipped_dirs: dict[str, str] = {}

    for extracted_dir in extracted_dirs:
        try:
            shard_root = _find_shard_root(extracted_dir)
        except FileNotFoundError:
            skipped_dirs[extracted_dir.name] = "missing_transcript_index"
            continue
        transcript_indexes = sorted(shard_root.glob("Transcripts*"))
        if not transcript_indexes:
            skipped_dirs[extracted_dir.name] = "no_transcript_files"
            continue
        transcript_map = _parse_transcript_index(transcript_indexes[0])
        shard_name = extracted_dir.name
        shard_row_count = 0

        for audio_path in sorted(shard_root.rglob("*.wav")):
            utterance_id = audio_path.stem
            transcript = transcript_map.get(utterance_id)
            if not transcript:
                continue
            speaker_id = audio_path.parent.name
            rows.append(
                {
                    "audio_path": str(audio_path.relative_to(VOLUME_DIR)),
                    "transcript": transcript,
                    "speaker_id": speaker_id,
                    "language_shard": shard_name,
                    "utterance_id": utterance_id,
                }
            )
            shard_row_count += 1

        per_shard_counts[shard_name] = shard_row_count

    output_path = _resolve_volume_path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_root": str(input_root),
        "output_path": str(output_path),
        "row_count": len(rows),
        "per_shard_counts": per_shard_counts,
        "skipped_dirs": skipped_dirs,
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    volume.commit()
    return summary


def _split_manifest_impl(config: SplitConfig) -> dict[str, Any]:
    input_manifest = _resolve_volume_path(config.input_manifest)
    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    rows: list[dict[str, Any]] = []
    by_shard_speakers: defaultdict[str, set[str]] = defaultdict(set)
    with input_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(row)
            by_shard_speakers[row["language_shard"]].add(row["speaker_id"])

    validation_speakers: dict[str, set[str]] = {}
    randomizer = random.Random(config.seed)
    for shard, speakers in sorted(by_shard_speakers.items()):
        ordered = sorted(speakers)
        if len(ordered) <= 1:
            validation_speakers[shard] = set()
            continue
        desired = max(1, round(len(ordered) * config.validation_fraction))
        desired = min(desired, len(ordered) - 1)
        shuffled = ordered[:]
        randomizer.shuffle(shuffled)
        validation_speakers[shard] = set(shuffled[:desired])

    output_root = _resolve_volume_path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train.jsonl"
    validation_path = output_root / "validation.jsonl"

    per_shard_counts = {
        "train": Counter(),
        "validation": Counter(),
    }
    with train_path.open("w", encoding="utf-8") as train_handle, validation_path.open(
        "w", encoding="utf-8"
    ) as validation_handle:
        for row in rows:
            output_row = dict(row)
            audio_path = str(output_row.get("audio_path", ""))
            if audio_path and not audio_path.startswith("/"):
                output_row["audio_path"] = str(Path("/artifacts") / audio_path)
            target = (
                "validation"
                if output_row["speaker_id"] in validation_speakers[output_row["language_shard"]]
                else "train"
            )
            if target == "validation":
                validation_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            else:
                train_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            per_shard_counts[target][output_row["language_shard"]] += 1

    summary = {
        "input_manifest": str(input_manifest),
        "output_root": str(output_root),
        "validation_fraction": config.validation_fraction,
        "seed": config.seed,
        "speaker_counts_by_shard": {key: len(value) for key, value in sorted(by_shard_speakers.items())},
        "validation_speakers_by_shard": {
            key: sorted(value) for key, value in sorted(validation_speakers.items())
        },
        "row_counts": {
            "train": dict(per_shard_counts["train"]),
            "validation": dict(per_shard_counts["validation"]),
        },
    }
    _write_json(output_root / "split.summary.json", summary)
    volume.commit()
    return summary


@app.function(timeout=60 * 60 * 4, volumes={str(VOLUME_DIR): volume})
def extract_archive_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    return _extract_archive_impl(ExtractConfig(**config_payload))


@app.function(timeout=60 * 30, volumes={str(VOLUME_DIR): volume})
def profile_tree_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    return _profile_tree_impl(ProfileConfig(**config_payload))


@app.function(timeout=60 * 60, volumes={str(VOLUME_DIR): volume})
def convert_tree_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    return _convert_tree_impl(ConvertConfig(**config_payload))


@app.function(timeout=60 * 30, volumes={str(VOLUME_DIR): volume})
def split_manifest_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    return _split_manifest_impl(SplitConfig(**config_payload))


@app.local_entrypoint()
def main(
    mode: str = "extract",
    archive_path: str = "",
    archive_manifest: str = "",
    extract_root: str = "datasets/indic-timit-v2-extracted",
    profile_path: str = "",
    profile_paths: str = "",
    overwrite: bool = False,
    sample_limit: int = 10,
    convert_input_root: str = "",
    convert_output_path: str = "datasets/indic-timit-v2-index/manifest.jsonl",
    split_input_manifest: str = "",
    split_output_root: str = "datasets/indic-timit-v2-splits",
    split_validation_fraction: float = 0.1,
    split_seed: int = 42,
) -> None:
    if mode == "extract":
        if not archive_path:
            raise ValueError("archive_path is required for extract mode")
        result = extract_archive_remote.remote(
            asdict(
                ExtractConfig(
                    archive_path=archive_path,
                    extract_root=extract_root,
                    overwrite=overwrite,
                )
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if mode == "extract_many":
        if not archive_manifest:
            raise ValueError("archive_manifest is required for extract_many mode")
        manifest_path = Path(archive_manifest).expanduser()
        archive_paths = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(archive_paths, list) or not archive_paths:
            raise ValueError("archive_manifest must be a non-empty JSON array of archive paths")
        payloads = [
            asdict(
                ExtractConfig(
                    archive_path=str(item),
                    extract_root=extract_root,
                    overwrite=overwrite,
                )
            )
            for item in archive_paths
        ]
        results = list(extract_archive_remote.map(payloads))
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if mode == "profile":
        if not profile_path:
            raise ValueError("profile_path is required for profile mode")
        result = profile_tree_remote.remote(
            asdict(ProfileConfig(path=profile_path, sample_limit=sample_limit))
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if mode == "profile_many":
        candidates = [value.strip() for value in profile_paths.split(",") if value.strip()]
        if not candidates:
            raise ValueError("profile_paths is required for profile_many mode")
        results = list(
            profile_tree_remote.map(
                [asdict(ProfileConfig(path=value, sample_limit=sample_limit)) for value in candidates]
            )
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if mode == "convert":
        if not convert_input_root:
            raise ValueError("convert_input_root is required for convert mode")
        result = convert_tree_remote.remote(
            asdict(
                ConvertConfig(
                    input_root=convert_input_root,
                    output_path=convert_output_path,
                )
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if mode == "split":
        if not split_input_manifest:
            raise ValueError("split_input_manifest is required for split mode")
        result = split_manifest_remote.remote(
            asdict(
                SplitConfig(
                    input_manifest=split_input_manifest,
                    output_root=split_output_root,
                    validation_fraction=split_validation_fraction,
                    seed=split_seed,
                )
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    raise ValueError(
        "Unsupported mode. Expected one of: extract, extract_many, profile, profile_many, convert, split"
    )
