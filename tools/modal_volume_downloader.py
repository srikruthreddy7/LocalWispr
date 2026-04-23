from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LOCALWISPR_MODAL_DOWNLOAD_APP_NAME", "localwispr-volume-downloader")
VOLUME_NAME = os.environ.get(
    "LOCALWISPR_MODAL_DOWNLOAD_VOLUME", "localwispr-whisper-lora-artifacts"
)
VOLUME_DIR = Path("/volume")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").apt_install("wget")
app = modal.App(APP_NAME, image=image)


@dataclass
class ArchiveDownload:
    filename: str
    url: str
    target_dir: str
    overwrite: bool = False
    timeout_seconds: int = 120


def _sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "-", value.strip())
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    return cleaned.strip() or "downloads"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _download_archive_impl(config: ArchiveDownload) -> dict[str, Any]:
    print(f"[download] starting {config.filename}")
    request = urllib.request.Request(
        config.url,
        headers={"User-Agent": "LocalWispr Modal Volume Downloader/1.0"},
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        content_length_header = response.headers.get("Content-Length")
        expected_bytes = int(content_length_header) if content_length_header else None

    target_dir = VOLUME_DIR / _sanitize_component(config.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / config.filename
    temporary_destination = target_dir / f"{config.filename}.part"

    if (
        destination.exists()
        and not config.overwrite
        and expected_bytes is not None
        and destination.stat().st_size == expected_bytes
    ):
        result = {
            "filename": config.filename,
            "artifact_path": str(destination),
            "bytes_written": destination.stat().st_size,
            "content_length_bytes": expected_bytes,
            "status": "skipped",
        }
        print(f"[download] skipping {config.filename}; existing file matches expected size")
        volume.commit()
        return result

    if destination.exists():
        destination.unlink()
    if temporary_destination.exists():
        temporary_destination.unlink()

    command = [
        "wget",
        "--tries=4",
        f"--timeout={config.timeout_seconds}",
        "--waitretry=5",
        "-O",
        str(temporary_destination),
        config.url,
    ]
    subprocess.run(command, check=True)

    bytes_written = temporary_destination.stat().st_size
    if expected_bytes is not None and bytes_written != expected_bytes:
        temporary_destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download for {config.filename} was truncated: expected {expected_bytes} bytes, got {bytes_written}"
        )

    temporary_destination.replace(destination)
    result = {
        "filename": config.filename,
        "artifact_path": str(destination),
        "bytes_written": bytes_written,
        "content_length_bytes": expected_bytes,
        "status": "downloaded",
    }
    _write_json(target_dir / f"{config.filename}.download.json", result)
    print(f"[download] completed {config.filename} ({bytes_written} bytes)")
    volume.commit()
    return result


@app.function(timeout=60 * 60 * 8, volumes={str(VOLUME_DIR): volume})
def download_archive_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
    return _download_archive_impl(ArchiveDownload(**config_payload))


@app.local_entrypoint()
def main(
    manifest: str,
    target_dir: str = "datasets/indic-timit-v2",
    overwrite: bool = False,
    timeout_seconds: int = 120,
) -> None:
    manifest_path = Path(manifest).expanduser()
    archive_specs = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(archive_specs, list) or not archive_specs:
        raise ValueError("manifest must be a non-empty JSON array")

    payloads = [
        asdict(
            ArchiveDownload(
                filename=str(item["filename"]),
                url=str(item["url"]),
                target_dir=target_dir,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
            )
        )
        for item in archive_specs
    ]
    results = list(download_archive_remote.map(payloads))
    print(json.dumps(results, indent=2, sort_keys=True))
