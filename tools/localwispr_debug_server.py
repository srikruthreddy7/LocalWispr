#!/usr/bin/env python3
import html
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

DEBUG_ROOT = Path("/tmp/localwispr-debug-captures")
DEBUG_LOG = Path("/tmp/localwispr-debug.log")
HOST = "127.0.0.1"
PORT = 8765


def latest_session_dir() -> Path | None:
    if not DEBUG_ROOT.exists():
        return None
    sessions = sorted(
        [path for path in DEBUG_ROOT.iterdir() if path.is_dir() and path.name.startswith("session-")]
    )
    return sessions[-1] if sessions else None


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def read_latest_payload() -> dict:
    session_dir = latest_session_dir()
    payload: dict = {
        "session_dir": str(session_dir) if session_dir else None,
        "session": None,
        "raw_text": None,
        "cleaned_text": None,
        "error_text": None,
        "log_tail": "",
    }

    if DEBUG_LOG.exists():
        lines = DEBUG_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        payload["log_tail"] = "\n".join(lines[-120:])

    if session_dir is None:
        return payload

    session_json = session_dir / "session.json"
    if session_json.exists():
        payload["session"] = json.loads(session_json.read_text(encoding="utf-8"))

    payload["raw_text"] = read_text(session_dir / "raw.txt")
    payload["cleaned_text"] = read_text(session_dir / "cleaned.txt")
    payload["error_text"] = read_text(session_dir / "error.txt")
    return payload


def render_html(payload: dict) -> str:
    session = payload.get("session") or {}
    audio_path = session.get("audioPath")
    audio_url = f"/audio?path={html.escape(audio_path)}" if audio_path else None
    start_context = session.get("startContext") or {}
    hints = session.get("contextualHints") or []

    def pre_block(value: str | None) -> str:
        if not value:
            return "<p>None</p>"
        return f"<pre>{html.escape(value)}</pre>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="2">
  <title>LocalWispr Debug</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #111; color: #eee; margin: 24px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ background: #1b1b1b; border: 1px solid #333; border-radius: 12px; padding: 16px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0c0c0c; padding: 12px; border-radius: 8px; }}
    code {{ color: #9cdcfe; }}
    .muted {{ color: #aaa; }}
  </style>
</head>
<body>
  <h1>LocalWispr Debug</h1>
  <p class="muted">Auto-refreshing latest session inspector</p>
  <div class="grid">
    <div class="card">
      <h2>Latest Session</h2>
      <p><strong>Dir:</strong> <code>{html.escape(payload.get("session_dir") or "none")}</code></p>
      <p><strong>Audio:</strong> {"<audio controls src=\"" + audio_url + "\"></audio>" if audio_url else "none"}</p>
      <pre>{html.escape(json.dumps(session, indent=2, sort_keys=True))}</pre>
    </div>
    <div class="card">
      <h2>Start Context</h2>
      <pre>{html.escape(json.dumps(start_context, indent=2, sort_keys=True))}</pre>
      <h2>Contextual Hints</h2>
      <pre>{html.escape(json.dumps(hints, indent=2))}</pre>
    </div>
    <div class="card">
      <h2>Raw Text</h2>
      {pre_block(payload.get("raw_text"))}
      <h2>Cleaned Text</h2>
      {pre_block(payload.get("cleaned_text"))}
      <h2>Error</h2>
      {pre_block(payload.get("error_text"))}
    </div>
    <div class="card">
      <h2>Log Tail</h2>
      <pre>{html.escape(payload.get("log_tail") or "")}</pre>
    </div>
  </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/latest":
            payload = read_latest_payload()
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/audio":
            from urllib.parse import parse_qs

            qs = parse_qs(parsed.query)
            path = qs.get("path", [None])[0]
            if not path:
                self.send_error(404)
                return
            audio_path = Path(path)
            if not audio_path.exists():
                self.send_error(404)
                return
            data = audio_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        payload = read_latest_payload()
        data = render_html(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args):
        return


if __name__ == "__main__":
    HTTPServer((HOST, PORT), Handler).serve_forever()
