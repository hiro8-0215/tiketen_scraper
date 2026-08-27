"""Upload master CSV files through an authenticated Google Apps Script endpoint."""
from __future__ import annotations

import base64
import glob
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


JST = timezone(timedelta(hours=9), "JST")
MAX_FILE_BYTES = 35 * 1024 * 1024


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def upload_file(webapp_url: str, token: str, path: Path, subfolder: str) -> None:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise RuntimeError(
            f"{path.name} is {size / 1024**2:.1f} MiB; split it below "
            f"{MAX_FILE_BYTES / 1024**2:.0f} MiB before Apps Script upload"
        )
    payload = {
        "token": token,
        "subfolderName": subfolder,
        "filename": path.name,
        "filedata": base64.b64encode(path.read_bytes()).decode("ascii"),
    }
    request = urllib.request.Request(
        webapp_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Upload failed for {path.name}: {error}") from error
    if body.get("status") != "success":
        raise RuntimeError(
            f"Google Apps Script rejected {path.name}: "
            f"{body.get('message', 'unknown error')}"
        )
    print(f"uploaded: {path.name} -> {body.get('fileId')}", flush=True)


def main() -> None:
    webapp_url = required_environment("GDRIVE_WEBAPP_URL")
    token = required_environment("GDRIVE_UPLOAD_TOKEN")
    source_dir = Path(os.environ.get("GDRIVE_SOURCE_DIR", "data"))
    now = datetime.now(JST)
    subfolder = f"data_{now.month}_{now.day}"
    paths = [Path(value) for value in sorted(glob.glob(str(source_dir / "*_master.csv")))]
    if not paths:
        raise RuntimeError(f"No *_master.csv files found under {source_dir.resolve()}")
    print(f"Uploading {len(paths)} master files to {subfolder}", flush=True)
    for path in paths:
        upload_file(webapp_url, token, path, subfolder)
    print("Google Drive backup completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
