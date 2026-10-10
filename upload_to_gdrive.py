"""Upload master CSV files through an authenticated Google Apps Script endpoint."""
from __future__ import annotations

import base64
import glob
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ticket_change_log import change_log_paths, migrate_legacy_logs
from drive_backup_parts import prepare_uploads


JST = timezone(timedelta(hours=9), "JST")
MAX_FILE_BYTES = 35 * 1024 * 1024
MAX_UPLOAD_ATTEMPTS = 5
UPLOAD_BUDGET_SECONDS = 15 * 60
RECOVERY_DELAYS = (60, 120)
RETRYABLE_HTTP_CODES = {404, 408, 429, 500, 502, 503, 504}


class TransientUploadError(RuntimeError):
    """Transport/service failure that can be retried after other files finish."""


def wait_for_retry(delay: float, deadline: float | None) -> None:
    if deadline is not None and time.monotonic() + delay >= deadline:
        raise TransientUploadError('Google Drive upload time budget exhausted')
    time.sleep(delay)


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def upload_file(webapp_url: str, token: str, path: Path, subfolder: str,
                deadline: float | None = None) -> None:
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
    body = None
    for attempt in range(1, MAX_UPLOAD_ATTEMPTS + 1):
        remaining = 120 if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            raise TransientUploadError('Google Drive upload time budget exhausted')
        try:
            with urllib.request.urlopen(request, timeout=min(120, remaining)) as response:
                body = json.loads(response.read().decode("utf-8"))
            if not isinstance(body, dict):
                raise json.JSONDecodeError('Expected a JSON object', '', 0)
            break
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise RuntimeError(
                    "Apps Script denied anonymous access. Redeploy the web app "
                    "with 'Who has access: Anyone' and update GDRIVE_WEBAPP_URL."
                ) from error
            if error.code not in RETRYABLE_HTTP_CODES:
                raise RuntimeError(
                    f"Upload failed for {path.name}: HTTP {error.code} "
                    f"after {attempt} attempt(s)"
                ) from error
            if attempt == MAX_UPLOAD_ATTEMPTS:
                raise TransientUploadError(
                    f"Upload failed for {path.name}: HTTP {error.code} after {attempt} attempts"
                ) from error
            delay = min(60, 10 * 2 ** (attempt - 1))
            print(
                f"retrying: {path.name} after HTTP {error.code} "
                f"({attempt}/{MAX_UPLOAD_ATTEMPTS}, wait {delay}s)",
                flush=True,
            )
            wait_for_retry(delay, deadline)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == MAX_UPLOAD_ATTEMPTS:
                raise TransientUploadError(
                    f"Upload failed for {path.name} after {attempt} attempts: {error}"
                ) from error
            delay = min(60, 10 * 2 ** (attempt - 1))
            print(
                f"retrying: {path.name} after transient error "
                f"({attempt}/{MAX_UPLOAD_ATTEMPTS}, wait {delay}s): {error}",
                flush=True,
            )
            wait_for_retry(delay, deadline)
    if body is None:
        raise RuntimeError(f"Upload failed for {path.name} without a response")
    if body.get("status") != "success":
        raise RuntimeError(
            f"Google Apps Script rejected {path.name}: "
            f"{body.get('message', 'unknown error')}"
        )
    print(f"uploaded: {path.name} -> {body.get('fileId')}", flush=True)


def upload_files(webapp_url: str, token: str, paths: list[Path], subfolder: str,
                 deadline: float | None = None) -> None:
    """Keep successful uploads, then recover only transiently failed files."""
    if deadline is None:
        deadline = time.monotonic() + UPLOAD_BUDGET_SECONDS
    pending = list(paths)
    failures = {}
    for pass_number in range(len(RECOVERY_DELAYS) + 1):
        if pass_number:
            delay = RECOVERY_DELAYS[pass_number - 1]
            print(f'Retrying {len(pending)} failed files after {delay}s', flush=True)
            wait_for_retry(delay, deadline)
        retry = []
        for path in pending:
            try:
                upload_file(webapp_url, token, path, subfolder, deadline=deadline)
                failures.pop(path.name, None)
            except TransientUploadError as error:
                failures[path.name] = str(error)
                retry.append(path)
                print(f'Deferred transient failure: {path.name}', flush=True)
        if not retry:
            return
        pending = retry
    raise TransientUploadError('Google Drive backup incomplete: ' +
                               '; '.join(f'{name}: {error}' for name, error in failures.items()))


def main() -> None:
    webapp_url = required_environment("GDRIVE_WEBAPP_URL")
    token = required_environment("GDRIVE_UPLOAD_TOKEN")
    source_dir = Path(os.environ.get("GDRIVE_SOURCE_DIR", "data"))
    now = datetime.now(JST)
    subfolder = f"data_{now.month}_{now.day}"
    paths = [Path(value) for value in sorted(glob.glob(str(source_dir / "*_master.csv")))]
    if not paths:
        raise RuntimeError(f"No *_master.csv files found under {source_dir.resolve()}")
    # The deployed Apps Script may still accept only *_master.csv. Enable JSONL
    # only after redeploying Code.gs with the expanded filename allowlist.
    if os.environ.get('GDRIVE_INCLUDE_JSONL', '').lower() in {'1', 'true', 'yes'}:
        migrate_legacy_logs(source_dir)
        paths += sorted(source_dir.glob('observation_*.jsonl'))
        paths += change_log_paths(source_dir)
        anonymous_sold = source_dir / 'anonymous_sold_inventory.jsonl'
        if anonymous_sold.exists():
            paths.append(anonymous_sold)
    with tempfile.TemporaryDirectory(prefix='drive-backup-') as directory:
        data, manifests = prepare_uploads(paths, Path(directory), MAX_FILE_BYTES)
        print(f"Uploading {len(data)} data files and {len(manifests)} manifests to {subfolder}", flush=True)
        deadline = time.monotonic() + UPLOAD_BUDGET_SECONDS
        upload_files(webapp_url, token, data, subfolder, deadline=deadline)
        # Publish metadata only when every part succeeded, using the same budget.
        if manifests:
            upload_files(webapp_url, token, manifests, subfolder, deadline=deadline)
    print("Google Drive backup completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
