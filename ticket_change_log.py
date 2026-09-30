"""Size-bounded JSONL change history; record schemas and original bytes stay intact.

Single-writer storage, matching the collector workflow's concurrency group.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


MAX_PART_BYTES = 30 * 1024 * 1024
LOG_NAME = re.compile(r'^ticket_changes_(\d{8})(?:_(\d{3,}))?\.jsonl$')


def change_log_paths(folder, date=None):
    """Read legacy and numbered files in date/part order, including part 1000+."""
    matches = []
    for path in Path(folder).glob('ticket_changes_*.jsonl'):
        match = LOG_NAME.fullmatch(path.name)
        if match and (date is None or match[1] == date):
            matches.append((match[1], int(match[2] or 0), path))
    return [path for _, _, path in sorted(matches)]


def _digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def migrate_legacy_log(path, max_bytes=MAX_PART_BYTES):
    """Stage whole lines, verify SHA-256, then publish parts and remove the source.

    An interrupted publication can be retried: existing identical parts are reused.
    Conflicting parts and oversized individual records leave the source intact.
    """
    path = Path(path)
    match = LOG_NAME.fullmatch(path.name)
    if not match or match[2] is not None or max_bytes < 1:
        raise ValueError('Expected a legacy change log and a positive byte limit')
    source_hash = _digest([path])
    with tempfile.TemporaryDirectory(prefix='.ticket-log-', dir=path.parent) as temporary:
        staged, size, count = [], 0, 0
        output = None
        try:
            with path.open('rb') as source:
                for line in source:
                    if not line.endswith(b'\n'):
                        raise ValueError(f'{path.name}: incomplete final JSONL record')
                    if len(line) > max_bytes:
                        raise ValueError(f'{path.name}: a single record exceeds {max_bytes} bytes')
                    if output is None or size + len(line) > max_bytes:
                        if output is not None:
                            output.close()
                        name = f'ticket_changes_{match[1]}_{len(staged) + 1:03d}.jsonl'
                        part = Path(temporary) / name
                        staged.append(part)
                        output = part.open('wb')
                        size = 0
                    output.write(line)
                    size += len(line)
                    count += 1
        finally:
            if output is not None:
                output.close()
        if _digest(staged) != source_hash or _digest([path]) != source_hash:
            raise RuntimeError(f'{path.name}: source changed or staged bytes differ')
        destinations = [path.parent / part.name for part in staged]
        unexpected = set(change_log_paths(path.parent, match[1])) - set(destinations) - {path}
        if unexpected:
            raise RuntimeError(f'{path.name}: unexpected existing numbered parts; source retained')
        # Check all conflicts before publishing any new part.
        for part, destination in zip(staged, destinations):
            if destination.exists() and _digest([destination]) != _digest([part]):
                raise RuntimeError(f'Conflicting change log part: {destination.name}')
        for part, destination in zip(staged, destinations):
            if not destination.exists():
                os.replace(part, destination)
        if _digest(destinations) != source_hash or _digest([path]) != source_hash:
            raise RuntimeError(f'{path.name}: published bytes differ; source retained')
        path.unlink()
    return {'source': path.name, 'parts': len(destinations), 'records': count,
            'sha256': source_hash, 'bytes_verified': True}


def migrate_legacy_logs(folder, max_bytes=MAX_PART_BYTES):
    reports = []
    for path in change_log_paths(folder):
        if LOG_NAME.fullmatch(path.name)[2] is None:
            reports.append(migrate_legacy_log(path, max_bytes))
        elif path.stat().st_size > max_bytes:
            raise ValueError(f'{path.name}: numbered part exceeds {max_bytes} bytes')
    return reports


def append_changes(folder, date, records, max_bytes=MAX_PART_BYTES):
    """Append the unchanged JSON serialization, rotating before crossing the cap."""
    if not re.fullmatch(r'\d{8}', date) or max_bytes < 1:
        raise ValueError('Expected YYYYMMDD and a positive byte limit')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    legacy = folder / f'ticket_changes_{date}.jsonl'
    if legacy.exists():
        migrate_legacy_log(legacy, max_bytes)
    paths = change_log_paths(folder, date)
    index = int(LOG_NAME.fullmatch(paths[-1].name)[2]) if paths else 1
    path = paths[-1] if paths else folder / f'ticket_changes_{date}_{index:03d}.jsonl'
    size = path.stat().st_size if path.exists() else 0
    if size:
        with path.open('rb') as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b'\n':
                raise ValueError(f'{path.name}: incomplete final JSONL record')
    for record in records:
        line = (json.dumps(record, ensure_ascii=True) + '\n').encode('utf-8')
        if len(line) > max_bytes:
            raise ValueError(f'A single ticket change exceeds {max_bytes} bytes')
        if size + len(line) > max_bytes:
            index += 1
            path = folder / f'ticket_changes_{date}_{index:03d}.jsonl'
            size = 0
        with path.open('ab') as stream:
            stream.write(line)
        size += len(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data')
    args = parser.parse_args()
    for report in migrate_legacy_logs(args.data_dir):
        print(json.dumps(report))


if __name__ == '__main__':
    main()
