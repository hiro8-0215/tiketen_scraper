"""Lossless CSV parts for the existing Apps Script filename allowlist."""
from __future__ import annotations

import csv
import hashlib
import itertools
import re
import shutil
import tempfile
from pathlib import Path

PART_BYTES = 30 * 1024 * 1024
MANIFEST_SUFFIX = '.backup_manifest_master.csv'
BACKUP_MANIFEST = 'drive_backup_manifest_master.csv'
PART_PATTERN = re.compile(r'\.part\d{3,}_master\.csv$')
FIELDS = ('version', 'source_filename', 'source_bytes', 'source_sha256',
          'source_rows', 'header_bytes', 'header_sha256', 'part_index',
          'part_filename', 'part_bytes', 'part_sha256', 'part_rows')


class CapturedLines:
    """Keep original UTF-8/newline bytes consumed for each logical CSV record."""
    def __init__(self, stream):
        self.stream = stream
        self.lines = []

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.stream)
        self.lines.append(line.encode('utf-8'))
        return line


def records(path: Path):
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(2**31 - 1)
    try:
        with path.open('r', encoding='utf-8', newline='') as stream:
            lines = CapturedLines(stream)
            reader = csv.reader(lines, strict=True)
            while True:
                lines.lines.clear()
                try:
                    next(reader)
                except StopIteration:
                    return
                yield b''.join(lines.lines)
    finally:
        csv.field_size_limit(previous_limit)


def is_backup_metadata(name: str) -> bool:
    return name == BACKUP_MANIFEST or name.endswith(MANIFEST_SUFFIX) or bool(PART_PATTERN.search(name))


def split_csv(source: Path, staging: Path, limit: int = PART_BYTES):
    """Return parts and a last-published manifest; never modify the source."""
    if not re.fullmatch(r'[A-Za-z0-9._-]+_master\.csv', source.name) or is_backup_metadata(source.name):
        raise ValueError(f'Not an original master CSV: {source.name}')
    staging.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    source_bytes = source_rows = 0
    stem = source.name.removesuffix('_master.csv')
    parts, entries = [], []
    iterator = records(source)
    try:
        header = next(iterator, None)
        if header is None or len(header) > limit:
            raise ValueError(f'{source.name}: empty CSV or header exceeds part size')
        source_hash.update(header)
        source_bytes += len(header)
        stream = None
        part_size = part_rows = 0
        part_hash = None

        def finish():
            if stream is not None:
                stream.close()
                entries.append((parts[-1].name, part_size, part_hash.hexdigest(), part_rows))

        try:
            for raw in iterator:
                if len(header) + len(raw) > limit:
                    raise ValueError(f'{source.name}: one CSV record exceeds {limit} bytes including header')
                if stream is None or part_size + len(raw) > limit:
                    finish()
                    path = staging / f'{stem}.part{len(parts) + 1:03d}_master.csv'
                    stream = path.open('wb')
                    parts.append(path)
                    stream.write(header)
                    part_hash = hashlib.sha256(header)
                    part_size, part_rows = len(header), 0
                stream.write(raw)
                part_hash.update(raw)
                part_size += len(raw)
                part_rows += 1
                source_hash.update(raw)
                source_bytes += len(raw)
                source_rows += 1
            if stream is None:
                path = staging / f'{stem}.part001_master.csv'
                path.write_bytes(header)
                parts.append(path)
                entries.append((path.name, len(header), hashlib.sha256(header).hexdigest(), 0))
            else:
                finish()
        finally:
            if stream is not None:
                stream.close()
    finally:
        iterator.close()
    manifest = staging / f'{stem}{MANIFEST_SUFFIX}'
    with manifest.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for index, (name, size, digest, rows) in enumerate(entries, 1):
            writer.writerow((1, source.name, source_bytes, source_hash.hexdigest(), source_rows,
                             len(header), hashlib.sha256(header).hexdigest(), index,
                             name, size, digest, rows))
    return parts, manifest


def prepare_uploads(paths: list[Path], staging: Path, upload_limit: int):
    data, entries = [], []
    for path in paths:
        if is_backup_metadata(path.name):
            raise ValueError('Restore Drive CSV parts before using this directory: ' + path.name)
        if path.stat().st_size <= upload_limit:
            data.append(path)
            if path.name.endswith('_master.csv'):
                digest = hashlib.sha256()
                size = rows = 0
                header = None
                for raw in records(path):
                    if header is None:
                        header = raw
                    else:
                        rows += 1
                    size += len(raw)
                    digest.update(raw)
                if header is None:
                    raise ValueError(f'Empty CSV: {path.name}')
                entries.append(dict(zip(FIELDS, (1, path.name, size, digest.hexdigest(), rows,
                               len(header), hashlib.sha256(header).hexdigest(), 1,
                               path.name, size, digest.hexdigest(), rows))))
        elif path.name.endswith('_master.csv'):
            parts, manifest = split_csv(path, staging, min(PART_BYTES, upload_limit))
            print(f'Split {path.name}: {len(parts)} CSV parts; original file unchanged', flush=True)
            data.extend(parts)
            with manifest.open('r', encoding='utf-8', newline='') as stream:
                entries.extend(csv.DictReader(stream))
            manifest.unlink()
        else:
            raise ValueError(f'{path.name}: exceeds upload limit; unsupported backup type')
    manifests = []
    if entries:
        staging.mkdir(parents=True, exist_ok=True)
        manifest = staging / BACKUP_MANIFEST
        with manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(entries)
        manifests.append(manifest)
    return data, manifests


def read_manifest(manifest: Path):
    with manifest.open('r', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f'Invalid manifest columns: {manifest.name}')
        entries = list(reader)
    if not entries:
        raise ValueError(f'Empty manifest: {manifest.name}')
    return entries


def restore_entries(manifest: Path, entries: list[dict], output: Path) -> Path:
    """Verify every part and source SHA-256 before replacing the output CSV."""
    first = entries[0]
    name = first['source_filename']
    if not re.fullmatch(r'[A-Za-z0-9._-]+_master\.csv', name) or is_backup_metadata(name):
        raise ValueError('Invalid source filename in manifest')
    stem = name.removesuffix('_master.csv')
    if manifest.name not in {BACKUP_MANIFEST, stem + MANIFEST_SUFFIX}:
        raise ValueError('Manifest filename does not match its source')
    common = FIELDS[:7]
    source_hash = hashlib.sha256()
    source_size = source_rows = 0
    header = None
    output.mkdir(parents=True, exist_ok=True)
    target = output / name
    with tempfile.NamedTemporaryFile(dir=output, prefix='.restore-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for index, entry in enumerate(entries, 1):
                if any(entry[key] != first[key] for key in common) or entry['version'] != '1':
                    raise ValueError('Inconsistent manifest metadata')
                unsplit = len(entries) == 1 and entry['part_filename'] == name
                expected = name if unsplit else f'{stem}.part{index:03d}_master.csv'
                if entry['part_index'] != str(index) or entry['part_filename'] != expected:
                    raise ValueError('Invalid part filename or order')
                part = manifest.parent / expected
                part_hash = hashlib.sha256()
                part_size = part_rows = 0
                for row_index, raw in enumerate(records(part)):
                    part_hash.update(raw)
                    part_size += len(raw)
                    if row_index == 0:
                        if header is None:
                            header = raw
                            if len(raw) != int(first['header_bytes']) or hashlib.sha256(raw).hexdigest() != first['header_sha256']:
                                raise ValueError('CSV header checksum mismatch')
                        elif raw != header:
                            raise ValueError('CSV part headers differ')
                        if index > 1:
                            continue
                    else:
                        part_rows += 1
                        source_rows += 1
                    stream.write(raw)
                    source_hash.update(raw)
                    source_size += len(raw)
                if part_size != int(entry['part_bytes']) or part_hash.hexdigest() != entry['part_sha256'] or part_rows != int(entry['part_rows']):
                    raise ValueError(f'CSV part checksum/size/row mismatch: {expected}')
            if source_size != int(first['source_bytes']) or source_hash.hexdigest() != first['source_sha256'] or source_rows != int(first['source_rows']):
                raise ValueError('Restored CSV checksum/size/row mismatch')
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def restore_manifest(manifest: Path, output: Path) -> Path:
    return restore_entries(manifest, read_manifest(manifest), output)


def restore_directory(source: Path, output: Path) -> list[Path]:
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output directory for restored master CSVs')
    if any(is_backup_metadata(path.name) for path in output.glob('*_master.csv')):
        raise ValueError('Output directory contains backup parts/metadata; use a clean output directory')
    global_manifest = source / BACKUP_MANIFEST
    if global_manifest.exists():
        entries = read_manifest(global_manifest)
        names = [name for name, _ in itertools.groupby(entry['source_filename'] for entry in entries)]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate source groups in manifest')
        if any(path.name not in names for path in output.glob('*_master.csv')):
            raise ValueError('Output directory contains masters outside this snapshot; use a clean output directory')
        output.mkdir(parents=True, exist_ok=True)
        # Validate the entire snapshot before replacing any user's existing CSV.
        with tempfile.TemporaryDirectory(dir=output, prefix='.verify-backup-') as directory:
            staging = Path(directory)
            restored = [restore_entries(global_manifest, list(group), staging)
                        for _, group in itertools.groupby(entries, key=lambda entry: entry['source_filename'])]
            return [path.replace(output / path.name) for path in restored]
    manifests = sorted(source.glob('*' + MANIFEST_SUFFIX))
    masters = sorted(path for path in source.glob('*_master.csv') if not is_backup_metadata(path.name))
    covered_parts = {manifest.name.removesuffix(MANIFEST_SUFFIX) for manifest in manifests}
    if any(PART_PATTERN.search(p.name) and p.name.rsplit('.part', 1)[0] not in covered_parts
           for p in source.glob('*_master.csv')):
        raise ValueError('CSV parts exist without a completed manifest; download a successful backup')
    if not manifests and not masters:
        raise ValueError('No master CSVs or backup manifests found')
    restored = [restore_manifest(manifest, output) for manifest in manifests]
    covered = {path.name for path in restored}
    # Prefer verified parts over a stale original left in the daily Drive folder.
    output.mkdir(parents=True, exist_ok=True)
    for path in masters:
        if path.name not in covered:
            target = output / path.name
            shutil.copyfile(path, target)
            restored.append(target)
    return restored
