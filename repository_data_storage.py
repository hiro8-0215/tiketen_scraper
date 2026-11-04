"""Lossless gzip storage for large operational CSVs in Git (not Git LFS)."""
import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

COMPRESS_AT_BYTES = 50 * 1024 * 1024
MAX_GIT_BYTES = 95 * 1024 * 1024


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def pack_csv(path, threshold=COMPRESS_AT_BYTES, force=False):
    path = Path(path)
    archive = path.with_suffix(path.suffix + '.gz')
    if path.stat().st_size < threshold and not archive.exists() and not force:
        return None
    temporary = archive.with_suffix(archive.suffix + '.tmp')
    try:
        with path.open('rb') as source:
            before = digest(source)
            source.seek(0)
            with temporary.open('wb') as output:
                with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed, length=1024 * 1024)
        if temporary.stat().st_size >= MAX_GIT_BYTES:
            raise RuntimeError(f'{archive.name}: compressed file still exceeds safe Git limit')
        with gzip.open(temporary, 'rb') as restored, path.open('rb') as source:
            if digest(restored) != before or digest(source) != before:
                raise RuntimeError(f'{path.name}: gzip verification failed or source changed')
        original_size = path.stat().st_size
        os.replace(temporary, archive)
        path.unlink()
        print(f'[GIT STORAGE] {path}: {original_size} -> {archive.stat().st_size} bytes; SHA256={before}', flush=True)
        return archive
    finally:
        temporary.unlink(missing_ok=True)


def restore_data(folder, masters_only=False, skip_existing=False):
    folder = Path(folder)
    paths = sorted(folder.glob('*_master.csv.gz') if masters_only else folder.rglob('*.csv.gz'))
    restored = []
    for archive in paths:
        target = archive.with_suffix('')
        if target.exists():
            if skip_existing:
                continue
            with target.open('rb') as source, gzip.open(archive, 'rb') as source_archive:
                if digest(source) != digest(source_archive):
                    raise RuntimeError(f'{target}: existing CSV differs; refusing to overwrite edits')
            continue
        temporary = target.with_suffix(target.suffix + '.tmp')
        try:
            with gzip.open(archive, 'rb') as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            os.replace(temporary, target)
            restored.append(target)
            print(f'[GIT STORAGE] Restored {target}', flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    return restored


def pack_data(folder, threshold=COMPRESS_AT_BYTES):
    folder = Path(folder)
    # A compressed snapshot may be tracked but absent in a sparse checkout.
    # Keep its representation even when this run regenerates a smaller CSV.
    tracked = subprocess.run(['git', 'ls-files', '-z', '--', '*.csv.gz'], cwd=folder,
                             capture_output=True)
    archives = {folder / name.decode('utf-8') for name in tracked.stdout.split(b'\0') if name} if tracked.returncode == 0 else set()
    packed = [result for path in sorted(folder.rglob('*.csv'))
              if (result := pack_csv(path, threshold, force=path.with_suffix('.csv.gz') in archives)) is not None]
    oversized = [path for path in folder.rglob('*') if path.is_file() and path.stat().st_size >= MAX_GIT_BYTES]
    if oversized:
        raise RuntimeError('Oversized Git data files: ' + ', '.join(str(path) for path in oversized))
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['restore', 'pack'])
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--masters-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'restore':
        restore_data(args.data_dir, masters_only=args.masters_only)
    else:
        pack_data(args.data_dir)


if __name__ == '__main__':
    main()
