"""Restore downloaded Drive CSV parts to the original master filenames."""
import argparse
import csv
from pathlib import Path

from drive_backup_parts import restore_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        paths = restore_directory(args.source_dir, args.output_dir)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f'Restore failed: {error}\n')
    for path in paths:
        print(f'Restored: {path.name}')


if __name__ == '__main__':
    main()
