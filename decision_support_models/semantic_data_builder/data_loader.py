"""Load unique descriptions from the latest snapshot without outcome fields."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATA_ROOT
from schema import normalize_text, text_hash


def snapshot_key(path: Path):
    values = path.name.removeprefix("data_").split("_")
    if len(values) == 2 and all(value.isdigit() for value in values):
        return (0, int(values[0]), int(values[1]))
    if len(values) == 3 and all(value.isdigit() for value in values):
        return tuple(map(int, values))
    return (-1, -1, -1)


def latest_data_dir() -> Path:
    choices = [path for path in DATA_ROOT.glob("data_*") if path.is_dir() and any(path.glob("*_master.csv"))]
    if not choices:
        raise FileNotFoundError(f"No snapshot under {DATA_ROOT}")
    return max(choices, key=snapshot_key)


def load_descriptions(data_dir: Path | None = None) -> tuple[pd.DataFrame, int]:
    selected = data_dir or latest_data_dir()
    frames, ticket_rows = [], 0
    for path in sorted(selected.glob("*_master.csv")):
        frame = pd.read_csv(path, usecols=lambda name: name in {"ticket_id", "raw_description"}, low_memory=False)
        ticket_rows += len(frame)
        if "raw_description" not in frame:
            frame["raw_description"] = ""
        frames.append(frame[["raw_description"]])
    if not frames:
        raise ValueError(f"No master CSV under {selected}")
    values = pd.concat(frames, ignore_index=True)["raw_description"].fillna("").map(normalize_text)
    descriptions = pd.DataFrame({"description": values.drop_duplicates()})
    descriptions["text_hash"] = descriptions["description"].map(text_hash)
    if descriptions.text_hash.duplicated().any():
        raise AssertionError("SHA-256 collision or inconsistent text normalization")
    descriptions.attrs["snapshot_dir"] = str(selected)
    return descriptions.reset_index(drop=True), ticket_rows
