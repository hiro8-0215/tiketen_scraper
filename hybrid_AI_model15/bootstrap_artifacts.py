"""Bootstrap Model15 semantics, dataset manifest, and verified BERT cache rows."""
from __future__ import annotations
import hashlib
import json
import os
import numpy as np
import pandas as pd

from config import ARTIFACT_DIR, ROOT, SEMANTIC_FEATURES_FILE


def _binary(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "なし"}:
            return 0
        if normalized in {"1", "true", "yes", "あり"}:
            return 1
    return int(bool(value))


def bootstrap_semantics():
    legacy_path = ROOT / "hybrid_AI_model13" / "train_data" / "llm_extracted_features.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    # Preserve already enriched Qwen15 rows when bootstrap is run again.  The
    # legacy file is only a fallback for descriptions that have not yet been
    # processed by Model15.
    if SEMANTIC_FEATURES_FILE.exists():
        output = json.loads(SEMANTIC_FEATURES_FILE.read_text(encoding="utf-8"))
    else:
        output = {}
    legacy_added = 0
    for description, row in legacy.items():
        description = str(description)
        if description in output:
            continue
        output[description] = {
            "semantic_seat_level": row.get("seat_level", "不明"),
            "semantic_row_position": row.get("row_position", "不明"),
            "semantic_is_fc_early": _binary(row.get("is_fc_early", False)),
            "semantic_is_random": _binary(row.get("is_random", False)),
            "semantic_winning_route": "不明",
            "semantic_name_status": "不明",
            "semantic_identity_check": "不明",
            "semantic_distribution_type": "不明",
            "semantic_visibility": "不明",
            "semantic_confidence": 0.5,
            "semantic_available": 1,
            "semantic_source": "model13_legacy",
            "semantic_schema_version": "model13_legacy_v1",
        }
        legacy_added += 1
    # Defence in depth: a price estimate is never a Model15 semantic feature,
    # even if a manually edited cache accidentally contains one.
    for row in output.values():
        row.pop("price_estimate", None)
        row.pop("semantic_price_estimate", None)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_DIR / "semantic_features.bootstrap.tmp.json"
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, SEMANTIC_FEATURES_FILE)
    print(f"semantic JSON: {len(output):,} descriptions; legacy added={legacy_added:,} (price_estimate excluded)")


def main():
    bootstrap_semantics()
    from data_loader import prepare_dataset
    df = prepare_dataset()
    ids = df.ticket_id.tolist()
    text_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in df["model_text"].fillna("").astype(str)
    ]

    # Reuse only Model15 embeddings whose ticket ID *and exact BERT input text*
    # are unchanged. Model14 stored IDs but no text hashes, so using that cache
    # after a snapshot update cannot be proven safe and is intentionally avoided.
    combined = np.full((len(ids), 768), np.nan, dtype=np.float32)
    current_embedding = ARTIFACT_DIR / "bert_raw.npy"
    current_rows_path = ARTIFACT_DIR / "bert_rows.json"
    current_hashes_path = ARTIFACT_DIR / "bert_text_hashes.json"
    if current_embedding.exists() and current_rows_path.exists() and current_hashes_path.exists():
        current_rows = json.loads(current_rows_path.read_text(encoding="utf-8"))
        current_hashes = json.loads(current_hashes_path.read_text(encoding="utf-8"))
        current_bert = np.load(current_embedding, mmap_mode="r")
        if (
            current_bert.shape == (len(current_rows), 768)
            and len(current_hashes) == len(current_rows)
            and len(set(current_rows)) == len(current_rows)
        ):
            current_positions = pd.Index(current_rows).get_indexer(ids)
            current_match = np.array([
                position >= 0
                and current_hashes[position] == text_hashes[index]
                and np.isfinite(current_bert[position]).all()
                for index, position in enumerate(current_positions)
            ])
            if current_match.any():
                combined[current_match] = np.asarray(current_bert[current_positions[current_match]])
        del current_bert

    # Write to a temporary file first: an older bootstrap may have hard-linked
    # the target to Model14, and direct overwrite would corrupt that source.
    bert_target = ARTIFACT_DIR / "bert_raw.npy"
    bert_temp = ARTIFACT_DIR / "bert_raw.model15.tmp.npy"
    np.save(bert_temp, combined)
    os.replace(bert_temp, bert_target)
    rows_target = ARTIFACT_DIR / "bert_rows.json"
    rows_temp = ARTIFACT_DIR / "bert_rows.model15.tmp.json"
    rows_temp.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    os.replace(rows_temp, rows_target)
    hashes_target = ARTIFACT_DIR / "bert_text_hashes.json"
    hashes_temp = ARTIFACT_DIR / "bert_text_hashes.model15.tmp.json"
    hashes_temp.write_text(json.dumps(text_hashes), encoding="utf-8")
    os.replace(hashes_temp, hashes_target)
    (ARTIFACT_DIR / "dataset_manifest.json").write_text(
        json.dumps({
            "model": "Model 15",
            "cleaning_policy": "model13_exact",
            "rows": len(ids),
            "ticket_ids": ids,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    coverage = float(df.semantic_available.mean() * 100)
    bert_ready = int(np.isfinite(combined).all(axis=1).sum())
    print(
        f"BERT cache: reused={bert_ready:,}, missing={len(ids) - bert_ready:,}; "
        f"clean rows={len(ids):,}; semantic coverage={coverage:.1f}%. "
        "Extract missing BERT rows, create folds, and retrain Qwen OOF."
    )


if __name__ == "__main__":
    main()
