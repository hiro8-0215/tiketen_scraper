"""Print a compact comparison of the latest frozen-model future validation."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.folder:
        folder = args.folder.resolve()
    else:
        pointer = HERE / "artifacts" / "latest.json"
        if not pointer.exists():
            raise SystemExit("完了済みの3モデル将来検証はまだありません")
        folder = Path(load(pointer)["folder"])
    manifest = load(folder / "manifest.json")
    if manifest.get("status") != "completed":
        raise SystemExit(f'検証は未完了です: {manifest.get("status")}')
    if manifest.get('evaluation_version') == 'snapshot_evidence_v2':
        print(json.dumps(load(folder/'strict_report.json'),ensure_ascii=False,indent=2))
        print(f'Results: {folder}')
        return
    print('WARNING: legacy evaluation extrapolated stale observations. Use the new snapshot validation before interpreting these metrics.')
    demand = load(folder / "demand_report.json")
    alternative = load(folder / "alternative_report.json")
    buy = load(folder / "buy_report.json")
    result = {
        "evaluation_period": {
            "start_exclusive": manifest["evaluation_start_exclusive"],
            "end": demand["observation_end"],
        },
        "data_audit": manifest["data_audit"],
        "demand": {
            "rows": demand["eligible_rows"],
            "class_counts": demand["class_counts"],
            "metrics": demand["metrics"],
            "sold_validation_supported": demand.get("sold_validation_supported", {}),
        },
        "alternative": {
            "rows": alternative["eligible_rows"],
            "class_counts": alternative["class_counts"],
            "metrics": alternative["metrics"],
        },
        "buy_timing": buy,
        "interpretation_guard": (
            "new sold transitions are absent, so sold-class and realized-purchase-profit "
            "claims are not supported by this validation"
        ),
        "result_folder": str(folder),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
