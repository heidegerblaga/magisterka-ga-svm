from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_COLUMNS = [
    "Dataset",
    "Algorithm",
    "Training set selection",
    "Validation set selection",
    "Training time",
    "Training set size",
    "Validation set size",
    "No. of SVs",
    "Full training set precision",
    "Full training set recall",
    "Full training set F-score",
    "Full validation set precision",
    "Full validation set recall",
    "Full validation set F-score",
    "Test set precision",
    "Test set recall",
    "Test set F-score",
]


SOURCE_TO_OUTPUT = {
    "dataset": "Dataset",
    "algorithm": "Algorithm",
    "training_set_selection": "Training set selection",
    "validation_set_selection": "Validation set selection",
    "training_time_sec": "Training time",
    "training_set_size": "Training set size",
    "validation_set_size": "Validation set size",
    "num_support_vectors": "No. of SVs",
    "full_train_precision": "Full training set precision",
    "full_train_recall": "Full training set recall",
    "full_train_f1": "Full training set F-score",
    "full_validation_precision": "Full validation set precision",
    "full_validation_recall": "Full validation set recall",
    "full_validation_f1": "Full validation set F-score",
    "test_precision": "Test set precision",
    "test_recall": "Test set recall",
    "test_f1": "Test set F-score",
}


def _sort_key(row: dict[str, str]) -> tuple[str, str]:
    created_at = (row.get("created_at") or "").strip()
    run_id = (row.get("run_id") or "").strip()
    return (created_at, run_id)


def export_csv(source_path: Path, output_path: Path) -> int:
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rows.sort(key=_sort_key)
    exported_rows = []
    for row in rows:
        exported_rows.append(
            {output_key: row.get(source_key, "") for source_key, output_key in SOURCE_TO_OUTPUT.items()}
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(exported_rows)
    return len(exported_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained models into a thesis-friendly CSV.")
    parser.add_argument(
        "--source",
        default="artifacts/history_table.csv",
        help="Path to the source history table CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/all_trained_models_summary.csv",
        help="Path to the exported CSV.",
    )
    args = parser.parse_args()

    count = export_csv(Path(args.source), Path(args.output))
    print(f"exported_rows={count}")
    print(f"output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
