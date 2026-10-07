import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.reporting import (
    HISTORY_TABLE_COLUMNS,
    algorithm_metadata,
    build_report_row,
    rebuild_history_table,
)


def test_algorithm_metadata_mapping():
    assert algorithm_metadata("full_svm")["training_set_selection"] == "No"
    assert algorithm_metadata("gasvm")["validation_set_selection"] == "No"
    # DAGASVM (adaptacyjne K) działa na stałej walidacji; wybór V' ma dopiero wariant fazowy
    assert algorithm_metadata("dagasvm")["validation_set_selection"] == "No"
    assert algorithm_metadata("dagasvm_phase")["validation_set_selection"] == "Yes"


def test_build_report_row_normalizes_existing_report_row():
    row = {column: None for column in HISTORY_TABLE_COLUMNS}
    row["dataset"] = "Chessboard"
    row["algorithm"] = "GASVM"
    row["training_set_selection"] = "Yes"
    row["validation_set_selection"] = "No"

    summary = {
        "report_row": row,
        "run_id": "run_001",
    }

    normalized = build_report_row(summary)

    assert normalized["dataset"] == "Chessboard"
    assert normalized["algorithm"] == "GASVM"
    assert set(normalized.keys()) == set(HISTORY_TABLE_COLUMNS)


def test_rebuild_history_table_scans_summary_files(tmp_path):
    summary_dir = tmp_path / "artifacts" / "runs" / "run_001" / "metrics"
    summary_dir.mkdir(parents=True)
    row = {column: None for column in HISTORY_TABLE_COLUMNS}
    row.update(
        {
            "dataset": "Chessboard",
            "algorithm": "Full SVM",
            "training_set_selection": "No",
            "validation_set_selection": "No",
            "run_id": "run_001",
            "summary_path": str(summary_dir / "summary.json"),
        }
    )
    summary = {
        "run_id": "run_001",
        "report_row": row,
    }
    (summary_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    rows = rebuild_history_table(tmp_path / "artifacts")
    history = pd.read_csv(tmp_path / "artifacts" / "history_table.csv")

    assert len(rows) == 1
    assert history.iloc[0]["run_id"] == "run_001"
    assert history.iloc[0]["algorithm"] == "Full SVM"


def test_build_report_row_reconstructs_from_summary_and_mask(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 1.0, 1.1, 0.2, 0.9],
            "x2": [0.0, 0.2, 1.0, 0.9, 0.1, 1.2],
            "label": [0, 0, 1, 1, 0, 1],
        }
    )
    val = pd.DataFrame(
        {
            "x1": [0.05, 1.05, 0.15, 0.95],
            "x2": [0.05, 0.95, 0.15, 1.05],
            "label": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "x1": [0.02, 1.02, 0.12, 0.98],
            "x2": [0.02, 0.98, 0.18, 1.08],
            "label": [0, 1, 0, 1],
        }
    )
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    run_metrics = tmp_path / "artifacts" / "runs" / "run_001" / "metrics"
    checkpoints = tmp_path / "artifacts" / "runs" / "run_001" / "checkpoints"
    run_metrics.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    mask_path = checkpoints / "global_best_train_mask.npy"
    np.save(mask_path, np.array([True, False, True, False, True, False], dtype=bool))

    cfg = {
        "seed": 42,
        "dataset_name": "Chessboard",
        "algorithm_mode": "gasvm",
        "data": {
            "train_csv": str(train_path),
            "validation_csv": str(val_path),
            "test_csv": str(test_path),
            "target": "label",
            "stratify": False,
            "header": True,
            "target_candidates": ["label"],
            "max_rows": None,
        },
        "svm": {
            "kernel": "linear",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
        },
        "fitness": {"metric": "balanced_accuracy"},
    }
    (run_metrics / "config_snapshot.json").write_text(json.dumps(cfg), encoding="utf-8")

    summary = {
        "run_id": "run_001",
        "dataset_name": "Chessboard",
        "algorithm_mode": "gasvm",
        "experiment_duration_sec": 1.5,
        "created_at": "2026-03-31T12:00:00",
        "global_best": {
            "train_summary": {"rows": 3},
            "train_mask_path": "artifacts/runs/run_001/checkpoints/global_best_train_mask.npy",
        },
        "representative_model": {
            "algorithm_mode": "gasvm",
            "training_rows": 3,
            "validation_rows": 4,
            "train_mask_path": "artifacts/runs/run_001/checkpoints/global_best_train_mask.npy",
            "cycle": 1,
        },
    }
    summary_path = run_metrics / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    row = build_report_row(summary_path)

    assert row["algorithm"] == "GASVM"
    assert row["training_set_selection"] == "Yes"
    assert row["validation_set_selection"] == "No"
    assert row["training_set_size"] == 3
    assert row["validation_set_size"] == 4


def test_phase_mode_report_row_uses_representative_train_mask(tmp_path, monkeypatch):
    """Regresja: tryb dagasvm_phase pomijał maskę T' i liczył wiersz raportu na całym Train."""
    import ga.reporting as reporting

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.9, 1.0, 0.2, 0.8],
            "x2": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
            "label": [0, 0, 1, 1, 0, 1],
        }
    )
    for name in ("train", "validation", "test"):
        frame.to_csv(data_dir / f"{name}.csv", index=False)
    run = tmp_path / "artifacts" / "runs" / "run_001"
    (run / "metrics").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    mask_path = run / "checkpoints" / "global_best_train_mask.npy"
    np.save(mask_path, np.array([True, False, True, False, False, False]))
    cfg = {
        "seed": 1,
        "data": {
            "train_csv": str(data_dir / "train.csv"),
            "validation_csv": str(data_dir / "validation.csv"),
            "test_csv": str(data_dir / "test.csv"),
            "target": "label",
            "header": True,
            "max_rows": None,
        },
        "svm": {"kernel": "linear", "C": 1.0, "gamma": "scale", "standardize": False},
        "fitness": {"metric": "balanced_accuracy"},
    }
    summary = {
        "run_id": "run_001",
        "algorithm_mode": "dagasvm_phase",
        "representative_model": {
            "training_rows": 2,
            "validation_rows": 6,
            "train_mask_path": str(mask_path),
        },
    }
    seen_rows = []
    real_fit = reporting.fit_svm_model

    def spy(X, y, *args, **kwargs):
        seen_rows.append(len(X))
        return real_fit(X, y, *args, **kwargs)

    monkeypatch.setattr(reporting, "fit_svm_model", spy)
    reporting.build_report_row(summary, cfg=cfg)
    assert seen_rows == [2]  # model z maski T', a nie z całego Train (6 wierszy)
