import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_moons


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_experiment.py"


def _write_dataset_bundle(tmp_path: Path):
    X, y = make_moons(n_samples=180, noise=0.18, random_state=42)
    train_df = pd.DataFrame({"x1": X[:80, 0], "x2": X[:80, 1], "label": y[:80]})
    val_df = pd.DataFrame({"x1": X[80:130, 0], "x2": X[80:130, 1], "label": y[80:130]})
    test_df = pd.DataFrame({"x1": X[130:, 0], "x2": X[130:, 1], "label": y[130:]})

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, val_path, test_path


def _base_config(train_path: Path, val_path: Path, test_path: Path, mode: str):
    return {
        "seed": 42,
        "experiment_name": f"test_{mode}",
        "dataset_name": "Chessboard",
        "algorithm_mode": mode,
        "data": {
            "train_csv": str(train_path),
            "validation_csv": str(val_path),
            "test_csv": str(test_path),
            "target": "label",
            "stratify": True,
            "header": True,
            "target_candidates": ["label"],
            "max_rows": None,
        },
        "svm": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
        },
        "encoding": {
            "k_per_class": 8,
        },
        "fitness": {
            "metric": "balanced_accuracy",
            "class_imbalance_penalty": 0.0,
        },
        "ga": {
            "population_size": 8,
            "generations": 3,
            "tournament_k": 3,
            "crossover_rate": 0.8,
            "mutation_rate": 0.05,
            "elitism": 2,
            "early_stopping": {"patience": 2, "min_delta": 0.0},
        },
        "val_ga": {
            "subset_fraction": 0.5,
            "population_size": 8,
            "generations": 3,
            "tournament_k": 3,
            "crossover_rate": 0.8,
            "mutation_rate": 0.05,
            "elitism": 2,
            "early_stopping": {"patience": 2, "min_delta": 0.0},
        },
        "k_schedule": {
            "num_cycles": 2,
            "k_start": 8,
            "k_multiplier": 1.25,
        },
        "val_cumulation": {
            "initial_fraction": 0.5,
        },
    }


def _run_mode(tmp_path: Path, mode: str):
    train_path, val_path, test_path = _write_dataset_bundle(tmp_path / mode)
    cfg = _base_config(train_path, val_path, test_path, mode)
    config_path = tmp_path / f"{mode}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--config", str(config_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    latest = json.loads((tmp_path / "artifacts" / "latest_run.json").read_text(encoding="utf-8"))
    summary_path = tmp_path / latest["summary_path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return result, summary


def test_runner_writes_history_for_all_modes(tmp_path):
    _, full_summary = _run_mode(tmp_path, "full_svm")
    _, gasvm_summary = _run_mode(tmp_path, "gasvm")
    _, dagasvm_summary = _run_mode(tmp_path, "dagasvm")

    history = pd.read_csv(tmp_path / "artifacts" / "history_table.csv")

    assert full_summary["report_row"]["algorithm"] == "Full SVM"
    assert full_summary["report_row"]["training_set_selection"] == "No"
    assert full_summary["report_row"]["validation_set_selection"] == "No"

    assert gasvm_summary["report_row"]["algorithm"] == "GASVM"
    assert gasvm_summary["report_row"]["validation_set_selection"] == "No"
    assert len(gasvm_summary["cycles"]) == 1

    assert dagasvm_summary["report_row"]["algorithm"] == "DAGASVM"
    assert dagasvm_summary["report_row"]["validation_set_selection"] == "No"
    assert dagasvm_summary["report_row"]["validation_set_size"] == dagasvm_summary["representative_model"]["validation_rows"]

    assert len(history) == 3
    assert set(history["algorithm"]) == {"Full SVM", "GASVM", "DAGASVM"}
