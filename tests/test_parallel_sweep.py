import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_moons


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from ga.sweep import generate_trial_plan


SWEEP_SCRIPT = REPO_ROOT / "scripts" / "run_parallel_sweep.py"


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


def _base_config(train_path: Path, val_path: Path, test_path: Path):
    return {
        "seed": 42,
        "experiment_name": "parallel_sweep_demo",
        "dataset_name": "Chessboard",
        "algorithm_mode": "gasvm",
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
        "encoding": {"k_per_class": 8},
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
        "val_cumulation": {"initial_fraction": 0.5},
    }


def test_generate_trial_plan_deduplicates_irrelevant_full_svm_dimensions():
    base_cfg = {
        "experiment_name": "dedupe",
        "dataset_name": "Chessboard",
        "algorithm_mode": "full_svm",
        "seed": 42,
        "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "ga": {"population_size": 8},
        "encoding": {"k_per_class": 8},
    }
    search_space = {
        "algorithm_mode": ["full_svm"],
        "svm.C": [0.5, 1.0, 2.0],
        "ga.population_size": [8, 16],
        "encoding.k_per_class": [4, 8, 12],
    }

    trials = generate_trial_plan(base_cfg, search_space, num_trials=20, selection_seed=42, sweep_id="swp")

    assert len(trials) == 3
    assert {trial["config"]["svm"]["C"] for trial in trials} == {0.5, 1.0, 2.0}


def test_parallel_sweep_runs_multiple_trials(tmp_path):
    train_path, val_path, test_path = _write_dataset_bundle(tmp_path)
    base_cfg = _base_config(train_path, val_path, test_path)
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    sweep_cfg = {
        "sweep_name": "smoke_parallel",
        "num_trials": 4,
        "max_parallel": 2,
        "selection_seed": 42,
        "objective": {"name": "full_validation_f1", "mode": "max"},
        "search_space": {
            "algorithm_mode": ["gasvm"],
            "svm.C": [0.5, 1.0],
            "encoding.k_per_class": [6, 8],
            "ga.population_size": [6, 8],
        },
    }
    sweep_config_path = tmp_path / "sweep.yaml"
    sweep_config_path.write_text(yaml.safe_dump(sweep_cfg, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SWEEP_SCRIPT),
            "--config",
            str(base_config_path),
            "--sweep-config",
            str(sweep_config_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    sweeps = list((tmp_path / "artifacts" / "sweeps").iterdir())
    assert len(sweeps) == 1

    sweep_root = sweeps[0]
    state = json.loads((sweep_root / "state.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv(sweep_root / "ranking.csv")
    history = pd.read_csv(tmp_path / "artifacts" / "history_table.csv")
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "artifacts" / "runs").glob("*/metrics/summary.json"))
    ]

    assert state["status"] == "completed"
    assert state["completed_trials"] == 4
    assert state["failed_trials"] == 0
    assert len(ranking) == 4
    assert len(history) == 4
    assert len(summaries) == 4
    assert len({summary["run_id"] for summary in summaries}) == 4
    assert {summary["sweep_id"] for summary in summaries} == {state["sweep_id"]}
    assert {summary["trial_id"] for summary in summaries} == {f"trial_{idx:03d}" for idx in range(1, 5)}
