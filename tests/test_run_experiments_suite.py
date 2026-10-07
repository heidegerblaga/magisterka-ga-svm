import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from run_experiments import ExperimentSettings, SVMConfig, _config_id, _k_for_requested_size


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_experiments.py"


def _write_dataset(tmp_path: Path):
    X, y = make_moons(n_samples=140, noise=0.15, random_state=7)
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    train, temp = train_test_split(frame, test_size=0.4, random_state=7, stratify=frame["label"])
    validation, test = train_test_split(temp, test_size=0.5, random_state=7, stratify=temp["label"])

    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, validation_path, test_path, len(train), len(validation), len(test)


def _run_suite(tmp_path: Path, *extra_args: str, check: bool = True):
    train_path, validation_path, test_path, train_count, validation_count, test_count = _write_dataset(
        tmp_path
    )
    output_dir = tmp_path / "results"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--dataset",
        "smoke",
        "--dataset-name",
        "Smoke",
        "--train-csv",
        str(train_path),
        "--validation-csv",
        str(validation_path),
        "--test-csv",
        str(test_path),
        "--target",
        "label",
        "--header",
        "true",
        "--algorithms",
        "full_svm",
        "ga_svm",
        "dag_svm",
        "--train-sizes",
        "10",
        "20",
        "--runs",
        "2",
        "--generations",
        "2",
        "--population-size",
        "6",
        "--early-stopping-patience",
        "1",
        "--dag-cycles",
        "2",
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )
    return completed, output_dir, train_count, validation_count, test_count


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _config_id_for_prefix(config_ids: set[str], prefix: str) -> str:
    matches = [config_id for config_id in config_ids if config_id.startswith(prefix)]
    assert len(matches) == 1
    return matches[0]


def test_fixed_validation_suite_exports_consistent_raw_configs_and_aggregates(tmp_path):
    _, output_dir, train_count, validation_count, _ = _run_suite(tmp_path)

    expected_files = [
        output_dir / "raw_runs.csv",
        output_dir / "configs.csv",
        output_dir / "aggregated_results.csv",
        output_dir / "manifest.json",
    ]
    for path in expected_files:
        assert path.exists()

    raw_rows = _read_csv(output_dir / "raw_runs.csv")
    config_rows = _read_csv(output_dir / "configs.csv")
    aggregate_rows = _read_csv(output_dir / "aggregated_results.csv")

    assert len(raw_rows) == 9
    assert len(config_rows) == 5
    assert len(aggregate_rows) == 5
    assert {row["validation_adaptive"] for row in raw_rows} == {"False"}
    assert {row["validation_adaptive"] for row in config_rows} == {"False"}
    assert {row["validation_adaptive"] for row in aggregate_rows} == {"False"}
    expected_config_ids = {row["config_id"] for row in config_rows}
    assert {row["config_id"] for row in aggregate_rows} == expected_config_ids
    assert {row["config_id"] for row in raw_rows} == expected_config_ids
    assert len({row["run_id"] for row in raw_rows}) == len(raw_rows)
    for prefix in [
        "smoke__full_svm__full_train__cfg_",
        "smoke__ga_svm__train_10__cfg_",
        "smoke__ga_svm__train_20__cfg_",
        "smoke__dag_svm__train_10__cfg_",
        "smoke__dag_svm__train_20__cfg_",
    ]:
        _config_id_for_prefix(expected_config_ids, prefix)

    rows_by_config = defaultdict(list)
    for row in raw_rows:
        rows_by_config[row["config_id"]].append(row)

    full_config_id = _config_id_for_prefix(expected_config_ids, "smoke__full_svm__full_train__cfg_")
    full_rows = rows_by_config[full_config_id]
    assert len(full_rows) == 1
    assert full_rows[0]["algorithm_name"] == "full_svm"
    assert full_rows[0]["requested_training_set_size"] == str(train_count)
    assert full_rows[0]["training_set_size"] == str(train_count)
    assert full_rows[0]["validation_set_size"] == str(validation_count)

    full_config = next(row for row in config_rows if row["config_id"] == full_config_id)
    assert full_config["requested_training_set_size"] == ""
    assert full_config["runs"] == "1"
    assert full_config["generations"] == ""
    assert full_config["population_size"] == ""
    assert not any("__full_svm__train_" in row["config_id"] for row in config_rows)

    for config_id in expected_config_ids - {full_config_id}:
        assert len(rows_by_config[config_id]) == 2
        assert {row["seed"] for row in rows_by_config[config_id]} == {"42", "43"}
        assert "seed" not in config_id
        assert "run_" not in config_id
        for row in rows_by_config[config_id]:
            assert row["requested_training_set_size"] in {"10", "20"}
            if row["algorithm_name"] == "ga_svm":
                assert row["training_set_size"] == row["requested_training_set_size"]
            if row["algorithm_name"] == "dag_svm":
                assert 0 < int(row["training_set_size"]) <= int(row["requested_training_set_size"])
            assert row["validation_set_size"] == str(validation_count)

    for row in config_rows:
        if row["algorithm_name"] != "full_svm":
            assert row["requested_training_set_size"] in {"10", "20"}
            assert row["runs"] == "2"
            assert row["generations"] == "2"
            assert row["population_size"] == "6"

    aggregate_by_config = {row["config_id"]: row for row in aggregate_rows}
    for config_id, grouped_rows in rows_by_config.items():
        aggregate = aggregate_by_config[config_id]
        assert int(aggregate["runs_count"]) == len(grouped_rows)
        assert aggregate["mean_test_accuracy"] != ""
        assert aggregate["std_test_accuracy"] != ""
        assert aggregate["mean_validation_accuracy"] != ""
        expected_mean = sum(float(row["test_accuracy"]) for row in grouped_rows) / len(grouped_rows)
        assert abs(float(aggregate["mean_test_accuracy"]) - expected_mean) < 1e-12

    assert float(aggregate_by_config[full_config_id]["std_test_accuracy"]) == 0.0

    run_ids = set()
    for row in raw_rows:
        run_ids.add(row["run_id"])
        assert json.loads(row["confusion_matrix_validation"])
        run_json_path = Path(row["run_json_path"])
        config_path = Path(row["config_path"])
        assert run_json_path.exists()
        assert config_path.exists()

        run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert run_payload["row"]["config_id"] == row["config_id"]
        assert run_payload["row"]["run_id"] == row["run_id"]
        assert str(run_payload["row"]["seed"]) == row["seed"]
        assert run_payload["row"]["validation_adaptive"] is False
        assert config_payload["config_id"] == row["config_id"]
        assert config_payload["validation_adaptive"] is False
        if row["algorithm_name"] == "full_svm":
            assert run_payload["selected_train_indices"] is None
        else:
            assert len(run_payload["selected_train_indices"]) == int(row["training_set_size"])

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_adaptive"] is False
    assert manifest["raw_runs_csv"].endswith("raw_runs.csv")
    assert manifest["configs_count"] == len(config_rows)
    assert manifest["raw_runs_count"] == len(raw_rows)
    assert manifest["aggregated_configs_count"] == len(aggregate_rows)
    assert set(manifest["config_ids"]) == expected_config_ids
    assert set(manifest["run_ids"]) == run_ids


def test_fixed_validation_false_is_blocked(tmp_path):
    completed, _, _, _, _ = _run_suite(tmp_path, "--fixed-validation", "false", check=False)

    assert completed.returncode != 0
    assert "only --fixed-validation true" in completed.stderr


def test_runs_must_be_positive(tmp_path):
    completed, _, _, _, _ = _run_suite(tmp_path, "--runs", "0", check=False)

    assert completed.returncode != 0
    assert "--runs must be >= 1" in completed.stderr


def test_requested_train_size_must_map_to_exact_k_per_class():
    y_three_classes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    try:
        _k_for_requested_size(10, y_three_classes)
    except ValueError as exc:
        assert "must be divisible" in str(exc)
    else:
        raise AssertionError("Expected non-divisible train size to fail")


def test_config_id_is_stable_without_seed_and_changes_with_core_params():
    def settings(**overrides):
        base = {
            "dataset_name": "Smoke",
            "algorithms": ["ga_svm"],
            "train_sizes": [10],
            "runs": 2,
            "fixed_validation": True,
            "output_dir": Path("results/smoke"),
            "base_seed": 42,
            "metric": "accuracy",
            "generations": 2,
            "population_size": 6,
            "mutation_rate": 0.05,
            "crossover_rate": 0.8,
            "tournament_k": 3,
            "elitism": 2,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.001,
            "dag_cycles": 2,
            "dag_multiplier": 1.5,
            "svm_cfg": SVMConfig(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=False),
        }
        base.update(overrides)
        return ExperimentSettings(**base)

    base_id = _config_id("ga_svm", 10, settings())

    assert base_id == _config_id("ga_svm", 10, settings(base_seed=999, runs=20))
    assert "seed" not in base_id
    assert base_id != _config_id("ga_svm", 20, settings())
    assert base_id != _config_id("dag_svm", 10, settings())
    assert base_id != _config_id("ga_svm", 10, settings(dataset_name="Other"))
    assert base_id != _config_id("ga_svm", 10, settings(population_size=8))
    assert base_id != _config_id(
        "ga_svm",
        10,
        settings(
            svm_cfg=SVMConfig(
                kernel="rbf",
                C=11.0,
                gamma="scale",
                class_weight="balanced",
                probability=False,
            )
        ),
    )
