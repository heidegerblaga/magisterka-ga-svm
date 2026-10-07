from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix


REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT / "src"))

from ga.data.loader import _guess_target_name, load_dataset, split_features_labels
from ga.encoding import KPerClassConfig
from ga.fitness import FitnessConfig
from ga.loop.engine import EarlyStopping, GAConfig, run_ga_k_per_class
from ga.models.svm import SVMConfig, evaluate_fitted_svm, fit_svm_model
from ga.utils.random import set_global_seed


DATASET_REGISTRY = {
    "chessboard": {
        "dataset_name": "Chessboard",
        "train_csv": "data/raw/train.csv",
        "validation_csv": "data/raw/validation.csv",
        "test_csv": "data/raw/test.csv",
        "target": "label",
        "header": False,
    },
    "magic_gamma": {
        "dataset_name": "MAGIC Gamma Telescope",
        "train_csv": "data/benchmarks/magic_gamma/train.csv",
        "validation_csv": "data/benchmarks/magic_gamma/validation.csv",
        "test_csv": "data/benchmarks/magic_gamma/test.csv",
        "target": "label",
        "header": True,
    },
}


RAW_RUN_COLUMNS = [
    "config_id",
    "run_id",
    "algorithm_name",
    "dataset_name",
    "requested_training_set_size",
    "training_set_size",
    "validation_set_size",
    "validation_adaptive",
    "generations",
    "population_size",
    "mutation_rate",
    "crossover_rate",
    "seed",
    "svm_kernel",
    "svm_C",
    "svm_gamma",
    "svm_class_weight",
    "training_time_sec",
    "selected_model_fit_time_sec",
    "validation_accuracy",
    "test_accuracy",
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "validation_f1",
    "test_f1",
    "num_support_vectors",
    "selected_cycle",
    "confusion_matrix_validation",
    "confusion_matrix_test",
    "config_path",
    "run_json_path",
]


AGGREGATE_COLUMNS = [
    "config_id",
    "algorithm_name",
    "dataset_name",
    "requested_training_set_size",
    "validation_adaptive",
    "runs_count",
    "mean_test_accuracy",
    "std_test_accuracy",
    "min_test_accuracy",
    "max_test_accuracy",
    "mean_validation_accuracy",
    "std_validation_accuracy",
    "mean_training_time",
]


CONFIG_COLUMNS = [
    "config_id",
    "algorithm_name",
    "dataset_name",
    "requested_training_set_size",
    "validation_set_size",
    "validation_adaptive",
    "runs",
    "generations",
    "population_size",
    "mutation_rate",
    "crossover_rate",
    "svm_kernel",
    "svm_C",
    "svm_gamma",
    "svm_class_weight",
    "dag_cycles",
    "dag_multiplier",
    "config_path",
]


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray | None
    y_test: np.ndarray | None


@dataclass(frozen=True)
class ExperimentSettings:
    dataset_name: str
    algorithms: list[str]
    train_sizes: list[int]
    runs: int
    fixed_validation: bool
    output_dir: Path
    base_seed: int
    metric: str
    generations: int
    population_size: int
    mutation_rate: float
    crossover_rate: float
    tournament_k: int
    elitism: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    dag_cycles: int
    dag_multiplier: float
    svm_cfg: SVMConfig


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "tak"}:
        return True
    if normalized in {"0", "false", "no", "n", "nie"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "dataset"


def _config_identity_payload(
    algorithm: str,
    train_size: int | None,
    settings: ExperimentSettings,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "algorithm_name": algorithm,
        "dataset_name": settings.dataset_name,
        "requested_training_set_size": train_size,
        "validation_adaptive": False,
        "metric": settings.metric,
        "svm": asdict(settings.svm_cfg),
    }
    if algorithm != "full_svm":
        payload.update(
            {
                "generations": settings.generations,
                "population_size": settings.population_size,
                "mutation_rate": settings.mutation_rate,
                "crossover_rate": settings.crossover_rate,
                "tournament_k": settings.tournament_k,
                "elitism": settings.elitism,
                "early_stopping_patience": settings.early_stopping_patience,
                "early_stopping_min_delta": settings.early_stopping_min_delta,
            }
        )
    if algorithm == "dag_svm":
        payload.update(
            {
                "dag_cycles": settings.dag_cycles,
                "dag_multiplier": settings.dag_multiplier,
            }
        )
    return payload


def _config_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]


def _resolve_dataset_args(args: argparse.Namespace) -> dict[str, Any]:
    registry = DATASET_REGISTRY.get(args.dataset, {})
    dataset_name = args.dataset_name or registry.get("dataset_name") or args.dataset
    train_csv = args.train_csv or registry.get("train_csv")
    validation_csv = args.validation_csv or registry.get("validation_csv")
    test_csv = args.test_csv or registry.get("test_csv")
    target = args.target or registry.get("target") or "label"
    header = args.header if args.header is not None else registry.get("header")

    if not train_csv or not validation_csv:
        raise SystemExit("Provide --train-csv and --validation-csv, or use a known --dataset.")

    return {
        "dataset_name": dataset_name,
        "train_csv": train_csv,
        "validation_csv": validation_csv,
        "test_csv": test_csv,
        "target": target,
        "header": header,
    }


def _load_split(path: str, target: str, header: bool | None) -> tuple[np.ndarray, np.ndarray]:
    df = load_dataset(path, header=header)
    target_col = target
    if target_col not in df.columns:
        guessed = _guess_target_name(df)
        if guessed is not None:
            target_col = guessed
    X, y = split_features_labels(df, target_col)
    return X.to_numpy(dtype=float), y.to_numpy()


def load_dataset_bundle(dataset_cfg: dict[str, Any]) -> DatasetBundle:
    train_path = str(REPO_ROOT / dataset_cfg["train_csv"]) if not Path(dataset_cfg["train_csv"]).is_absolute() else str(dataset_cfg["train_csv"])
    validation_path = str(REPO_ROOT / dataset_cfg["validation_csv"]) if not Path(dataset_cfg["validation_csv"]).is_absolute() else str(dataset_cfg["validation_csv"])
    test_csv = dataset_cfg.get("test_csv")
    test_path = None if not test_csv else str(REPO_ROOT / test_csv) if not Path(test_csv).is_absolute() else str(test_csv)

    X_train, y_train = _load_split(train_path, dataset_cfg["target"], dataset_cfg.get("header"))
    X_validation, y_validation = _load_split(validation_path, dataset_cfg["target"], dataset_cfg.get("header"))
    X_test, y_test = (None, None)
    if test_path:
        X_test, y_test = _load_split(test_path, dataset_cfg["target"], dataset_cfg.get("header"))

    return DatasetBundle(
        dataset_name=str(dataset_cfg["dataset_name"]),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
    )


def _normalize_algorithms(values: list[str]) -> list[str]:
    aliases = {
        "full": "full_svm",
        "full_svm": "full_svm",
        "svm": "full_svm",
        "ga": "ga_svm",
        "gasvm": "ga_svm",
        "ga_svm": "ga_svm",
        "dag": "dag_svm",
        "dagasvm": "dag_svm",
        "dag_svm": "dag_svm",
    }
    normalized = []
    for value in values:
        key = value.strip().lower()
        if key not in aliases:
            raise SystemExit(f"Unsupported algorithm: {value}")
        normalized.append(aliases[key])
    return list(dict.fromkeys(normalized))


def _ga_config(settings: ExperimentSettings) -> GAConfig:
    return GAConfig(
        population_size=settings.population_size,
        generations=settings.generations,
        tournament_k=settings.tournament_k,
        crossover_rate=settings.crossover_rate,
        mutation_rate=settings.mutation_rate,
        elitism=settings.elitism,
        early_stopping=EarlyStopping(
            patience=settings.early_stopping_patience,
            min_delta=settings.early_stopping_min_delta,
        ),
    )


def _classes_count(y: np.ndarray) -> int:
    return int(np.unique(np.asarray(y)).size)


def _k_for_requested_size(train_size: int, y_train: np.ndarray) -> int:
    class_count = _classes_count(y_train)
    if int(train_size) % class_count != 0:
        raise ValueError(
            f"train size {train_size} must be divisible by the number of classes ({class_count})"
        )
    k = int(train_size) // class_count
    if k < 1:
        raise ValueError(f"train size {train_size} is too small for {class_count} classes")
    _, counts = np.unique(np.asarray(y_train), return_counts=True)
    min_class_count = int(counts.min())
    if k > min_class_count:
        raise ValueError(
            f"train size {train_size} requires k={k} per class, but the smallest class has "
            f"only {min_class_count} samples"
        )
    return k


def _dag_k_schedule(target_k: int, cycles: int, multiplier: float) -> list[int]:
    cycles = max(1, int(cycles))
    if cycles == 1:
        return [int(target_k)]

    start = max(1, int(round(target_k / (float(multiplier) ** (cycles - 1)))))
    values = [start]
    current = start
    for _ in range(cycles - 1):
        current = min(target_k, max(current + 1, int(round(current * multiplier))))
        values.append(current)
    values[-1] = target_k

    deduped = []
    for value in values:
        if not deduped or value != deduped[-1]:
            deduped.append(int(value))
    return deduped


def _evaluate_model(clf, X, y, metric: str) -> dict[str, Any] | None:
    if X is None or y is None:
        return None
    result = evaluate_fitted_svm(clf, X, y, metric=metric)
    pred = clf.predict(X)
    labels = np.unique(np.concatenate([np.asarray(y), np.asarray(clf.classes_)]))
    result["confusion_matrix"] = confusion_matrix(y, pred, labels=labels).tolist()
    result["labels"] = labels.tolist()
    return result


def _finalize_run(
    *,
    run_id: str,
    config_id: str,
    algorithm_name: str,
    requested_train_size: int | None,
    selected_cycle: int | None,
    seed: int | None,
    selected_mask: np.ndarray | None,
    selected_model_fit_time: float,
    start_time: float,
    dataset: DatasetBundle,
    settings: ExperimentSettings,
    config_path: Path,
    run_json_path: Path,
    clf,
) -> dict[str, Any]:
    validation_eval = _evaluate_model(clf, dataset.X_validation, dataset.y_validation, settings.metric)
    test_eval = _evaluate_model(clf, dataset.X_test, dataset.y_test, settings.metric)
    training_size = int(dataset.X_train.shape[0]) if selected_mask is None else int(selected_mask.sum())
    training_time_sec = float(time.perf_counter() - start_time)
    validation_metrics = validation_eval["all"] if validation_eval else {}
    test_metrics = test_eval["all"] if test_eval else {}
    num_support_vectors = int((validation_eval or {}).get("num_support_vectors") or 0)

    row = {
        "config_id": config_id,
        "run_id": run_id,
        "algorithm_name": algorithm_name,
        "dataset_name": dataset.dataset_name,
        "requested_training_set_size": requested_train_size,
        "training_set_size": training_size,
        "validation_set_size": int(dataset.X_validation.shape[0]),
        "validation_adaptive": False,
        "generations": settings.generations if algorithm_name != "full_svm" else None,
        "population_size": settings.population_size if algorithm_name != "full_svm" else None,
        "mutation_rate": settings.mutation_rate if algorithm_name != "full_svm" else None,
        "crossover_rate": settings.crossover_rate if algorithm_name != "full_svm" else None,
        "seed": seed,
        "svm_kernel": settings.svm_cfg.kernel,
        "svm_C": settings.svm_cfg.C,
        "svm_gamma": settings.svm_cfg.gamma,
        "svm_class_weight": settings.svm_cfg.class_weight,
        "training_time_sec": training_time_sec,
        "selected_model_fit_time_sec": float(selected_model_fit_time),
        "validation_accuracy": validation_metrics.get("accuracy"),
        "test_accuracy": test_metrics.get("accuracy"),
        "validation_balanced_accuracy": validation_metrics.get("balanced_accuracy"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "validation_f1": validation_metrics.get("f1"),
        "test_f1": test_metrics.get("f1"),
        "num_support_vectors": num_support_vectors,
        "selected_cycle": selected_cycle,
        "confusion_matrix_validation": json.dumps(validation_eval["confusion_matrix"]) if validation_eval else None,
        "confusion_matrix_test": json.dumps(test_eval["confusion_matrix"]) if test_eval else None,
        "config_path": str(config_path),
        "run_json_path": str(run_json_path),
    }

    _write_json(
        run_json_path,
        {
            "row": row,
            "validation": validation_eval,
            "test": test_eval,
            "selected_train_indices": None if selected_mask is None else np.flatnonzero(selected_mask).tolist(),
        },
    )
    return row


def run_full_svm(config_id: str, dataset: DatasetBundle, settings: ExperimentSettings, config_path: Path) -> dict[str, Any]:
    run_id = f"{config_id}__run_001"
    run_json_path = settings.output_dir / "raw_runs" / f"{run_id}.json"
    start = time.perf_counter()
    clf, fit_meta = fit_svm_model(dataset.X_train, dataset.y_train, settings.svm_cfg, return_metadata=True)
    row = _finalize_run(
        run_id=run_id,
        config_id=config_id,
        algorithm_name="full_svm",
        requested_train_size=int(dataset.X_train.shape[0]),
        selected_cycle=None,
        seed=settings.base_seed,
        selected_mask=None,
        selected_model_fit_time=float(fit_meta["fit_time"]),
        start_time=start,
        dataset=dataset,
        settings=settings,
        config_path=config_path,
        run_json_path=run_json_path,
        clf=clf,
    )
    return row


def run_ga_svm(
    config_id: str,
    requested_train_size: int,
    seed: int,
    dataset: DatasetBundle,
    settings: ExperimentSettings,
    config_path: Path,
) -> dict[str, Any]:
    run_id = f"{config_id}__seed_{seed}"
    run_json_path = settings.output_dir / "raw_runs" / f"{run_id}.json"
    k_per_class = _k_for_requested_size(requested_train_size, dataset.y_train)
    fcfg = FitnessConfig(metric=settings.metric, class_imbalance_penalty=0.0)
    start = time.perf_counter()
    set_global_seed(seed)
    _, best_mask = run_ga_k_per_class(
        X_pool=dataset.X_train,
        y_pool=dataset.y_train,
        X_val=dataset.X_validation,
        y_val=dataset.y_validation,
        kcfg=KPerClassConfig(k_per_class=k_per_class, random_state=seed),
        svm_cfg=settings.svm_cfg,
        fcfg=fcfg,
        gcfg=_ga_config(settings),
        seed=seed,
    )
    clf, fit_meta = fit_svm_model(dataset.X_train[best_mask], dataset.y_train[best_mask], settings.svm_cfg, return_metadata=True)
    row = _finalize_run(
        run_id=run_id,
        config_id=config_id,
        algorithm_name="ga_svm",
        requested_train_size=requested_train_size,
        selected_cycle=1,
        seed=seed,
        selected_mask=best_mask,
        selected_model_fit_time=float(fit_meta["fit_time"]),
        start_time=start,
        dataset=dataset,
        settings=settings,
        config_path=config_path,
        run_json_path=run_json_path,
        clf=clf,
    )
    return row


def run_dag_svm(
    config_id: str,
    requested_train_size: int,
    seed: int,
    dataset: DatasetBundle,
    settings: ExperimentSettings,
    config_path: Path,
) -> dict[str, Any]:
    run_id = f"{config_id}__seed_{seed}"
    run_json_path = settings.output_dir / "raw_runs" / f"{run_id}.json"
    target_k = _k_for_requested_size(requested_train_size, dataset.y_train)
    k_values = _dag_k_schedule(target_k, settings.dag_cycles, settings.dag_multiplier)
    fcfg = FitnessConfig(metric=settings.metric, class_imbalance_penalty=0.0)
    gcfg = _ga_config(settings)

    start = time.perf_counter()
    set_global_seed(seed)
    best_score = -np.inf
    best_cycle = None
    best_mask = None
    cycle_summaries = []

    for cycle_index, k_per_class in enumerate(k_values, start=1):
        _, candidate_mask = run_ga_k_per_class(
            X_pool=dataset.X_train,
            y_pool=dataset.y_train,
            X_val=dataset.X_validation,
            y_val=dataset.y_validation,
            kcfg=KPerClassConfig(k_per_class=k_per_class, random_state=seed + cycle_index - 1),
            svm_cfg=settings.svm_cfg,
            fcfg=fcfg,
            gcfg=gcfg,
            seed=seed + cycle_index - 1,
        )
        clf = fit_svm_model(dataset.X_train[candidate_mask], dataset.y_train[candidate_mask], settings.svm_cfg)
        validation_eval = _evaluate_model(clf, dataset.X_validation, dataset.y_validation, settings.metric)
        score = float((validation_eval or {}).get("metric", -np.inf))
        cycle_summaries.append(
            {
                "cycle": cycle_index,
                "seed": seed + cycle_index - 1,
                "k_per_class": k_per_class,
                "training_set_size": int(candidate_mask.sum()),
                "validation_metric": score,
                "validation_accuracy": None if validation_eval is None else validation_eval["all"]["accuracy"],
            }
        )
        if score > best_score:
            best_score = score
            best_cycle = cycle_index
            best_mask = candidate_mask.copy()

    if best_mask is None or best_cycle is None:
        raise RuntimeError(f"No DAG-SVM candidate was evaluated for {config_id}.")

    clf, fit_meta = fit_svm_model(dataset.X_train[best_mask], dataset.y_train[best_mask], settings.svm_cfg, return_metadata=True)
    row = _finalize_run(
        run_id=run_id,
        config_id=config_id,
        algorithm_name="dag_svm",
        requested_train_size=requested_train_size,
        selected_cycle=best_cycle,
        seed=seed,
        selected_mask=best_mask,
        selected_model_fit_time=float(fit_meta["fit_time"]),
        start_time=start,
        dataset=dataset,
        settings=settings,
        config_path=config_path,
        run_json_path=run_json_path,
        clf=clf,
    )

    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    payload["dag_cycles"] = cycle_summaries
    _write_json(run_json_path, payload)
    return row


def _config_id(algorithm: str, train_size: int | None, settings: ExperimentSettings) -> str:
    fingerprint = _config_fingerprint(_config_identity_payload(algorithm, train_size, settings))
    dataset_slug = _slug(settings.dataset_name)
    if algorithm == "full_svm":
        return f"{dataset_slug}__full_svm__full_train__cfg_{fingerprint}"
    return f"{dataset_slug}__{algorithm}__train_{train_size}__cfg_{fingerprint}"


def _config_payload(
    config_id: str,
    algorithm: str,
    requested_train_size: int | None,
    dataset: DatasetBundle,
    settings: ExperimentSettings,
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "algorithm_name": algorithm,
        "dataset_name": dataset.dataset_name,
        "requested_training_set_size": requested_train_size,
        "validation_set_size": int(dataset.X_validation.shape[0]),
        "validation_adaptive": False,
        "runs": 1 if algorithm == "full_svm" else settings.runs,
        "metric": settings.metric,
        "generations": None if algorithm == "full_svm" else settings.generations,
        "population_size": None if algorithm == "full_svm" else settings.population_size,
        "mutation_rate": None if algorithm == "full_svm" else settings.mutation_rate,
        "crossover_rate": None if algorithm == "full_svm" else settings.crossover_rate,
        "tournament_k": None if algorithm == "full_svm" else settings.tournament_k,
        "elitism": None if algorithm == "full_svm" else settings.elitism,
        "dag_cycles": settings.dag_cycles if algorithm == "dag_svm" else None,
        "dag_multiplier": settings.dag_multiplier if algorithm == "dag_svm" else None,
        "svm": asdict(settings.svm_cfg),
    }


def _config_row(payload: dict[str, Any], config_path: Path, settings: ExperimentSettings) -> dict[str, Any]:
    svm = payload["svm"]
    return {
        "config_id": payload["config_id"],
        "algorithm_name": payload["algorithm_name"],
        "dataset_name": payload["dataset_name"],
        "requested_training_set_size": payload["requested_training_set_size"],
        "validation_set_size": payload["validation_set_size"],
        "validation_adaptive": payload["validation_adaptive"],
        "runs": payload["runs"],
        "generations": payload["generations"],
        "population_size": payload["population_size"],
        "mutation_rate": payload["mutation_rate"],
        "crossover_rate": payload["crossover_rate"],
        "svm_kernel": svm["kernel"],
        "svm_C": svm["C"],
        "svm_gamma": svm["gamma"],
        "svm_class_weight": svm["class_weight"],
        "dag_cycles": payload["dag_cycles"],
        "dag_multiplier": payload["dag_multiplier"],
        "config_path": str(config_path),
    }


def _aggregate(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(raw_rows)
    rows = []
    for config_id, group in frame.groupby("config_id", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "config_id": config_id,
                "algorithm_name": first["algorithm_name"],
                "dataset_name": first["dataset_name"],
                "requested_training_set_size": first["requested_training_set_size"],
                "validation_adaptive": bool(first["validation_adaptive"]),
                "runs_count": int(len(group)),
                "mean_test_accuracy": float(group["test_accuracy"].mean()),
                "std_test_accuracy": float(group["test_accuracy"].std(ddof=0)),
                "min_test_accuracy": float(group["test_accuracy"].min()),
                "max_test_accuracy": float(group["test_accuracy"].max()),
                "mean_validation_accuracy": float(group["validation_accuracy"].mean()),
                "std_validation_accuracy": float(group["validation_accuracy"].std(ddof=0)),
                "mean_training_time": float(group["training_time_sec"].mean()),
            }
        )
    return rows


def _validate_cli_args(args: argparse.Namespace, algorithms: list[str]) -> None:
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1.")
    if args.generations < 0:
        raise SystemExit("--generations must be >= 0.")
    if args.population_size < 2:
        raise SystemExit("--population-size must be >= 2.")
    if args.tournament_k < 1 or args.tournament_k > args.population_size:
        raise SystemExit("--tournament-k must be in range [1, population-size].")
    if args.elitism < 0 or args.elitism > args.population_size:
        raise SystemExit("--elitism must be in range [0, population-size].")
    if not 0.0 <= args.mutation_rate <= 1.0:
        raise SystemExit("--mutation-rate must be in range [0, 1].")
    if not 0.0 <= args.crossover_rate <= 1.0:
        raise SystemExit("--crossover-rate must be in range [0, 1].")
    if any(algorithm in {"ga_svm", "dag_svm"} for algorithm in algorithms):
        if any(size < 1 for size in args.train_sizes):
            raise SystemExit("--train-sizes must contain only positive integers.")
    if "dag_svm" in algorithms:
        if args.dag_cycles < 1:
            raise SystemExit("--dag-cycles must be >= 1.")
        if args.dag_cycles > 1 and args.dag_multiplier <= 1.0:
            raise SystemExit("--dag-multiplier must be > 1.0 for increasing DAG schedules.")


def run_experiment_suite(dataset: DatasetBundle, settings: ExperimentSettings) -> dict[str, Path]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "configs").mkdir(exist_ok=True)
    (settings.output_dir / "raw_runs").mkdir(exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []

    for algorithm in settings.algorithms:
        requested_sizes = [None] if algorithm == "full_svm" else settings.train_sizes
        for requested_train_size in requested_sizes:
            config_id = _config_id(algorithm, requested_train_size, settings)
            config_path = settings.output_dir / "configs" / f"{config_id}.yaml"
            payload = _config_payload(config_id, algorithm, requested_train_size, dataset, settings)
            _write_yaml(config_path, payload)
            config_rows.append(_config_row(payload, config_path, settings))

            if algorithm == "full_svm":
                raw_rows.append(run_full_svm(config_id, dataset, settings, config_path))
                continue

            for run_index in range(settings.runs):
                seed = settings.base_seed + run_index
                if algorithm == "ga_svm":
                    raw_rows.append(run_ga_svm(config_id, int(requested_train_size), seed, dataset, settings, config_path))
                elif algorithm == "dag_svm":
                    raw_rows.append(run_dag_svm(config_id, int(requested_train_size), seed, dataset, settings, config_path))
                else:
                    raise RuntimeError(f"Unsupported algorithm: {algorithm}")

    aggregate_rows = _aggregate(raw_rows)
    config_ids = [row["config_id"] for row in config_rows]
    run_ids = [row["run_id"] for row in raw_rows]

    raw_runs_csv = settings.output_dir / "raw_runs.csv"
    aggregate_csv = settings.output_dir / "aggregated_results.csv"
    configs_csv = settings.output_dir / "configs.csv"
    manifest_json = settings.output_dir / "manifest.json"

    _write_csv(raw_runs_csv, raw_rows, RAW_RUN_COLUMNS)
    _write_csv(aggregate_csv, aggregate_rows, AGGREGATE_COLUMNS)
    _write_csv(configs_csv, config_rows, CONFIG_COLUMNS)
    _write_json(
        manifest_json,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_name": dataset.dataset_name,
            "output_dir": settings.output_dir,
            "raw_runs_csv": raw_runs_csv,
            "aggregated_results_csv": aggregate_csv,
            "configs_csv": configs_csv,
            "validation_adaptive": False,
            "configs_count": len(config_rows),
            "raw_runs_count": len(raw_rows),
            "aggregated_configs_count": len(aggregate_rows),
            "config_ids": config_ids,
            "run_ids": run_ids,
            "settings": {
                **asdict(settings),
                "output_dir": str(settings.output_dir),
                "svm_cfg": asdict(settings.svm_cfg),
            },
        },
    )

    return {
        "raw_runs_csv": raw_runs_csv,
        "aggregated_results_csv": aggregate_csv,
        "configs_csv": configs_csv,
        "manifest_json": manifest_json,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-validation Full SVM / GA-SVM / DAG-SVM experiment suites.")
    parser.add_argument("--dataset", default="chessboard", help="Dataset key, e.g. chessboard or magic_gamma.")
    parser.add_argument("--dataset-name", default=None, help="Display name written to results.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--validation-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--header", type=_bool_arg, default=None)
    parser.add_argument("--algorithms", nargs="+", default=["full_svm", "ga_svm", "dag_svm"])
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[10, 20, 40, 80])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--fixed-validation", type=_bool_arg, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/chessboard_baseline"))
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "balanced_accuracy", "f1"])
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--dag-cycles", type=int, default=3)
    parser.add_argument("--dag-multiplier", type=float, default=1.5)
    parser.add_argument("--svm-kernel", default="rbf")
    parser.add_argument("--svm-C", type=float, default=10.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-class-weight", default="balanced")
    parser.add_argument("--svm-probability", type=_bool_arg, default=False)
    return parser


def _parse_gamma(value: str) -> str | float:
    if value in {"scale", "auto"}:
        return value
    try:
        return float(value)
    except ValueError:
        return value


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    algorithms = _normalize_algorithms(args.algorithms)

    if not args.fixed_validation:
        raise SystemExit("This runner currently supports only --fixed-validation true.")
    _validate_cli_args(args, algorithms)

    dataset_cfg = _resolve_dataset_args(args)
    dataset = load_dataset_bundle(dataset_cfg)
    settings = ExperimentSettings(
        dataset_name=dataset.dataset_name,
        algorithms=algorithms,
        train_sizes=args.train_sizes,
        runs=int(args.runs),
        fixed_validation=True,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        metric=args.metric,
        generations=args.generations,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        tournament_k=args.tournament_k,
        elitism=args.elitism,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        dag_cycles=args.dag_cycles,
        dag_multiplier=args.dag_multiplier,
        svm_cfg=SVMConfig(
            kernel=args.svm_kernel,
            C=args.svm_C,
            gamma=_parse_gamma(str(args.svm_gamma)),
            class_weight=None if str(args.svm_class_weight).lower() == "none" else args.svm_class_weight,
            probability=args.svm_probability,
        ),
    )

    paths = run_experiment_suite(dataset, settings)
    print(f"raw_runs={paths['raw_runs_csv'].resolve()}")
    print(f"aggregated_results={paths['aggregated_results_csv'].resolve()}")
    print(f"configs={paths['configs_csv'].resolve()}")
    print(f"manifest={paths['manifest_json'].resolve()}")


if __name__ == "__main__":
    main()
