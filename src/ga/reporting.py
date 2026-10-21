from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ga.data.loader import _guess_target_name, load_dataset, prepare_train_val, split_features_labels
from ga.models.svm import SVMConfig, evaluate_fitted_svm, fit_svm_model
from ga.utils.artifacts import write_csv_rows
from ga.utils.random import limit_dataset


HISTORY_TABLE_COLUMNS = [
    "dataset",
    "algorithm",
    "training_set_selection",
    "validation_set_selection",
    "training_time_sec",
    "training_set_size",
    "validation_set_size",
    "num_support_vectors",
    "full_train_precision",
    "full_train_recall",
    "full_train_f1",
    "full_validation_precision",
    "full_validation_recall",
    "full_validation_f1",
    "test_precision",
    "test_recall",
    "test_f1",
    "run_id",
    "created_at",
    "summary_path",
]


def algorithm_metadata(mode: str) -> dict[str, str]:
    mapping = {
        "full_svm": {
            "algorithm": "Full SVM",
            "training_set_selection": "No",
            "validation_set_selection": "No",
        },
        "gasvm": {
            "algorithm": "GASVM",
            "training_set_selection": "Yes",
            "validation_set_selection": "No",
        },
        "dagasvm": {
            "algorithm": "DAGASVM",
            "training_set_selection": "Yes",
            "validation_set_selection": "No",
        },
        "dagasvm_valga": {
            "algorithm": "DAGASVM (phased)",
            "training_set_selection": "Yes",
            "validation_set_selection": "Yes",
        },
        "dagasvm_phase": {
            "algorithm": "DAGASVM (phased)",
            "training_set_selection": "Yes",
            "validation_set_selection": "Yes",
        },
        "dagasvm_phase_switch": {
            "algorithm": "DAGASVM (T'/V' phase switch)",
            "training_set_selection": "Yes",
            "validation_set_selection": "Yes",
        },
    }
    return mapping.get(mode, mapping["dagasvm"])


def normalize_report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in HISTORY_TABLE_COLUMNS}


def write_history_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    normalized_rows = [normalize_report_row(row) for row in rows]
    write_csv_rows(path, normalized_rows, fieldnames=HISTORY_TABLE_COLUMNS)


def _read_test_dataset(
    test_csv: str,
    *,
    target_col: str,
    header: bool | None = None,
    target_candidates: list[str] | None = None,
):
    if not test_csv:
        return None, None

    df_test = load_dataset(test_csv, header=header)
    tgt = target_col
    if tgt not in df_test.columns:
        if target_candidates:
            for candidate in target_candidates:
                if candidate in df_test.columns:
                    tgt = candidate
                    break
        if tgt not in df_test.columns:
            guessed = _guess_target_name(df_test)
            if guessed is not None:
                tgt = guessed

    if tgt not in df_test.columns:
        raise ValueError(
            f"Nie mogę znaleźć kolumny celu w test_csv={test_csv}. "
            f"Kolumny: {df_test.columns.tolist()[:20]} ..."
        )

    X_test, y_test = split_features_labels(df_test, tgt)
    return X_test, y_test


def load_datasets_from_config(cfg: dict[str, Any], apply_limit: bool = True) -> dict[str, Any]:
    X_train, y_train, X_val, y_val = prepare_train_val(
        cfg["data"]["train_csv"],
        cfg["data"].get("validation_csv", ""),
        cfg["data"]["target"],
        stratify=cfg["data"].get("stratify", True),
        seed=cfg.get("seed", 42),
        header=cfg["data"].get("header", None),
        target_candidates=cfg["data"].get("target_candidates", []),
    )

    if apply_limit:
        X_train, y_train, X_val, y_val = limit_dataset(
            X_train,
            X_val,
            y_train,
            y_val,
            max_rows=cfg.get("data", {}).get("max_rows", 1000),
            seed=cfg.get("seed", 42),
        )

    X_test, y_test = _read_test_dataset(
        cfg["data"].get("test_csv", ""),
        target_col=cfg["data"]["target"],
        header=cfg["data"].get("header", None),
        target_candidates=cfg["data"].get("target_candidates", []),
    )
    return {
        "train": {"X": X_train, "y": y_train},
        "validation": {"X": X_val, "y": y_val},
        "test": {"X": X_test, "y": y_test},
    }


def infer_algorithm_mode(summary: dict[str, Any], cfg: dict[str, Any] | None = None) -> str:
    for candidate in (summary.get("algorithm_mode"), (cfg or {}).get("algorithm_mode")):
        if candidate in {"full_svm", "gasvm", "dagasvm", "dagasvm_valga", "dagasvm_phase", "dagasvm_phase_switch"}:
            return str(candidate)

    cycles = summary.get("cycles") or []
    if not cycles:
        return "full_svm"
    if any("val_ga" in cycle for cycle in cycles):
        return "dagasvm_valga"
    if any((cycle.get("daga") is not None) for cycle in cycles):
        return "dagasvm"
    return "gasvm"


def _repo_root_from_summary_path(summary_path: Path) -> Path:
    parts = summary_path.parts
    if "artifacts" in parts:
        artifacts_idx = parts.index("artifacts")
        return Path(*parts[:artifacts_idx]) if artifacts_idx > 0 else Path.cwd()
    return Path.cwd()


def _resolve_artifact_path(summary_path: Path, stored_path: str | None) -> Path | None:
    if not stored_path:
        return None

    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate

    repo_root = _repo_root_from_summary_path(summary_path)
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate

    local_candidate = summary_path.parent / candidate
    if local_candidate.exists():
        return local_candidate

    return repo_candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_summary_and_context(summary_or_path: dict[str, Any] | str | Path):
    if isinstance(summary_or_path, dict):
        return summary_or_path, None, None

    summary_path = Path(summary_or_path)
    summary = _read_json(summary_path)
    config_path = summary_path.parent / "config_snapshot.json"
    cfg = _read_json(config_path) if config_path.exists() else None
    return summary, summary_path, cfg


def _representative_train_mask_path(
    summary: dict[str, Any],
    summary_path: Path | None,
    mode: str,
) -> Path | None:
    rep = summary.get("representative_model") or {}
    stored_path = rep.get("train_mask_path")
    if stored_path:
        return _resolve_artifact_path(summary_path, stored_path) if summary_path else Path(stored_path)

    global_best = summary.get("global_best") or {}
    stored_path = global_best.get("train_mask_path")
    if stored_path:
        return _resolve_artifact_path(summary_path, stored_path) if summary_path else Path(stored_path)

    cycles = summary.get("cycles") or []
    if mode in {"gasvm", "dagasvm", "dagasvm_valga", "dagasvm_phase", "dagasvm_phase_switch"} and cycles:
        cycle_index = max(0, int((global_best.get("cycle") or 1) - 1))
        cycle = cycles[min(cycle_index, len(cycles) - 1)]
        stored_path = ((cycle.get("train_ga") or {}).get("best_train_mask_path"))
        if stored_path:
            return _resolve_artifact_path(summary_path, stored_path) if summary_path else Path(stored_path)
    return None


def _representative_validation_size(
    summary: dict[str, Any],
    mode: str,
    actual_validation_rows: int,
) -> int:
    rep = summary.get("representative_model") or {}
    validation_size = rep.get("validation_rows")
    if validation_size is not None:
        return int(validation_size)

    if mode in {"dagasvm_valga", "dagasvm_phase", "dagasvm_phase_switch"}:
        global_best = summary.get("global_best") or {}
        best_cycle = int(global_best.get("cycle") or 1)
        for cycle in summary.get("cycles") or []:
            if int(cycle.get("cycle", -1)) == best_cycle:
                current_val = cycle.get("current_val") or {}
                if current_val.get("rows") is not None:
                    return int(current_val["rows"])
    return int(actual_validation_rows)


def _dataset_name(summary: dict[str, Any], cfg: dict[str, Any] | None) -> str:
    return str(
        summary.get("dataset_name")
        or (cfg or {}).get("dataset_name")
        or summary.get("experiment_name")
        or "dataset"
    )


def _representative_training_size(summary: dict[str, Any], mode: str, default_rows: int) -> int:
    rep = summary.get("representative_model") or {}
    if rep.get("training_rows") is not None:
        return int(rep["training_rows"])

    if mode in {"gasvm", "dagasvm", "dagasvm_valga", "dagasvm_phase", "dagasvm_phase_switch"}:
        global_best = summary.get("global_best") or {}
        train_summary = global_best.get("train_summary") or {}
        if train_summary.get("rows") is not None:
            return int(train_summary["rows"])

    return int(default_rows)


def build_report_row(
    summary_or_path: dict[str, Any] | str | Path,
    cfg: dict[str, Any] | None = None,
    actual_datasets: dict[str, Any] | None = None,
    full_eval_datasets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary, summary_path, loaded_cfg = _load_summary_and_context(summary_or_path)
    cfg = cfg or loaded_cfg or {}

    if summary.get("report_row"):
        return normalize_report_row(summary["report_row"])

    mode = infer_algorithm_mode(summary, cfg)
    metadata = algorithm_metadata(mode)

    if actual_datasets is None:
        actual_datasets = load_datasets_from_config(cfg, apply_limit=True)
    if full_eval_datasets is None:
        full_eval_datasets = load_datasets_from_config(cfg, apply_limit=False)

    actual_train = actual_datasets["train"]
    actual_validation = actual_datasets["validation"]
    full_train = full_eval_datasets["train"]
    full_validation = full_eval_datasets["validation"]
    full_test = full_eval_datasets["test"]

    train_X = actual_train["X"]
    train_y = actual_train["y"]
    train_mask_path = _representative_train_mask_path(summary, summary_path, mode)
    # Błąd naprawiony: wcześniej brakowało tu "dagasvm_phase", więc dla trybu fazowego wiersz
    # raportu (a z niego ranking sweepa) liczył metryki SVM wytrenowanego na CAŁYM zbiorze
    # treningowym zamiast na reprezentatywnym T'. Efekt: stałe, niewiarygodne wartości
    # (np. identyczna liczba SV i F1 we wszystkich seedach).
    if mode in {"gasvm", "dagasvm", "dagasvm_valga", "dagasvm_phase", "dagasvm_phase_switch"}:
        if train_mask_path is None or not train_mask_path.exists():
            raise FileNotFoundError(f"Nie znaleziono maski treningowej dla summary: {summary_path}")
        train_mask = np.load(train_mask_path)
        train_X = train_X[train_mask]
        train_y = train_y[train_mask]

    svm_section = cfg.get("svm", {})
    svm_cfg = SVMConfig(
        kernel=svm_section.get("kernel", "rbf"),
        C=svm_section.get("C", 1.0),
        gamma=svm_section.get("gamma", "scale"),
        class_weight=svm_section.get("class_weight", "balanced"),
        probability=svm_section.get("probability", True),
        standardize=svm_section.get("standardize", True),
    )
    metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")
    clf, fit_meta = fit_svm_model(train_X, train_y, svm_cfg, return_metadata=True)

    train_eval = evaluate_fitted_svm(clf, full_train["X"], full_train["y"], metric=metric)
    validation_eval = evaluate_fitted_svm(
        clf, full_validation["X"], full_validation["y"], metric=metric
    )
    test_eval = None
    if full_test["X"] is not None:
        test_eval = evaluate_fitted_svm(clf, full_test["X"], full_test["y"], metric=metric)

    row = {
        "dataset": _dataset_name(summary, cfg),
        "algorithm": metadata["algorithm"],
        "training_set_selection": metadata["training_set_selection"],
        "validation_set_selection": metadata["validation_set_selection"],
        "training_time_sec": float(summary.get("experiment_duration_sec", 0.0)),
        "training_set_size": _representative_training_size(summary, mode, len(train_X)),
        "validation_set_size": _representative_validation_size(
            summary, mode, len(actual_validation["X"])
        ),
        "num_support_vectors": int(fit_meta["num_support_vectors"]),
        "full_train_precision": train_eval["all"]["precision"],
        "full_train_recall": train_eval["all"]["recall"],
        "full_train_f1": train_eval["all"]["f1"],
        "full_validation_precision": validation_eval["all"]["precision"],
        "full_validation_recall": validation_eval["all"]["recall"],
        "full_validation_f1": validation_eval["all"]["f1"],
        "test_precision": None if test_eval is None else test_eval["all"]["precision"],
        "test_recall": None if test_eval is None else test_eval["all"]["recall"],
        "test_f1": None if test_eval is None else test_eval["all"]["f1"],
        "run_id": summary.get("run_id"),
        "created_at": summary.get("created_at"),
        "summary_path": str(summary_path) if summary_path is not None else summary.get("summary_path"),
    }
    return normalize_report_row(row)


def rebuild_history_table(artifacts_root: str | Path = "artifacts") -> list[dict[str, Any]]:
    artifacts_root = Path(artifacts_root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(artifacts_root.glob("runs/*/metrics/summary.json")):
        try:
            rows.append(build_report_row(summary_path))
        except Exception:
            continue

    write_history_table(artifacts_root / "history_table.csv", rows)
    return rows
