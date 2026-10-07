from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import random
import re
from typing import Any

from ga.reporting import infer_algorithm_mode
from ga.utils.artifacts import write_csv_rows, write_json


SWEEP_RANKING_COLUMNS = [
    "rank",
    "trial_id",
    "run_id",
    "status",
    "algorithm_mode",
    "algorithm",
    "objective_name",
    "objective_mode",
    "objective_value",
    "full_validation_f1",
    "test_f1",
    "training_time_sec",
    "num_support_vectors",
    "best_val_metric",
    "seed",
    "created_at",
    "summary_path",
    "config_path",
    "log_path",
    "overrides_json",
]


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._") or "sweep"


def default_objective_mode(objective_name: str) -> str:
    if objective_name in {"training_time_sec", "num_support_vectors"}:
        return "min"
    return "max"


def estimate_search_space_size(search_space: dict[str, Any]) -> int:
    lengths = []
    for values in search_space.values():
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        if values:
            lengths.append(len(values))
    if not lengths:
        return 1
    return math.prod(lengths)


def set_nested_value(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = payload
    parts = dotted_path.split(".")
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def build_sweep_layout(
    artifacts_root: str | Path = "artifacts",
    sweep_name: str = "parallel_sweep",
    sweep_id: str | None = None,
) -> dict[str, Path | str]:
    root = Path(artifacts_root)
    sweeps_root = root / "sweeps"
    sweeps_root.mkdir(parents=True, exist_ok=True)

    base_sweep_id = sweep_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{_slugify(sweep_name)}"
    suffix = 1
    while True:
        candidate = sweeps_root / base_sweep_id if suffix == 1 else sweeps_root / f"{base_sweep_id}_{suffix:02d}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            sweep_root = candidate
            break
        except FileExistsError:
            suffix += 1

    layout = {
        "root": root,
        "sweeps_root": sweeps_root,
        "sweep_root": sweep_root,
        "sweep_id": sweep_root.name,
        "configs": sweep_root / "configs",
        "logs": sweep_root / "logs",
        "manifest_json": sweep_root / "manifest.json",
        "state_json": sweep_root / "state.json",
        "ranking_csv": sweep_root / "ranking.csv",
        "orchestrator_log": sweep_root / "logs" / f"{sweep_root.name}.log",
    }
    for key, path in layout.items():
        if key in {"root", "sweeps_root", "sweep_root", "sweep_id"} or not isinstance(path, Path):
            continue
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return layout


def _signature_config(cfg: dict[str, Any]) -> str:
    payload = deepcopy(cfg)
    for key in ["sweep_id", "trial_id", "trial_overrides"]:
        payload.pop(key, None)

    mode = payload.get("algorithm_mode", "dagasvm")
    if mode == "full_svm":
        for key in ["encoding", "ga", "val_ga", "k_schedule", "val_cumulation"]:
            payload.pop(key, None)
    elif mode == "gasvm":
        for key in ["val_ga", "k_schedule", "val_cumulation"]:
            payload.pop(key, None)

    return json.dumps(payload, sort_keys=True, default=str)


def _normalize_search_space(search_space: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    normalized: list[tuple[str, list[Any]]] = []
    for path, values in search_space.items():
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        filtered = [value for value in values if value is not None]
        if filtered:
            normalized.append((path, filtered))
    return normalized


def _apply_overrides(base_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(base_cfg)
    for path, value in overrides.items():
        set_nested_value(payload, path, value)
    return payload


def _materialize_override_candidates(
    base_cfg: dict[str, Any],
    search_items: list[tuple[str, list[Any]]],
) -> list[dict[str, Any]]:
    if not search_items:
        return [{}]

    deduped: dict[str, dict[str, Any]] = {}
    paths = [path for path, _ in search_items]
    values = [value_list for _, value_list in search_items]
    for combo in itertools.product(*values):
        overrides = {path: value for path, value in zip(paths, combo)}
        signature = _signature_config(_apply_overrides(base_cfg, overrides))
        deduped.setdefault(signature, overrides)
    return list(deduped.values())


def _sample_override_candidates(
    base_cfg: dict[str, Any],
    search_items: list[tuple[str, list[Any]]],
    num_trials: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    if not search_items:
        return [{}]

    rng = random.Random(selection_seed)
    deduped: dict[str, dict[str, Any]] = {}
    attempts = 0
    max_attempts = max(2000, num_trials * 500)
    while len(deduped) < num_trials and attempts < max_attempts:
        overrides = {path: rng.choice(values) for path, values in search_items}
        signature = _signature_config(_apply_overrides(base_cfg, overrides))
        deduped.setdefault(signature, overrides)
        attempts += 1

    if not deduped:
        return [{}]
    return list(deduped.values())


def generate_trial_plan(
    base_cfg: dict[str, Any],
    search_space: dict[str, Any],
    num_trials: int,
    selection_seed: int = 42,
    sweep_id: str | None = None,
) -> list[dict[str, Any]]:
    desired_trials = max(1, int(num_trials))
    search_items = _normalize_search_space(search_space)
    total_combinations = estimate_search_space_size({path: values for path, values in search_items})

    if total_combinations <= min(5000, desired_trials * 250):
        candidates = _materialize_override_candidates(base_cfg, search_items)
    else:
        candidates = _sample_override_candidates(base_cfg, search_items, desired_trials, selection_seed)

    rng = random.Random(selection_seed)
    if len(candidates) > desired_trials:
        candidates = rng.sample(candidates, desired_trials)

    base_name = str(base_cfg.get("experiment_name", "ga_svm"))
    dataset_name = str(base_cfg.get("dataset_name") or base_name)
    sweep_suffix = (sweep_id or "sweep")[-8:]

    trials: list[dict[str, Any]] = []
    for index, overrides in enumerate(candidates, start=1):
        trial_id = f"trial_{index:03d}"
        trial_cfg = _apply_overrides(base_cfg, overrides)
        trial_cfg["dataset_name"] = str(trial_cfg.get("dataset_name") or dataset_name)
        trial_cfg["experiment_name"] = f"{base_name}_{sweep_suffix}_{trial_id}"
        trial_cfg["sweep_id"] = sweep_id
        trial_cfg["trial_id"] = trial_id
        trial_cfg["trial_overrides"] = overrides
        trials.append({"trial_id": trial_id, "overrides": overrides, "config": trial_cfg})
    return trials


def _best_val_metric(summary: dict[str, Any]) -> Any:
    mode = infer_algorithm_mode(summary)
    if mode == "full_svm":
        return ((summary.get("baseline") or {}).get("val") or {}).get("metric")
    return ((summary.get("global_best") or {}).get("metrics_val") or {}).get("metric")


def objective_value_from_summary(summary: dict[str, Any], objective_name: str) -> Any:
    report_row = summary.get("report_row") or {}
    if objective_name in report_row:
        return report_row.get(objective_name)
    if objective_name == "best_val_metric":
        return _best_val_metric(summary)
    return summary.get(objective_name)


def build_trial_result(summary_path: str | Path, objective_name: str, objective_mode: str) -> dict[str, Any]:
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_row = summary.get("report_row") or {}
    return {
        "rank": None,
        "trial_id": summary.get("trial_id"),
        "run_id": summary.get("run_id"),
        "status": "completed",
        "algorithm_mode": summary.get("algorithm_mode"),
        "algorithm": report_row.get("algorithm"),
        "objective_name": objective_name,
        "objective_mode": objective_mode,
        "objective_value": objective_value_from_summary(summary, objective_name),
        "full_validation_f1": report_row.get("full_validation_f1"),
        "test_f1": report_row.get("test_f1"),
        "training_time_sec": report_row.get("training_time_sec"),
        "num_support_vectors": report_row.get("num_support_vectors"),
        "best_val_metric": _best_val_metric(summary),
        "seed": summary.get("seed"),
        "created_at": summary.get("created_at"),
        "summary_path": str(summary_path),
        "config_path": None,
        "log_path": summary.get("log_path"),
        "overrides_json": json.dumps(summary.get("trial_overrides") or {}, ensure_ascii=False, sort_keys=True),
    }


def rank_trial_results(rows: list[dict[str, Any]], objective_name: str, objective_mode: str) -> list[dict[str, Any]]:
    ranked = [deepcopy(row) for row in rows if row.get("objective_value") is not None]
    ranked.sort(key=lambda row: row.get("objective_value"), reverse=objective_mode == "max")
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["objective_name"] = objective_name
        row["objective_mode"] = objective_mode
    return ranked


def discover_trial_summary(artifacts_root: str | Path, sweep_id: str, trial_id: str) -> Path | None:
    runs_root = Path(artifacts_root) / "runs"
    for summary_path in sorted(runs_root.glob("*/metrics/summary.json"), reverse=True):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("sweep_id") == sweep_id and summary.get("trial_id") == trial_id:
            return summary_path
    return None


def write_sweep_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def write_sweep_state(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def write_sweep_ranking(path: str | Path, rows: list[dict[str, Any]]) -> None:
    normalized = [{column: row.get(column) for column in SWEEP_RANKING_COLUMNS} for row in rows]
    write_csv_rows(path, normalized, fieldnames=SWEEP_RANKING_COLUMNS)
