from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from ga.reporting import infer_algorithm_mode, rebuild_history_table
from ga.sweep import SWEEP_RANKING_COLUMNS, default_objective_mode
from ga.utils.artifacts import write_csv_rows, write_json, write_latest_run_pointer


@dataclass
class RunRecord:
    run_id: str
    run_root: Path
    summary_path: Path
    config_path: Path
    experiment_name: str
    dataset: str
    algorithm_mode: str
    kernel: str
    gamma: str
    c_value: float | None
    k_per_class: int | None
    population_size: int | None
    generations: int | None
    mutation_rate: float | None
    subset_fraction: float | None
    num_cycles: int | None
    k_start: int | None
    k_multiplier: float | None
    training_set_size: float | None
    training_time_sec: float | None
    num_support_vectors: float | None
    full_validation_f1: float | None
    test_f1: float | None
    created_at: str | None
    sweep_id: str | None
    trial_id: str | None
    seed_reasons: list[str] = field(default_factory=list)
    novelty_score: float = 0.0
    quality_score: float = 0.0
    combined_score: float = 0.0
    selected_order: int | None = None
    nearest_kept_run_id: str | None = None
    nearest_kept_distance: float | None = None


CONFIG_NUMERIC_FIELDS = [
    "c_log",
    "k_per_class",
    "population_size",
    "generations",
    "mutation_rate",
    "subset_fraction",
    "num_cycles",
    "k_start",
    "k_multiplier",
]

CONFIG_CATEGORICAL_FIELDS = [
    "dataset",
    "algorithm_mode",
    "kernel",
    "gamma",
]

OUTCOME_FIELDS = [
    "train_size_log",
    "train_time_log",
    "num_support_vectors",
    "full_validation_f1",
    "test_f1",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        parsed = float(text)
        if math.isnan(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _safe_text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    text = str(value).strip()
    return text or "-"


def _normalized_dataset(
    history_row: dict[str, Any],
    summary: dict[str, Any],
    cfg: dict[str, Any],
    experiment_name: str,
) -> str:
    candidates = [
        history_row.get("dataset"),
        summary.get("dataset_name"),
        cfg.get("dataset_name"),
        summary.get("experiment_name"),
        cfg.get("experiment_name"),
    ]
    for value in candidates:
        if value is None:
            continue
        text = _safe_text(value)
        if text != "-" and not text.startswith("{"):
            return text
    return experiment_name


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _lookup_frame(frame: pd.DataFrame, key: str = "run_id") -> dict[str, dict[str, Any]]:
    if frame.empty or key not in frame.columns:
        return {}
    return {
        str(row[key]): {column: row[column] for column in frame.columns}
        for _, row in frame.iterrows()
    }


def _auto_keep_count(total_runs: int) -> int:
    return max(12, min(20, round(math.sqrt(max(total_runs, 1)) * 2)))


def _feature_payload(record: RunRecord) -> dict[str, Any]:
    return {
        "dataset": record.dataset,
        "algorithm_mode": record.algorithm_mode,
        "kernel": record.kernel,
        "gamma": record.gamma,
        "c_log": None if record.c_value in {None, 0} else math.log10(max(record.c_value, 1e-9)),
        "k_per_class": record.k_per_class,
        "population_size": record.population_size,
        "generations": record.generations,
        "mutation_rate": record.mutation_rate,
        "subset_fraction": record.subset_fraction,
        "num_cycles": record.num_cycles,
        "k_start": record.k_start,
        "k_multiplier": record.k_multiplier,
        "train_size_log": None
        if record.training_set_size in {None, 0}
        else math.log1p(max(record.training_set_size, 0.0)),
        "train_time_log": None
        if record.training_time_sec in {None, 0}
        else math.log1p(max(record.training_time_sec, 0.0)),
        "num_support_vectors": record.num_support_vectors,
        "full_validation_f1": record.full_validation_f1,
        "test_f1": record.test_f1,
    }


def _ranges(records: list[RunRecord]) -> dict[str, tuple[float, float]]:
    payloads = [_feature_payload(record) for record in records]
    result: dict[str, tuple[float, float]] = {}
    for field in CONFIG_NUMERIC_FIELDS + OUTCOME_FIELDS:
        values = [payload[field] for payload in payloads if payload[field] is not None]
        if not values:
            result[field] = (0.0, 0.0)
            continue
        result[field] = (min(values), max(values))
    return result


def _numeric_distance(a: float | None, b: float | None, low: float, high: float) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    span = high - low
    if span <= 1e-12:
        return 0.0
    return min(1.0, abs(a - b) / span)


def _categorical_distance(a: str, b: str) -> float:
    return 0.0 if a == b else 1.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _distance(a: RunRecord, b: RunRecord, field_ranges: dict[str, tuple[float, float]]) -> float:
    payload_a = _feature_payload(a)
    payload_b = _feature_payload(b)

    cfg_numeric = [
        _numeric_distance(payload_a[field], payload_b[field], *field_ranges[field])
        for field in CONFIG_NUMERIC_FIELDS
    ]
    cfg_categorical = [
        _categorical_distance(_safe_text(payload_a[field]), _safe_text(payload_b[field]))
        for field in CONFIG_CATEGORICAL_FIELDS
    ]
    outcome = [
        _numeric_distance(payload_a[field], payload_b[field], *field_ranges[field])
        for field in OUTCOME_FIELDS
    ]

    config_distance = 0.75 * _mean(cfg_numeric) + 0.25 * _mean(cfg_categorical)
    outcome_distance = _mean(outcome)
    return 0.75 * config_distance + 0.25 * outcome_distance


def _normalized_quality(record: RunRecord, scores: list[float]) -> float:
    if record.full_validation_f1 is None or not scores:
        return 0.0
    minimum = min(scores)
    maximum = max(scores)
    if maximum - minimum <= 1e-12:
        return 1.0
    return (record.full_validation_f1 - minimum) / (maximum - minimum)


def _select_seed_runs(records: list[RunRecord]) -> list[RunRecord]:
    chosen: dict[str, RunRecord] = {}

    def add(reason: str, record: RunRecord | None) -> None:
        if record is None:
            return
        if record.run_id not in chosen:
            chosen[record.run_id] = record
        chosen[record.run_id].seed_reasons.append(reason)

    def best_by(key: str, reverse: bool = True) -> RunRecord | None:
        candidates = [record for record in records if getattr(record, key) is not None]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: getattr(item, key), reverse=reverse)[0]

    add("best_validation_f1", best_by("full_validation_f1", reverse=True))
    add("best_test_f1", best_by("test_f1", reverse=True))
    add("fastest", best_by("training_time_sec", reverse=False))
    add("smallest_train", best_by("training_set_size", reverse=False))
    add("largest_train", best_by("training_set_size", reverse=True))

    for algorithm_mode in sorted({record.algorithm_mode for record in records}):
        candidates = [record for record in records if record.algorithm_mode == algorithm_mode]
        if not candidates:
            continue
        add(f"best_{algorithm_mode}", best_by_group(candidates, "full_validation_f1", reverse=True))

    for dataset in sorted({record.dataset for record in records}):
        candidates = [record for record in records if record.dataset == dataset]
        if not candidates:
            continue
        add(f"best_dataset_{dataset}", best_by_group(candidates, "full_validation_f1", reverse=True))

    return list(chosen.values())


def best_by_group(records: list[RunRecord], key: str, reverse: bool = True) -> RunRecord | None:
    candidates = [record for record in records if getattr(record, key) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: getattr(item, key), reverse=reverse)[0]


def _select_representatives(records: list[RunRecord], keep_count: int) -> tuple[list[RunRecord], list[RunRecord]]:
    if keep_count >= len(records):
        for order, record in enumerate(records, start=1):
            record.selected_order = order
            record.seed_reasons.append("kept_all")
        return records, []

    field_ranges = _ranges(records)
    quality_scores = [record.full_validation_f1 for record in records if record.full_validation_f1 is not None]
    selected = _select_seed_runs(records)
    selected_map = {record.run_id: record for record in selected}

    for record in selected:
        record.quality_score = _normalized_quality(record, quality_scores)

    selected.sort(
        key=lambda item: (
            -max(item.full_validation_f1 or -1.0, item.test_f1 or -1.0),
            item.training_time_sec if item.training_time_sec is not None else float("inf"),
        )
    )

    while len(selected) < keep_count:
        remaining = [record for record in records if record.run_id not in selected_map]
        if not remaining:
            break

        best_candidate: RunRecord | None = None
        best_score = -1.0
        for candidate in remaining:
            novelty = min(_distance(candidate, kept, field_ranges) for kept in selected)
            quality = _normalized_quality(candidate, quality_scores)
            combined = 0.75 * novelty + 0.25 * quality
            if combined > best_score:
                best_score = combined
                best_candidate = candidate
                candidate.novelty_score = novelty
                candidate.quality_score = quality
                candidate.combined_score = combined

        if best_candidate is None:
            break
        best_candidate.seed_reasons.append("greedy_diversity")
        selected.append(best_candidate)
        selected_map[best_candidate.run_id] = best_candidate

    for order, record in enumerate(selected, start=1):
        record.selected_order = order
        if record.combined_score == 0.0:
            record.combined_score = record.quality_score

    deleted: list[RunRecord] = []
    for record in records:
        if record.run_id in selected_map:
            continue
        nearest = min(selected, key=lambda kept: _distance(record, kept, field_ranges))
        record.nearest_kept_run_id = nearest.run_id
        record.nearest_kept_distance = _distance(record, nearest, field_ranges)
        deleted.append(record)

    return selected, deleted


def _load_run_records(artifacts_root: Path) -> list[RunRecord]:
    history_rows = _lookup_frame(_load_table(artifacts_root / "history_table.csv"))
    run_history_rows = _lookup_frame(_load_table(artifacts_root / "run_history.csv"))

    records: list[RunRecord] = []
    for summary_path in sorted(artifacts_root.glob("runs/*/metrics/summary.json")):
        run_root = summary_path.parents[1]
        config_path = summary_path.parent / "config_snapshot.json"
        if not config_path.exists():
            continue

        summary = _read_json(summary_path)
        cfg = _read_json(config_path)
        run_id = _safe_text(summary.get("run_id") or run_root.name)
        history_row = history_rows.get(run_id, {})
        run_history_row = run_history_rows.get(run_id, {})
        experiment_name = _safe_text(
            run_history_row.get("experiment_name")
            or summary.get("experiment_name")
            or cfg.get("experiment_name")
            or run_id
        )
        dataset = _normalized_dataset(history_row, summary, cfg, experiment_name)
        algorithm_mode = _safe_text(infer_algorithm_mode(summary, cfg))
        svm_cfg = cfg.get("svm") or {}
        ga_cfg = cfg.get("ga") or {}
        val_ga_cfg = cfg.get("val_ga") or {}
        encoding_cfg = cfg.get("encoding") or {}
        k_schedule_cfg = cfg.get("k_schedule") or {}

        records.append(
            RunRecord(
                run_id=run_id,
                run_root=run_root,
                summary_path=summary_path,
                config_path=config_path,
                experiment_name=experiment_name,
                dataset=dataset,
                algorithm_mode=algorithm_mode,
                kernel=_safe_text(svm_cfg.get("kernel", "rbf")),
                gamma=_safe_text(svm_cfg.get("gamma", "scale")),
                c_value=_safe_float(svm_cfg.get("C")),
                k_per_class=_safe_int(encoding_cfg.get("k_per_class")),
                population_size=_safe_int(ga_cfg.get("population_size")),
                generations=_safe_int(ga_cfg.get("generations")),
                mutation_rate=_safe_float(ga_cfg.get("mutation_rate")),
                subset_fraction=_safe_float(val_ga_cfg.get("subset_fraction")),
                num_cycles=_safe_int(k_schedule_cfg.get("num_cycles")),
                k_start=_safe_int(k_schedule_cfg.get("k_start")),
                k_multiplier=_safe_float(k_schedule_cfg.get("k_multiplier")),
                training_set_size=_safe_float(history_row.get("training_set_size")),
                training_time_sec=_safe_float(history_row.get("training_time_sec")),
                num_support_vectors=_safe_float(history_row.get("num_support_vectors")),
                full_validation_f1=_safe_float(history_row.get("full_validation_f1")),
                test_f1=_safe_float(history_row.get("test_f1")),
                created_at=_safe_text(history_row.get("created_at") or run_history_row.get("created_at")),
                sweep_id=_safe_text(summary.get("sweep_id")) if summary.get("sweep_id") else None,
                trial_id=_safe_text(summary.get("trial_id")) if summary.get("trial_id") else None,
            )
        )
    return records


def _manifest_rows(records: list[RunRecord], decision: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        payload = asdict(record)
        payload["run_root"] = str(record.run_root)
        payload["summary_path"] = str(record.summary_path)
        payload["config_path"] = str(record.config_path)
        payload["decision"] = decision
        payload["seed_reasons"] = ",".join(record.seed_reasons)
        rows.append(payload)
    return rows


def _write_selection_manifest(
    artifacts_root: Path,
    keep: list[RunRecord],
    delete: list[RunRecord],
    keep_count: int,
) -> Path:
    prune_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = artifacts_root / "pruning" / f"{prune_id}_novel_representatives"
    root.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "requested_keep_count": keep_count,
        "kept_runs": len(keep),
        "deleted_runs": len(delete),
        "keep_ids": [record.run_id for record in keep],
        "delete_ids": [record.run_id for record in delete],
        "kept": _manifest_rows(keep, "keep"),
        "deleted": _manifest_rows(delete, "delete"),
    }
    write_json(root / "selection.json", payload)
    write_csv_rows(root / "kept_runs.csv", payload["kept"], fieldnames=list(payload["kept"][0].keys()) if payload["kept"] else None)
    write_csv_rows(root / "deleted_runs.csv", payload["deleted"], fieldnames=list(payload["deleted"][0].keys()) if payload["deleted"] else None)
    return root


def _rewrite_run_history(artifacts_root: Path, keep_ids: set[str]) -> None:
    path = artifacts_root / "run_history.csv"
    frame = _load_table(path)
    if frame.empty or "run_id" not in frame.columns:
        return
    filtered = frame[frame["run_id"].astype(str).isin(keep_ids)].copy()
    write_csv_rows(path, filtered.to_dict(orient="records"), fieldnames=list(frame.columns))

    latest_path = artifacts_root / "latest_run.json"
    if filtered.empty:
        write_json(latest_path, {})
        return
    sorted_rows = filtered.sort_values("created_at", ascending=False, na_position="last")
    write_latest_run_pointer(latest_path, sorted_rows.iloc[0].to_dict())


def _recompute_sweep_state(state: dict[str, Any]) -> dict[str, Any]:
    trials = state.get("trials") or []
    state["num_trials"] = len(trials)
    state["completed_trials"] = sum(1 for trial in trials if trial.get("status") == "completed")
    state["failed_trials"] = sum(1 for trial in trials if trial.get("status") == "failed")
    state["running_trials"] = sum(1 for trial in trials if trial.get("status") == "running")
    state["queued_trials"] = sum(1 for trial in trials if trial.get("status") == "queued")

    objective_mode = _safe_text(state.get("objective_mode") or default_objective_mode(_safe_text(state.get("objective_name"))))
    completed = [
        trial
        for trial in trials
        if trial.get("status") == "completed" and _safe_float(trial.get("objective_value")) is not None
    ]
    if not completed:
        state["best_trial_id"] = None
        state["best_run_id"] = None
        state["best_objective_value"] = None
        return state

    reverse = objective_mode != "min"
    best_trial = sorted(completed, key=lambda trial: _safe_float(trial.get("objective_value")), reverse=reverse)[0]
    state["best_trial_id"] = best_trial.get("trial_id")
    state["best_run_id"] = best_trial.get("run_id")
    state["best_objective_value"] = _safe_float(best_trial.get("objective_value"))
    return state


def _clean_sweeps(artifacts_root: Path, keep_ids: set[str]) -> None:
    sweeps_root = artifacts_root / "sweeps"
    if not sweeps_root.exists():
        return

    for sweep_root in sweeps_root.iterdir():
        if not sweep_root.is_dir():
            continue

        state_path = sweep_root / "state.json"
        manifest_path = sweep_root / "manifest.json"
        ranking_path = sweep_root / "ranking.csv"

        removed_trial_ids: set[str] = set()
        if state_path.exists():
            state = _read_json(state_path)
            kept_trials = []
            for trial in state.get("trials") or []:
                run_id = trial.get("run_id")
                run_id_text = _safe_text(run_id) if run_id else None
                if run_id_text not in keep_ids:
                    removed_trial_ids.add(_safe_text(trial.get("trial_id")))
                    for key in ("config_path", "log_path"):
                        path_value = trial.get(key)
                        if not path_value:
                            continue
                        candidate = Path(path_value)
                        if candidate.exists():
                            candidate.unlink()
                    continue
                kept_trials.append(trial)
            state["trials"] = kept_trials
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(state_path, _recompute_sweep_state(state))

        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if removed_trial_ids:
                manifest["trials"] = [
                    trial
                    for trial in manifest.get("trials") or []
                    if _safe_text(trial.get("trial_id")) not in removed_trial_ids
                ]
                manifest["effective_trials"] = len(manifest.get("trials") or [])
                write_json(manifest_path, manifest)

        if ranking_path.exists():
            frame = _load_table(ranking_path)
            if frame.empty or "run_id" not in frame.columns:
                continue
            filtered = frame[frame["run_id"].astype(str).isin(keep_ids)].copy()
            objective_name = _safe_text(filtered["objective_name"].iloc[0]) if not filtered.empty and "objective_name" in filtered.columns else "full_validation_f1"
            objective_mode = _safe_text(filtered["objective_mode"].iloc[0]) if not filtered.empty and "objective_mode" in filtered.columns else default_objective_mode(objective_name)
            if not filtered.empty and "objective_value" in filtered.columns:
                ascending = objective_mode == "min"
                filtered = filtered.sort_values(
                    by=["objective_value", "created_at"],
                    ascending=[ascending, True if ascending else False],
                    na_position="last",
                )
                filtered["rank"] = range(1, len(filtered) + 1)
            write_csv_rows(ranking_path, filtered.to_dict(orient="records"), fieldnames=SWEEP_RANKING_COLUMNS)


def _apply_deletions(artifacts_root: Path, keep: list[RunRecord], delete: list[RunRecord]) -> None:
    keep_ids = {record.run_id for record in keep}

    for record in delete:
        if record.run_root.exists():
            shutil.rmtree(record.run_root)

    runs_root = artifacts_root / "runs"
    if runs_root.exists():
        for run_root in runs_root.iterdir():
            if not run_root.is_dir():
                continue
            if run_root.name not in keep_ids:
                shutil.rmtree(run_root)

    _rewrite_run_history(artifacts_root, keep_ids)
    rebuild_history_table(artifacts_root)
    _clean_sweeps(artifacts_root, keep_ids)


def _print_summary(keep: list[RunRecord], delete: list[RunRecord], manifest_root: Path | None) -> None:
    print(f"kept={len(keep)} deleted={len(delete)}")
    print("kept runs:")
    for record in sorted(keep, key=lambda item: item.selected_order or 999):
        reasons = ",".join(record.seed_reasons) or "-"
        print(
            f"  keep #{record.selected_order:02d} {record.run_id} "
            f"[{record.dataset} | {record.algorithm_mode} | {record.kernel}] "
            f"F1={record.full_validation_f1 or float('nan'):.4f} "
            f"time={record.training_time_sec or float('nan'):.1f}s reasons={reasons}"
        )

    print("deleted sample:")
    for record in sorted(delete, key=lambda item: (item.nearest_kept_distance or 1.0, -(item.full_validation_f1 or 0.0)))[:15]:
        print(
            f"  delete {record.run_id} -> nearest={record.nearest_kept_run_id} "
            f"dist={record.nearest_kept_distance or float('nan'):.3f} "
            f"F1={record.full_validation_f1 or float('nan'):.4f}"
        )

    if manifest_root is not None:
        print(f"manifest={manifest_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune similar runs and keep only representative ones.")
    parser.add_argument("--artifacts-root", default="artifacts", help="Path to the artifacts root.")
    parser.add_argument("--keep", type=int, default=None, help="How many representative runs should remain.")
    parser.add_argument("--apply", action="store_true", help="Actually delete redundant runs and rewrite history.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    records = _load_run_records(artifacts_root)
    if not records:
        print("No runs found.")
        return 1

    keep_count = args.keep or _auto_keep_count(len(records))
    keep, delete = _select_representatives(records, keep_count=min(keep_count, len(records)))
    manifest_root = _write_selection_manifest(artifacts_root, keep, delete, keep_count)
    _print_summary(keep, delete, manifest_root)

    if args.apply:
        _apply_deletions(artifacts_root, keep, delete)
        print("applied=true")
    else:
        print("applied=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
