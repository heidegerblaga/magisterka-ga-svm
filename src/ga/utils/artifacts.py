from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ga.encoding import decode
from ga.models.svm import train_eval_svm

try:  # Unix
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None

try:  # Windows
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None


def _lock_file(handle):
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return "fcntl"
    if os.name == "nt" and msvcrt is not None:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return "msvcrt"
        except Exception:
            return None
    return None


def _unlock_file(handle, method: str | None) -> None:
    if method == "fcntl" and fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if method == "msvcrt" and msvcrt is not None:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug or "run"


def _build_run_id(experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{_slugify(experiment_name)}"


def ensure_artifact_layout(
    outdir: str = "artifacts",
    experiment_name: str = "exp",
    run_id: str | None = None,
) -> dict[str, Path | str]:
    root = Path(outdir)
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    base_run_id = run_id or _build_run_id(experiment_name)
    suffix = 1
    while True:
        candidate = runs_root / base_run_id if suffix == 1 else runs_root / f"{base_run_id}_{suffix:02d}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            run_root = candidate
            break
        except FileExistsError:
            suffix += 1

    layout = {
        "root": root,
        "runs_root": runs_root,
        "run_root": run_root,
        "metrics": run_root / "metrics",
        "checkpoints": run_root / "checkpoints",
        "plots": run_root / "plots",
        "logs": run_root / "logs",
        "run_id": run_root.name,
        "index_csv": root / "run_history.csv",
        "history_table_csv": root / "history_table.csv",
        "latest_json": root / "latest_run.json",
    }

    for key, path in layout.items():
        if key == "run_id" or not isinstance(path, Path):
            continue
        if key == "run_root":
            continue
        if key in {"index_csv", "history_table_csv", "latest_json"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            continue
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: str | Path, payload) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv_rows(
    path: str | Path,
    rows: list[dict],
    fieldnames: list[str] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
        else:
            output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(fieldnames or rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {field: _json_ready(row.get(field)) for field in fieldnames}
            writer.writerow(normalized)


def append_csv_row(
    path: str | Path,
    row: dict,
    fieldnames: list[str] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_row = {str(key): _json_ready(value) for key, value in row.items()}
    explicit_fieldnames = fieldnames is not None
    fieldnames = list(fieldnames or normalized_row.keys())
    with output_path.open("a+", encoding="utf-8", newline="") as handle:
        lock_method = _lock_file(handle)
        try:
            handle.seek(0, os.SEEK_END)
            has_rows = handle.tell() > 0

            if has_rows and not explicit_fieldnames:
                handle.seek(0)
                reader = csv.DictReader(handle)
                existing_fieldnames = list(reader.fieldnames or [])
                for key in fieldnames:
                    if key not in existing_fieldnames:
                        existing_fieldnames.append(key)
                fieldnames = existing_fieldnames

            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not has_rows:
                writer.writeheader()
            writer.writerow({field: normalized_row.get(field) for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_file(handle, lock_method)


def write_latest_run_pointer(path: str | Path, payload: dict) -> None:
    write_json(path, payload)


def save_mask(path: str | Path, mask) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(mask, dtype=bool))


def class_distribution(y) -> dict[str, int]:
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def summarize_split(X, y):
    if X is None or y is None:
        return {
            "rows": 0,
            "features": None,
            "class_distribution": {},
        }
    return {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]) if len(X.shape) > 1 else 1,
        "class_distribution": class_distribution(y),
    }


def summarize_history(history, population_size: int, evaluation_label: str):
    history = np.asarray(history, dtype=float)
    return {
        "history": [float(value) for value in history],
        "population_size": int(population_size),
        "generations_recorded": int(history.size),
        "generations_executed": max(0, int(history.size) - 1),
        "best_final_fitness": float(history[-1]) if history.size else None,
        evaluation_label: int(population_size * history.size),
    }


def summarize_run_manifest_entry(
    run_id: str,
    experiment_name: str,
    run_root: str | Path,
    log_path: str | Path,
    summary_path: str | Path,
    seed: int,
    best_val_metric,
    best_cycle,
    best_k_per_class,
    duration_sec: float,
) -> dict:
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "run_root": str(run_root),
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "seed": seed,
        "best_val_metric": best_val_metric,
        "best_cycle": best_cycle,
        "best_k_per_class": best_k_per_class,
        "duration_sec": duration_sec,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_ga_artifacts(history, best_mask, X, y, svm_cfg, metric, log, outdir="artifacts"):
    """Backward-compatible helper for persisting a single GA run."""
    layout = ensure_artifact_layout(outdir)
    write_json(Path(layout["metrics"]) / "history.json", [float(x) for x in history])
    save_mask(Path(layout["checkpoints"]) / "best_mask.npy", best_mask)

    X_tr, y_tr, X_va, y_va = decode(best_mask, X, y)
    best_score = train_eval_svm(X_tr, y_tr, X_va, y_va, svm_cfg, metric=metric)

    log.info(f"GA finished. Best fitness (gen best): {history[-1]:.4f}")
    log.info(f"Best-split SVM ({svm_cfg.kernel}) {metric}: {best_score:.4f}")

    return best_score
