from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ga.reporting import HISTORY_TABLE_COLUMNS, rebuild_history_table


PROJECT_ROOT = Path.cwd()
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
CONFIGS_ROOT = PROJECT_ROOT / "configs"


def list_config_presets() -> list[Path]:
    return sorted(CONFIGS_ROOT.glob("*.yaml"))


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def history_table_path() -> Path:
    return ARTIFACTS_ROOT / "history_table.csv"


def run_history_path() -> Path:
    return ARTIFACTS_ROOT / "run_history.csv"


def latest_run_path() -> Path:
    return ARTIFACTS_ROOT / "latest_run.json"


def ensure_history_table() -> Path:
    target = history_table_path()
    if not target.exists():
        rebuild_history_table(ARTIFACTS_ROOT)
    return target


def load_history_table() -> pd.DataFrame:
    target = ensure_history_table()
    if not target.exists() or target.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_TABLE_COLUMNS)
    return pd.read_csv(target)


def load_run_history() -> pd.DataFrame:
    target = run_history_path()
    if not target.exists() or target.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(target)


def load_history_with_metadata() -> pd.DataFrame:
    history = load_history_table()
    run_history = load_run_history()
    if history.empty or run_history.empty:
        return history
    merge_cols = ["run_id", "seed", "experiment_name", "duration_sec", "best_val_metric", "created_at"]
    available_cols = [col for col in merge_cols if col in run_history.columns]
    merged = history.merge(run_history[available_cols], on="run_id", how="left", suffixes=("", "_run"))
    if "created_at_run" in merged.columns:
        if "created_at" not in merged.columns:
            merged["created_at"] = merged["created_at_run"]
        else:
            merged["created_at"] = merged["created_at"].fillna(merged["created_at_run"])
        merged = merged.drop(columns=["created_at_run"])
    return merged


def load_latest_run() -> dict[str, Any] | None:
    target = latest_run_path()
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_summary_by_run_id(run_id: str) -> Path | None:
    latest = load_latest_run()
    if latest and latest.get("run_id") == run_id:
        candidate = Path(latest["summary_path"])
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    for candidate in ARTIFACTS_ROOT.glob("runs/*/metrics/summary.json"):
        try:
            payload = load_summary(candidate)
        except Exception:
            continue
        if payload.get("run_id") == run_id:
            return candidate
    return None
