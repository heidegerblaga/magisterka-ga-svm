from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

from ga.ui.data import ARTIFACTS_ROOT, PROJECT_ROOT, load_latest_run


RUNTIME_ROOT = ARTIFACTS_ROOT / "ui_runtime"
STATE_PATH = RUNTIME_ROOT / "active_run.json"
CONFIGS_ROOT = RUNTIME_ROOT / "configs"
LOGS_ROOT = RUNTIME_ROOT / "logs"


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._") or "run"


def ensure_runtime_dirs() -> None:
    CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)


def is_process_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_active_run_state() -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    payload["running"] = is_process_running(payload.get("pid"))
    log_text = read_log_head(payload.get("console_log_path"), max_lines=20)
    log_text = "\n".join(part for part in [log_text, read_log_tail(payload.get("console_log_path"), max_lines=80)] if part)
    discovered = extract_run_context(log_text)
    payload.update({key: value for key, value in discovered.items() if value is not None})
    if payload.get("run_id"):
        artifact_log = (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / payload["run_id"]
            / "logs"
            / f"{payload['run_id']}.log"
        )
        if artifact_log.exists():
            payload["artifact_log_path"] = str(artifact_log)
            payload["live_log_path"] = str(artifact_log)
    if "live_log_path" not in payload:
        payload["live_log_path"] = payload.get("console_log_path")
    latest = load_latest_run()
    if latest is not None:
        payload["latest_run"] = latest
    return payload


def save_active_run_state(payload: dict[str, Any]) -> None:
    ensure_runtime_dirs()
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_active_run_state() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()


def _read_previous_latest_run_id() -> str | None:
    latest = load_latest_run()
    return None if latest is None else latest.get("run_id")


def write_temp_config(config_payload: dict[str, Any]) -> Path:
    ensure_runtime_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config_payload.get("experiment_name", "run")
    temp_path = CONFIGS_ROOT / f"{timestamp}_{_slugify(experiment_name)}.yaml"
    temp_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    return temp_path


def start_run(config_payload: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_dirs()
    temp_config_path = write_temp_config(config_payload)
    log_path = LOGS_ROOT / f"{temp_config_path.stem}.log"
    log_handle = log_path.open("w", encoding="utf-8")

    command = [
        sys.executable,
        "-u",
        "scripts/run_experiment.py",
        "--config",
        str(temp_config_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()

    state = {
        "pid": process.pid,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "temp_config_path": str(temp_config_path),
        "console_log_path": str(log_path),
        "previous_latest_run_id": _read_previous_latest_run_id(),
    }
    save_active_run_state(state)
    return load_active_run_state() or state


def read_log_tail(path: str | Path | None, max_lines: int = 200) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.exists():
        return ""
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def read_log_head(path: str | Path | None, max_lines: int = 40) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.exists():
        return ""
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[:max_lines])


def parse_progress(log_text: str) -> dict[str, Any]:
    progress = {
        "phase": None,
        "cycle_current": None,
        "cycle_total": None,
        "generation_current": None,
        "generation_total": None,
        "overall_ratio": 0.0,
        "last_message": None,
    }

    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    if not lines:
        return progress
    progress["last_message"] = lines[-1]

    for line in lines:
        cycle_match = re.search(r"cykl (\d+)/(\d+)", line, flags=re.IGNORECASE)
        if cycle_match:
            progress["cycle_current"] = int(cycle_match.group(1))
            progress["cycle_total"] = int(cycle_match.group(2))

        if "FAZA TRAIN-GA" in line:
            progress["phase"] = "TRAIN-GA"
        elif "FAZA VAL-GA" in line:
            progress["phase"] = "VAL-GA"
        elif "Baseline SVM" in line:
            progress["phase"] = "Baseline"

        val_gen_match = re.search(r"\[VAL-GA GEN\s+(\d+)/(\d+)\]", line)
        if val_gen_match:
            progress["phase"] = "VAL-GA"
            progress["generation_current"] = int(val_gen_match.group(1))
            progress["generation_total"] = int(val_gen_match.group(2))
            continue

        gen_match = re.search(r"\[GEN\s+(\d+)/(\d+)\]", line)
        if gen_match:
            progress["generation_current"] = int(gen_match.group(1))
            progress["generation_total"] = int(gen_match.group(2))

    if progress["cycle_current"] and progress["cycle_total"]:
        within_cycle = 0.0
        if progress["generation_current"] and progress["generation_total"]:
            within_cycle = progress["generation_current"] / max(progress["generation_total"], 1)
        progress["overall_ratio"] = min(
            0.99,
            ((progress["cycle_current"] - 1) + within_cycle) / max(progress["cycle_total"], 1),
        )
    elif progress["phase"] == "Baseline":
        progress["overall_ratio"] = 0.05

    return progress


def extract_run_context(log_text: str) -> dict[str, Any]:
    context = {
        "run_id": None,
        "artifacts_path": None,
    }
    run_match = re.search(r"run_id=([A-Za-z0-9._-]+)", log_text)
    artifacts_match = re.search(r"artifacts=([^\s]+)", log_text)
    if run_match:
        context["run_id"] = run_match.group(1)
    if artifacts_match:
        context["artifacts_path"] = artifacts_match.group(1)
    return context


def completed_run_from_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not state:
        return None
    if state.get("running"):
        return None
    latest = load_latest_run()
    if latest is None:
        return None
    if latest.get("run_id") == state.get("previous_latest_run_id"):
        return None
    return latest
