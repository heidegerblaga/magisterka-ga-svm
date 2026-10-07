from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys
import time

import click
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from ga.sweep import (
    build_sweep_layout,
    build_trial_result,
    default_objective_mode,
    discover_trial_summary,
    estimate_search_space_size,
    generate_trial_plan,
    rank_trial_results,
    write_sweep_manifest,
    write_sweep_ranking,
    write_sweep_state,
)


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("ga.sweep")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _wait_for_summary(artifacts_root: Path, sweep_id: str, trial_id: str, timeout_sec: float = 15.0) -> Path | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        summary_path = discover_trial_summary(artifacts_root, sweep_id, trial_id)
        if summary_path is not None:
            return summary_path
        time.sleep(0.25)
    return None


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--sweep-config", "-s", type=click.Path(exists=True), required=True)
def main(config, sweep_config):
    base_config_path = Path(config)
    sweep_config_path = Path(sweep_config)
    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    sweep_cfg = yaml.safe_load(sweep_config_path.read_text(encoding="utf-8")) or {}

    sweep_name = sweep_cfg.get("sweep_name") or base_cfg.get("experiment_name", "parallel_sweep")
    layout = build_sweep_layout(
        base_cfg.get("artifacts_root", "artifacts"),
        sweep_name=sweep_name,
        sweep_id=sweep_cfg.get("sweep_id"),
    )
    logger = build_logger(Path(layout["orchestrator_log"]))

    objective_name = str((sweep_cfg.get("objective") or {}).get("name", "full_validation_f1"))
    objective_mode = str((sweep_cfg.get("objective") or {}).get("mode", default_objective_mode(objective_name)))
    selection_seed = int(sweep_cfg.get("selection_seed", base_cfg.get("seed", 42)))
    requested_trials = int(sweep_cfg.get("num_trials", 20))
    max_parallel = max(1, int(sweep_cfg.get("max_parallel", 4)))
    search_space = sweep_cfg.get("search_space") or {}

    trials = generate_trial_plan(
        base_cfg=base_cfg,
        search_space=search_space,
        num_trials=requested_trials,
        selection_seed=selection_seed,
        sweep_id=str(layout["sweep_id"]),
    )

    for trial in trials:
        config_path = (Path(layout["configs"]) / f"{trial['trial_id']}.yaml").resolve()
        config_path.write_text(yaml.safe_dump(trial["config"], sort_keys=False), encoding="utf-8")
        trial["config_path"] = str(config_path)
        trial["log_path"] = str((Path(layout["logs"]) / f"{trial['trial_id']}.log").resolve())

    manifest = {
        "sweep_id": layout["sweep_id"],
        "sweep_name": sweep_name,
        "base_config_path": str(base_config_path.resolve()),
        "sweep_config_path": str(sweep_config_path.resolve()),
        "created_at": _now_iso(),
        "objective": {"name": objective_name, "mode": objective_mode},
        "selection_seed": selection_seed,
        "requested_trials": requested_trials,
        "effective_trials": len(trials),
        "max_parallel": max_parallel,
        "search_space": search_space,
        "search_space_size_estimate": estimate_search_space_size(search_space),
        "trials": [
            {
                "trial_id": trial["trial_id"],
                "config_path": trial["config_path"],
                "log_path": trial["log_path"],
                "overrides": trial["overrides"],
            }
            for trial in trials
        ],
    }
    write_sweep_manifest(layout["manifest_json"], manifest)

    logger.info(
        "Starting sweep: sweep_id=%s | requested=%s | effective=%s | parallel=%s | objective=%s (%s)",
        layout["sweep_id"],
        requested_trials,
        len(trials),
        max_parallel,
        objective_name,
        objective_mode,
    )

    trials_state = {
        trial["trial_id"]: {
            "trial_id": trial["trial_id"],
            "status": "queued",
            "overrides": trial["overrides"],
            "config_path": trial["config_path"],
            "log_path": trial["log_path"],
            "pid": None,
            "run_id": None,
            "summary_path": None,
            "started_at": None,
            "finished_at": None,
            "duration_sec": None,
            "returncode": None,
            "objective_value": None,
            "last_error": None,
        }
        for trial in trials
    }
    active_processes: dict[str, subprocess.Popen] = {}
    result_rows: dict[str, dict] = {}
    ranking_rows: list[dict] = []
    started_at = _now_iso()
    artifacts_root = Path(layout["root"])
    execution_cwd = Path.cwd()
    run_experiment_path = REPO_ROOT / "scripts" / "run_experiment.py"
    exit_code = 0
    sweep_status = "running"

    def persist_state(status: str) -> None:
        queued = [row for row in trials_state.values() if row["status"] == "queued"]
        running = [row for row in trials_state.values() if row["status"] == "running"]
        completed = [row for row in trials_state.values() if row["status"] == "completed"]
        failed = [row for row in trials_state.values() if row["status"] == "failed"]
        best_row = ranking_rows[0] if ranking_rows else None
        payload = {
            "sweep_id": layout["sweep_id"],
            "status": status,
            "started_at": started_at,
            "updated_at": _now_iso(),
            "finished_at": None if status == "running" else _now_iso(),
            "objective_name": objective_name,
            "objective_mode": objective_mode,
            "num_trials": len(trials_state),
            "max_parallel": max_parallel,
            "completed_trials": len(completed),
            "running_trials": len(running),
            "queued_trials": len(queued),
            "failed_trials": len(failed),
            "best_trial_id": None if best_row is None else best_row.get("trial_id"),
            "best_run_id": None if best_row is None else best_row.get("run_id"),
            "best_objective_value": None if best_row is None else best_row.get("objective_value"),
            "orchestrator_log": str(Path(layout["orchestrator_log"]).resolve()),
            "ranking_csv": str(Path(layout["ranking_csv"]).resolve()),
            "trials": sorted(trials_state.values(), key=lambda row: row["trial_id"]),
        }
        write_sweep_state(layout["state_json"], payload)
        write_sweep_ranking(layout["ranking_csv"], ranking_rows)

    persist_state("running")

    try:
        queue = [trial["trial_id"] for trial in trials]
        trial_map = {trial["trial_id"]: trial for trial in trials}

        while queue or active_processes:
            while queue and len(active_processes) < max_parallel:
                trial_id = queue.pop(0)
                trial = trial_map[trial_id]
                log_path = Path(trial["log_path"])
                log_handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    "-u",
                    str(run_experiment_path),
                    "--config",
                    trial["config_path"],
                ]
                process = subprocess.Popen(
                    command,
                    cwd=str(execution_cwd),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log_handle.close()
                active_processes[trial_id] = process
                trials_state[trial_id]["status"] = "running"
                trials_state[trial_id]["started_at"] = _now_iso()
                trials_state[trial_id]["pid"] = process.pid
                logger.info("Started %s | pid=%s | overrides=%s", trial_id, process.pid, trial["overrides"])

            time.sleep(1.0)

            for trial_id, process in list(active_processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue

                active_processes.pop(trial_id, None)
                trials_state[trial_id]["returncode"] = returncode
                trials_state[trial_id]["finished_at"] = _now_iso()
                if trials_state[trial_id]["started_at"]:
                    started = datetime.fromisoformat(trials_state[trial_id]["started_at"])
                    trials_state[trial_id]["duration_sec"] = round(
                        max(0.0, (datetime.now() - started).total_seconds()),
                        3,
                    )

                summary_path = _wait_for_summary(artifacts_root, str(layout["sweep_id"]), trial_id)
                if returncode == 0 and summary_path is not None:
                    result = build_trial_result(summary_path, objective_name, objective_mode)
                    result["config_path"] = trials_state[trial_id]["config_path"]
                    result["log_path"] = trials_state[trial_id]["log_path"]
                    result_rows[trial_id] = result
                    ranking_rows = rank_trial_results(list(result_rows.values()), objective_name, objective_mode)
                    trials_state[trial_id]["status"] = "completed"
                    trials_state[trial_id]["summary_path"] = str(summary_path.resolve())
                    trials_state[trial_id]["run_id"] = result.get("run_id")
                    trials_state[trial_id]["objective_value"] = result.get("objective_value")
                    logger.info(
                        "Completed %s | run_id=%s | objective=%s",
                        trial_id,
                        result.get("run_id"),
                        result.get("objective_value"),
                    )
                else:
                    trials_state[trial_id]["status"] = "failed"
                    trials_state[trial_id]["last_error"] = f"returncode={returncode}, summary_path={summary_path}"
                    logger.error(
                        "Failed %s | returncode=%s | summary_path=%s",
                        trial_id,
                        returncode,
                        summary_path,
                    )

            persist_state("running")

        sweep_status = "completed"
    except KeyboardInterrupt:
        sweep_status = "interrupted"
        exit_code = 1
        logger.warning("Sweep interrupted. Terminating active trial processes.")
        for process in active_processes.values():
            try:
                process.terminate()
            except Exception:
                continue
    except Exception:
        sweep_status = "failed"
        exit_code = 1
        logger.exception("Parallel sweep failed with an unhandled error.")
    finally:
        for trial_id, process in list(active_processes.items()):
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    continue
            active_processes.pop(trial_id, None)
            if trials_state[trial_id]["status"] == "running":
                trials_state[trial_id]["status"] = "failed"
                trials_state[trial_id]["finished_at"] = _now_iso()
                trials_state[trial_id]["last_error"] = "terminated by sweep orchestrator"

        persist_state(sweep_status)
        logger.info("Sweep finished: sweep_id=%s | status=%s", layout["sweep_id"], sweep_status)
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
