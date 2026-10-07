import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.utils.artifacts import (
    append_csv_row,
    class_distribution,
    ensure_artifact_layout,
    save_mask,
    summarize_run_manifest_entry,
    write_csv_rows,
    write_json,
    write_latest_run_pointer,
)
from ga.utils.logger import get_logger


def test_artifact_helpers_persist_files(tmp_path):
    layout = ensure_artifact_layout(tmp_path / "artifacts", experiment_name="baseline")

    assert layout["run_root"].parent == layout["runs_root"]
    assert layout["run_root"].name.startswith("20")
    assert layout["metrics"].is_dir()
    assert layout["checkpoints"].is_dir()
    assert layout["logs"].is_dir()

    payload = {"metric": 0.75, "nested": {"values": np.array([1, 2, 3])}}
    write_json(layout["metrics"] / "summary.json", payload)
    loaded_payload = json.loads((layout["metrics"] / "summary.json").read_text(encoding="utf-8"))
    assert loaded_payload["metric"] == 0.75
    assert loaded_payload["nested"]["values"] == [1, 2, 3]

    write_csv_rows(
        layout["metrics"] / "cycles.csv",
        [{"cycle": 1, "score": 0.5}, {"cycle": 2, "score": 0.7}],
    )
    csv_text = (layout["metrics"] / "cycles.csv").read_text(encoding="utf-8")
    assert "cycle,score" in csv_text
    assert "2,0.7" in csv_text

    mask = np.array([True, False, True], dtype=bool)
    save_mask(layout["checkpoints"] / "best_mask.npy", mask)
    loaded_mask = np.load(layout["checkpoints"] / "best_mask.npy")
    assert loaded_mask.tolist() == [True, False, True]

    assert class_distribution(np.array([1, 1, 2, 3, 3])) == {"1": 2, "2": 1, "3": 2}


def test_run_history_manifest_and_latest_pointer(tmp_path):
    layout = ensure_artifact_layout(tmp_path / "artifacts", experiment_name="exp_name", run_id="run_001")
    summary_path = layout["metrics"] / "summary.json"
    log_path = layout["logs"] / "run_001.log"

    manifest_entry = summarize_run_manifest_entry(
        run_id=layout["run_id"],
        experiment_name="exp_name",
        run_root=layout["run_root"],
        log_path=log_path,
        summary_path=summary_path,
        seed=42,
        best_val_metric=0.91,
        best_cycle=2,
        best_k_per_class=30,
        duration_sec=12.5,
    )
    append_csv_row(layout["index_csv"], manifest_entry)
    write_latest_run_pointer(layout["latest_json"], manifest_entry)

    history_text = layout["index_csv"].read_text(encoding="utf-8")
    latest_text = layout["latest_json"].read_text(encoding="utf-8")

    assert "run_001" in history_text
    assert "exp_name" in history_text
    assert '"run_id": "run_001"' in latest_text


def test_logger_can_write_to_file(tmp_path):
    log_path = tmp_path / "artifacts" / "logs" / "experiment.log"
    logger = get_logger(log_path=log_path)
    logger.info("artifact logger smoke test")

    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    log_text = log_path.read_text(encoding="utf-8")
    assert "artifact logger smoke test" in log_text
