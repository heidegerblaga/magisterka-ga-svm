from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import time
import warnings

import click
import numpy as np
import yaml

# Val-GA ocenia podzbiory, które bywają jednoklasowe — ostrzeżenia sklearn tylko zaśmiecają logi
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

# ensure src on path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.encoding import KPerClassConfig
from ga.fitness import FitnessConfig
from ga.loop.engine import (
    GAConfig,
    DagaKAdaptationConfig,
    run_ga_k_per_class,
    run_ga_k_per_class_with_trace,
    run_daga_k_per_class,
    build_ga_config,
)
from ga.models.svm import SVMConfig, evaluate_fitted_svm, train_eval_svm
from ga.phase_engine import run_phase_experiment
from ga.phase_switch import PHASE_SWITCH_COLUMNS, run_phase_switch_experiment
from ga.reporting import HISTORY_TABLE_COLUMNS, build_report_row, load_datasets_from_config
from ga.utils.artifacts import (
    append_csv_row,
    class_distribution,
    ensure_artifact_layout,
    save_mask,
    summarize_history,
    summarize_run_manifest_entry,
    summarize_split,
    write_csv_rows,
    write_json,
    write_latest_run_pointer,
)
from ga.utils.logger import get_logger
from ga.utils.random import set_global_seed


CYCLE_TABLE_COLUMNS = [
    "cycle",
    "k_per_class",
    "cycle_duration_sec",
    "train_ga_best_fitness",
    "train_ga_generations_executed",
    "train_ga_svm_evaluations",
    "train_ga_best_val_metric",
    "train_ga_test_metric",
    "train_ga_best_full_validation_metric",
    "val_ga_best_fitness",
    "val_ga_generations_executed",
    "val_ga_population_evaluations",
    "val_ga_adv_increment_metric",
    "decision",
    "perfect_on_current_val",
    "k_increased_this_cycle",
    "best_sv_ratio",
    "sv_ratio_at_hardening",
    "sv_ratio_delta_since_hardening",
    "sv_ratio_near_threshold",
    "sv_ratio_stable_since_hardening",
    "stable_cycles_at_current_k",
    "forced_growth_after_patience",
    "val_rows_before",
    "val_rows_after",
    "val_rows_added",
    # --- rozszerzenia badania V' (puste dla trybów innych niż dagasvm_phase) ---
    "k_after",
    "decision_reason",
    "k_increases_this_cycle",
    "vprime_new_elements",
    "vprime_dropped_elements",
    "increment_errors",
    "increment_error_rate",
    "candidate_pool_error_rate",
    "increment_mean_abs_margin",
    "num_support_vectors",
    "train_rows",
    "train_ga_test_accuracy",
    "train_ga_full_validation_accuracy",
    "control_model_predictions_cum",
    # --- reguła stopu BA(V')=1,0 (puste/domyślne dla przebiegów bez rozszerzonej reguły) ---
    "min_vprime_per_class_required",
    "min_vprime_size_required",
    "vprime_min_class_count",
    "min_vprime_size_reached",
    "perfect_rechecks_required",
    "perfect_rechecks_done",
    "perfect_gate_pending",
    "perfect_stop_blocked",
    "perfect_stop_blocked_reason",
    "stop_decision",
    "stop_category",
    # --- koszt doboru V' (faktyczne wiersze przekazane modelowi; Val-GA nominalnie osobno) ---
    "vprime_candidates",
    "vprime_scored_pool",
    "vprime_increment_rows",
    "selection_predict_rows",
    "selection_decision_rows",
    "val_ga_nominal_subset_rows_scored",
    "diagnostic_scored_rows",
]


def build_svm_config(cfg) -> SVMConfig:
    return SVMConfig(
        kernel=cfg["svm"]["kernel"],
        C=cfg["svm"]["C"],
        gamma=cfg["svm"]["gamma"],
        class_weight=cfg["svm"].get("class_weight", "balanced"),
        probability=cfg["svm"].get("probability", True),
        standardize=cfg["svm"].get("standardize", True),
    )


def build_ga_config_from_cfg(base_key: str, cfg) -> GAConfig:
    return build_ga_config(cfg.get(base_key, {}))


def load_datasets(cfg, log, apply_limit: bool = True):
    datasets = load_datasets_from_config(cfg, apply_limit=apply_limit)
    scope = "run" if apply_limit else "full-evaluation"
    log.info(
        "Loaded %s datasets: train=%s, validation=%s, test=%s",
        scope,
        len(datasets["train"]["X"]),
        len(datasets["validation"]["X"]),
        0 if datasets["test"]["X"] is None else len(datasets["test"]["X"]),
    )
    if datasets["test"]["X"] is None:
        log.warning("No external test_csv provided in config; external test will be skipped.")
    return datasets


def run_baseline_phase(X_train, y_train, X_val, y_val, X_test, y_test, svm_cfg, cfg, log):
    metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")

    res_val = train_eval_svm(
        X_train,
        y_train,
        X_val,
        y_val,
        svm_cfg,
        metric=metric,
        return_dict=True,
        return_model=True,
    )
    baseline_model = res_val.pop("model")
    log.info(
        "Baseline SVM (%s) VAL %s: %.4f | all=%s | fit=%.3fs, score=%.3fs",
        svm_cfg.kernel,
        metric,
        res_val["metric"],
        res_val["all"],
        res_val["fit_time"],
        res_val["score_time"],
    )

    res_test = None
    if X_test is not None:
        res_test = evaluate_fitted_svm(baseline_model, X_test, y_test, metric=metric)
        log.info(
            "Baseline SVM (%s) TEST %s: %.4f | all=%s | score=%.3fs",
            svm_cfg.kernel,
            metric,
            res_test["metric"],
            res_test["all"],
            res_test["score_time"],
        )
    return res_val, res_test


def run_full_svm_mode(X_train, y_train, X_val, y_val, cfg, log):
    return {
        "cycles": [],
        "cycle_table_rows": [],
        "global_best": None,
        "final_global_val": summarize_split(X_val, y_val),
        "representative_model": {
            "algorithm_mode": "full_svm",
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "train_mask_path": None,
            "cycle": None,
        },
    }


def run_gasvm_mode(
    X_train,
    y_train,
    X_val_full,
    y_val_full,
    X_test,
    y_test,
    svm_cfg,
    cfg,
    log,
    artifact_layout,
):
    baseline_metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")
    fcfg = FitnessConfig(
        metric=baseline_metric,
        class_imbalance_penalty=cfg.get("fitness", {}).get("class_imbalance_penalty", 0.0),
    )
    k_per_class = int(cfg.get("encoding", {}).get("k_per_class", 20))
    kcfg = KPerClassConfig(k_per_class=k_per_class, random_state=cfg.get("seed", 42))
    gcfg_train = build_ga_config_from_cfg("ga", cfg)

    metrics_dir = Path(artifact_layout["metrics"])
    checkpoints_dir = Path(artifact_layout["checkpoints"])
    cycle_start = time.perf_counter()
    log.info("=" * 60)
    log.info("FAZA TRAIN-GA (GASVM), k_per_class=%s", k_per_class)

    history_train, best_train_mask, trace, best_masks_per_gen = run_ga_k_per_class_with_trace(
        X_pool=X_train,
        y_pool=y_train,
        X_val=X_val_full,
        y_val=y_val_full,
        kcfg=kcfg,
        svm_cfg=svm_cfg,
        fcfg=fcfg,
        gcfg=gcfg_train,
        seed=cfg.get("seed", 42),
    )

    train_history_summary = summarize_history(
        history_train,
        gcfg_train.population_size,
        evaluation_label="svm_evaluations",
    )
    train_history_summary["trace"] = trace
    train_history_path = metrics_dir / "cycle_01_train_ga_history.json"
    write_json(train_history_path, train_history_summary)

    best_masks_path = checkpoints_dir / "cycle_01_best_masks_per_generation.npy"
    np.save(best_masks_path, best_masks_per_gen.astype(bool, copy=False))

    best_train_mask_path = checkpoints_dir / "cycle_01_best_train_mask.npy"
    global_best_mask_path = checkpoints_dir / "global_best_train_mask.npy"
    save_mask(best_train_mask_path, best_train_mask)
    save_mask(global_best_mask_path, best_train_mask)

    X_tr_best = X_train[best_train_mask]
    y_tr_best = y_train[best_train_mask]

    best_res_val = train_eval_svm(
        X_tr_best,
        y_tr_best,
        X_val_full,
        y_val_full,
        svm_cfg,
        metric=baseline_metric,
        return_dict=True,
    )
    log.info(
        "[TRAIN-GA GASVM] best VAL %s: %.4f | all=%s | fit=%.3fs, score=%.3fs",
        baseline_metric,
        best_res_val["metric"],
        best_res_val["all"],
        best_res_val["fit_time"],
        best_res_val["score_time"],
    )

    best_res_test = None
    if X_test is not None:
        best_res_test = train_eval_svm(
            X_tr_best,
            y_tr_best,
            X_test,
            y_test,
            svm_cfg,
            metric=baseline_metric,
            return_dict=True,
        )
        log.info(
            "[TRAIN-GA GASVM] TEST %s: %.4f | all=%s | fit=%.3fs, score=%.3fs",
            baseline_metric,
            best_res_test["metric"],
            best_res_test["all"],
            best_res_test["fit_time"],
            best_res_test["score_time"],
        )

    cycle_duration_sec = time.perf_counter() - cycle_start
    cycle_summary = {
        "cycle": 1,
        "k_per_class": k_per_class,
        "cycle_duration_sec": cycle_duration_sec,
        "train_pool": summarize_split(X_train, y_train),
        "current_val": summarize_split(X_val_full, y_val_full),
        "best_train": summarize_split(X_tr_best, y_tr_best),
        "train_ga": {
            "history_path": str(train_history_path),
            "best_train_mask_path": str(best_train_mask_path),
            "best_masks_per_generation_path": str(best_masks_path),
            "history_summary": train_history_summary,
            "best_val": best_res_val,
            "best_test": best_res_test,
        },
    }
    cycle_row = {
        "cycle": 1,
        "k_per_class": k_per_class,
        "cycle_duration_sec": cycle_duration_sec,
        "train_ga_best_fitness": train_history_summary["best_final_fitness"],
        "train_ga_generations_executed": train_history_summary["generations_executed"],
        "train_ga_svm_evaluations": train_history_summary["svm_evaluations"],
        "train_ga_best_val_metric": float(best_res_val["metric"]),
        "train_ga_test_metric": None if best_res_test is None else float(best_res_test["metric"]),
        "val_ga_best_fitness": None,
        "val_ga_generations_executed": None,
        "val_ga_population_evaluations": None,
        "val_ga_adv_increment_metric": None,
        "val_rows_before": int(len(X_val_full)),
        "val_rows_after": int(len(X_val_full)),
        "val_rows_added": 0,
    }

    return {
        "cycles": [cycle_summary],
        "cycle_table_rows": [cycle_row],
        "global_best": {
            "cycle": 1,
            "k_per_class": k_per_class,
            "metrics_val": best_res_val,
            "metrics_test": best_res_test,
            "train_summary": summarize_split(X_tr_best, y_tr_best),
            "train_mask_path": str(global_best_mask_path),
            "validation_rows": int(len(X_val_full)),
        },
        "final_global_val": summarize_split(X_val_full, y_val_full),
        "representative_model": {
            "algorithm_mode": "gasvm",
            "training_rows": int(len(X_tr_best)),
            "validation_rows": int(len(X_val_full)),
            "train_mask_path": str(global_best_mask_path),
            "cycle": 1,
        },
    }


def run_dagasvm_mode(
    X_train,
    y_train,
    X_val_full,
    y_val_full,
    X_test,
    y_test,
    svm_cfg,
    cfg,
    log,
    artifact_layout,
):
    """DAGASVM (2014): adaptacyjne zwiększanie K w trakcie pojedynczego runu.

    Implementacja opiera się na schemacie:
      - stable chromosome size counter: `cs` oraz próg `Ts`
      - SV ratio: ρ_SV = s / t'
      - próg TSV oraz mnożnik δK do zwiększania K
      - LGA mode (local/global) w selekcji rodziców
    """
    baseline_metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")
    fcfg = FitnessConfig(
        metric=baseline_metric,
        class_imbalance_penalty=cfg.get("fitness", {}).get("class_imbalance_penalty", 0.0),
    )

    daga_cfg = cfg.get("daga", {})
    ks_cfg = cfg.get("k_schedule", {})
    k_start = int(
        daga_cfg.get(
            "k_start",
            ks_cfg.get("k_start", cfg.get("encoding", {}).get("k_per_class", 20)),
        )
    )
    daca = DagaKAdaptationConfig(
        k_start=k_start,
        TSV=float(daga_cfg.get("TSV", daga_cfg.get("tsv", 0.25))),
        Ts=int(daga_cfg.get("Ts", daga_cfg.get("stable_chromosome_size_counter", 3))),
        epsilon=float(daga_cfg.get("epsilon", daga_cfg.get("lga_epsilon", 0.5))),
        k_max=None if daga_cfg.get("k_max") is None else int(daga_cfg.get("k_max")),
    )

    metrics_dir = Path(artifact_layout["metrics"])
    checkpoints_dir = Path(artifact_layout["checkpoints"])

    cycle_start = time.perf_counter()
    log.info("=" * 60)
    log.info(
        "FAZA TRAIN-GA (DAGASVM, adaptive K) | k_start=%s TSV=%s Ts=%s epsilon=%s k_max=%s",
        daca.k_start,
        daca.TSV,
        daca.Ts,
        daca.epsilon,
        daca.k_max,
    )

    gcfg_train = build_ga_config_from_cfg("ga", cfg)
    history_train, best_train_mask, trace, best_masks_per_gen = run_daga_k_per_class(
        X_pool=X_train,
        y_pool=y_train,
        X_val=X_val_full,
        y_val=y_val_full,
        kcfg=KPerClassConfig(k_per_class=k_start, random_state=cfg.get("seed", 42)),
        daca=daca,
        svm_cfg=svm_cfg,
        fcfg=fcfg,
        gcfg=gcfg_train,
        seed=cfg.get("seed", 42),
    )

    train_history_summary = summarize_history(
        history_train,
        gcfg_train.population_size,
        evaluation_label="svm_evaluations",
    )
    train_history_summary["trace"] = trace
    train_history_path = metrics_dir / "cycle_01_daga_train_history.json"
    write_json(train_history_path, train_history_summary)

    best_masks_path = checkpoints_dir / "cycle_01_best_masks_per_generation.npy"
    np.save(best_masks_path, best_masks_per_gen.astype(bool, copy=False))

    best_train_mask_path = checkpoints_dir / "cycle_01_best_train_mask.npy"
    global_best_mask_path = checkpoints_dir / "global_best_train_mask.npy"
    save_mask(best_train_mask_path, best_train_mask)
    save_mask(global_best_mask_path, best_train_mask)

    X_tr_best = X_train[best_train_mask]
    y_tr_best = y_train[best_train_mask]

    best_res_val = train_eval_svm(
        X_tr_best,
        y_tr_best,
        X_val_full,
        y_val_full,
        svm_cfg,
        metric=baseline_metric,
        return_dict=True,
    )
    log.info(
        "[DAGASVM] best VAL %s: %.4f | all=%s | fit=%.3fs, score=%.3fs | SV=%s",
        baseline_metric,
        best_res_val["metric"],
        best_res_val["all"],
        best_res_val["fit_time"],
        best_res_val["score_time"],
        best_res_val.get("num_support_vectors"),
    )

    best_res_test = None
    if X_test is not None:
        best_res_test = train_eval_svm(
            X_tr_best,
            y_tr_best,
            X_test,
            y_test,
            svm_cfg,
            metric=baseline_metric,
            return_dict=True,
        )
        log.info(
            "[DAGASVM] TEST %s: %.4f | all=%s | fit=%.3fs, score=%.3fs",
            baseline_metric,
            best_res_test["metric"],
            best_res_test["all"],
            best_res_test["fit_time"],
            best_res_test["score_time"],
        )

    cycle_duration_sec = time.perf_counter() - cycle_start
    cycle_summary = {
        "cycle": 1,
        "k_start": int(daca.k_start),
        "cycle_duration_sec": float(cycle_duration_sec),
        "train_pool": summarize_split(X_train, y_train),
        "validation_full": summarize_split(X_val_full, y_val_full),
        "best_train": summarize_split(X_tr_best, y_tr_best),
        "train_ga": {
            "history_path": str(train_history_path),
            "best_train_mask_path": str(best_train_mask_path),
            "best_masks_per_generation_path": str(best_masks_path),
            "history_summary": train_history_summary,
            "best_val": best_res_val,
            "best_test": best_res_test,
        },
        "daga": {
            "TSV": float(daca.TSV),
            "Ts": int(daca.Ts),
            "epsilon": float(daca.epsilon),
            "k_max": None if daca.k_max is None else int(daca.k_max),
        },
    }

    return {
        "cycles": [cycle_summary],
        "cycle_table_rows": [
            {
                "cycle": 1,
                "k_per_class": int(daca.k_start),
                "cycle_duration_sec": float(cycle_duration_sec),
                "train_ga_best_fitness": train_history_summary["best_final_fitness"],
                "train_ga_generations_executed": train_history_summary["generations_executed"],
                "train_ga_svm_evaluations": train_history_summary["svm_evaluations"],
                "train_ga_best_val_metric": float(best_res_val["metric"]),
                "train_ga_test_metric": None
                if best_res_test is None
                else float(best_res_test["metric"]),
                "val_ga_best_fitness": None,
                "val_ga_generations_executed": None,
                "val_ga_population_evaluations": None,
                "val_ga_adv_increment_metric": None,
                "val_rows_before": int(len(X_val_full)),
                "val_rows_after": int(len(X_val_full)),
                "val_rows_added": 0,
            }
        ],
        "global_best": {
            "cycle": 1,
            "k_per_class": int(daca.k_start),
            "metrics_val": best_res_val,
            "metrics_test": best_res_test,
            "train_summary": summarize_split(X_tr_best, y_tr_best),
            "train_mask_path": str(global_best_mask_path),
            "validation_rows": int(len(X_val_full)),
        },
        "final_global_val": summarize_split(X_val_full, y_val_full),
        "representative_model": {
            "algorithm_mode": "dagasvm",
            "training_rows": int(len(X_tr_best)),
            "validation_rows": int(len(X_val_full)),
            "train_mask_path": str(global_best_mask_path),
            "cycle": 1,
        },
    }


def run_dagasvm_phase_mode(
    X_train,
    y_train,
    X_val_full,
    y_val_full,
    X_test,
    y_test,
    svm_cfg,
    cfg,
    log,
    artifact_layout,
):
    """Fazowy DAGASVM (T'/V'); logika w ``ga.phase_engine`` (patrz opis rozdziału ról V i V')."""
    return run_phase_experiment(
        X_train,
        y_train,
        X_val_full,
        y_val_full,
        X_test,
        y_test,
        svm_cfg,
        cfg,
        log,
        artifact_layout,
    )


run_dagasvm_valga_mode = run_dagasvm_phase_mode


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option(
    "--artifacts-root",
    default=None,
    help="Katalog wyjściowy (domyślnie: cfg['artifacts_root'] albo 'artifacts').",
)
def main(config, artifacts_root):
    experiment_start = time.perf_counter()
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    cfg.setdefault("dataset_name", cfg.get("experiment_name", "dataset"))
    cfg.setdefault("algorithm_mode", "dagasvm")

    created_at = datetime.now().isoformat(timespec="seconds")
    set_global_seed(cfg.get("seed", 42))
    artifact_layout = ensure_artifact_layout(
        outdir=artifacts_root or cfg.get("artifacts_root", "artifacts"),
        experiment_name=cfg.get("experiment_name", "exp"),
    )
    run_root = Path(artifact_layout["run_root"])
    log_path = Path(artifact_layout["logs"]) / f"{artifact_layout['run_id']}.log"
    log = get_logger(log_path=log_path)
    log.info(
        "Loaded config: %s | dataset=%s | mode=%s | run_id=%s | artifacts=%s",
        cfg.get("experiment_name", "exp"),
        cfg.get("dataset_name", "dataset"),
        cfg.get("algorithm_mode", "dagasvm"),
        artifact_layout["run_id"],
        run_root,
    )
    write_json(Path(artifact_layout["metrics"]) / "config_snapshot.json", cfg)

    datasets = load_datasets(cfg, log, apply_limit=True)
    X_train = datasets["train"]["X"]
    y_train = datasets["train"]["y"]
    X_val_full = datasets["validation"]["X"]
    y_val_full = datasets["validation"]["y"]
    X_test = datasets["test"]["X"]
    y_test = datasets["test"]["y"]

    max_rows = cfg.get("data", {}).get("max_rows", 1000)
    if max_rows is None or max_rows <= 0:
        full_eval_datasets = datasets
    else:
        full_eval_datasets = load_datasets(cfg, log, apply_limit=False)

    svm_cfg = build_svm_config(cfg)
    baseline_val, baseline_test = run_baseline_phase(
        X_train,
        y_train,
        X_val_full,
        y_val_full,
        X_test,
        y_test,
        svm_cfg,
        cfg,
        log,
    )
    write_json(
        Path(artifact_layout["metrics"]) / "baseline.json",
        {"val": baseline_val, "test": baseline_test},
    )

    mode = cfg.get("algorithm_mode", "dagasvm")
    if mode == "full_svm":
        mode_outputs = run_full_svm_mode(X_train, y_train, X_val_full, y_val_full, cfg, log)
    elif mode == "gasvm":
        mode_outputs = run_gasvm_mode(
            X_train,
            y_train,
            X_val_full,
            y_val_full,
            X_test,
            y_test,
            svm_cfg,
            cfg,
            log,
            artifact_layout,
        )
    elif mode == "dagasvm_phase_switch":
        # automat T' <-> V' (followup_phase_switch); logika w ``ga.phase_switch``
        mode_outputs = run_phase_switch_experiment(
            X_train,
            y_train,
            X_val_full,
            y_val_full,
            X_test,
            y_test,
            svm_cfg,
            cfg,
            log,
            artifact_layout,
        )
    elif mode in {"dagasvm_valga", "dagasvm_phase"}:
        mode_outputs = run_dagasvm_phase_mode(
            X_train,
            y_train,
            X_val_full,
            y_val_full,
            X_test,
            y_test,
            svm_cfg,
            cfg,
            log,
            artifact_layout,
        )
    else:
        mode_outputs = run_dagasvm_mode(
            X_train,
            y_train,
            X_val_full,
            y_val_full,
            X_test,
            y_test,
            svm_cfg,
            cfg,
            log,
            artifact_layout,
        )

    experiment_summary = {
        "created_at": created_at,
        "run_id": artifact_layout["run_id"],
        "experiment_name": cfg.get("experiment_name", "exp"),
        "dataset_name": cfg.get("dataset_name", cfg.get("experiment_name", "dataset")),
        "algorithm_mode": mode,
        "sweep_id": cfg.get("sweep_id"),
        "trial_id": cfg.get("trial_id"),
        "trial_overrides": cfg.get("trial_overrides"),
        # w trybie dagasvm_phase_switch optymalizowana jest fitness.tprime_metric (domyślnie = fitness.metric)
        "objective_metric_name": (
            (cfg.get("fitness", {}).get("tprime_metric") if mode == "dagasvm_phase_switch" else None)
            or cfg.get("fitness", {}).get("metric", "balanced_accuracy")
        ),
        "run_root": str(run_root),
        "seed": cfg.get("seed", 42),
        "log_path": str(log_path),
        "dataset": {
            "train": summarize_split(X_train, y_train),
            "validation_full": summarize_split(X_val_full, y_val_full),
            "test": summarize_split(X_test, y_test),
        },
        "baseline": {
            "val": baseline_val,
            "test": baseline_test,
        },
        "phase_study": mode_outputs.get("phase_study"),
        "cycles": mode_outputs["cycles"],
        "global_best": mode_outputs["global_best"],
        "final_global_val": mode_outputs["final_global_val"],
        "representative_model": mode_outputs["representative_model"],
        "experiment_duration_sec": time.perf_counter() - experiment_start,
    }

    summary_path = Path(artifact_layout["metrics"]) / "summary.json"
    cycles_path = Path(artifact_layout["metrics"]) / "cycles.csv"
    experiment_summary["summary_path"] = str(summary_path)
    experiment_summary["report_row"] = build_report_row(
        experiment_summary,
        cfg=cfg,
        actual_datasets=datasets,
        full_eval_datasets=full_eval_datasets,
    )

    write_json(summary_path, experiment_summary)
    cycle_columns = PHASE_SWITCH_COLUMNS if mode == "dagasvm_phase_switch" else CYCLE_TABLE_COLUMNS
    write_csv_rows(cycles_path, mode_outputs["cycle_table_rows"], fieldnames=cycle_columns)

    global_best = mode_outputs["global_best"] or {}
    best_val_metric = (global_best.get("metrics_val") or {}).get("metric")
    best_cycle = global_best.get("cycle")
    best_k = global_best.get("k_per_class")
    if mode == "full_svm":
        best_val_metric = baseline_val["metric"]
        best_cycle = None
        best_k = None

    manifest_entry = summarize_run_manifest_entry(
        run_id=str(artifact_layout["run_id"]),
        experiment_name=cfg.get("experiment_name", "exp"),
        run_root=run_root,
        log_path=log_path,
        summary_path=summary_path,
        seed=cfg.get("seed", 42),
        best_val_metric=best_val_metric,
        best_cycle=best_cycle,
        best_k_per_class=best_k,
        duration_sec=experiment_summary["experiment_duration_sec"],
    )
    manifest_entry["created_at"] = created_at
    append_csv_row(artifact_layout["index_csv"], manifest_entry)
    write_latest_run_pointer(artifact_layout["latest_json"], manifest_entry)
    append_csv_row(
        artifact_layout["history_table_csv"],
        experiment_summary["report_row"],
        fieldnames=HISTORY_TABLE_COLUMNS,
    )

    log.info(
        "Saved artifacts to %s and updated %s + %s",
        run_root,
        artifact_layout["index_csv"],
        artifact_layout["history_table_csv"],
    )


if __name__ == "__main__":
    main()
