import click
import yaml
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ensure src on path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.utils.random import set_global_seed, limit_dataset
from ga.utils.io import ensure_dirs
from ga.utils.logger import get_logger
from ga.data.loader import prepare_train_val
from ga.models.svm import (
    SVMConfig,
    train_eval_svm,
    fit_svm_model,
    eval_fixed_svm,
)
from ga.encoding import KPerClassConfig
from ga.fitness import FitnessConfig
from ga.loop.engine import (
    GAConfig,
    EarlyStopping,
    run_ga_k_per_class,
    ValEncodingConfig,
    run_val_ga,
)

from ga.profiling import get_profile
import mlflow

mlflow.set_tracking_uri("file:///C:/Users/skyri/Desktop/magisterka/ga_svm_project_scaffold/mlruns")
mlflow.set_experiment("ga_svm_magisterka")


def _log_svm_profile(log):
    profile = get_profile()

    # do MLflow
    mlflow.log_metric("svm.n_train_eval", profile.n_train_eval)
    mlflow.log_metric("svm.n_fixed_eval", profile.n_fixed_eval)
    mlflow.log_metric("svm.val_points_evaluated", profile.val_points_evaluated)
    mlflow.log_metric("svm.approx_cost", profile.approx_cost)

    # do loggera konsolowego
    log.info(
        "SVM cost profile: "
        f"train_eval={profile.n_train_eval}, "
        f"fixed_eval={profile.n_fixed_eval}, "
        f"val_points_evaluated={profile.val_points_evaluated}, "
        f"approx_cost={profile.approx_cost}"
    )


# =====================================================================
# 1) ŁADOWANIE DANYCH
# =====================================================================

def load_datasets(cfg, log):
    """Wczytuje train/val przez prepare_train_val + zewnętrzny test (bez nagłówka, ostatnia kolumna = label).
       Zwraca: X_train, y_train, X_val_full, y_val_full, X_test, y_test
    """
    X_train, y_train, X_val, y_val = prepare_train_val(
        cfg["data"]["train_csv"],
        cfg["data"].get("validation_csv", ""),
        cfg["data"]["target"],
        stratify=cfg["data"].get("stratify", True),
        seed=cfg.get("seed", 42),
        header=cfg["data"].get("header", None),
        target_candidates=cfg["data"].get("target_candidates", []),
    )

    # ograniczenie rozmiaru do szybkich testów
    X_train, y_train, X_val, y_val = limit_dataset(X_train, X_val, y_train, y_val)
    log.info(f"Dataset reduced for test: {len(X_train)} train + {len(X_val)} val rows")

    # pełna pula walidacyjna (pod kumulację)
    X_val_full = X_val
    y_val_full = y_val

    # zewnętrzny test: brak nagłówka, ostatnia kolumna = label
    test_csv = cfg["data"].get("test_csv", "")
    X_test, y_test = None, None
    if test_csv:
        df_test = pd.read_csv(test_csv, header=None)
        arr = df_test.to_numpy()
        if arr.shape[1] < 2:
            raise ValueError(f"test_csv={test_csv} ma za mało kolumn: {arr.shape[1]}")
        X_test = arr[:, :-1]
        y_test = arr[:, -1]
        log.info(
            f"Loaded external TEST set from {test_csv}: "
            f"{X_test.shape[0]} rows, {X_test.shape[1]} features, last col as label"
        )
    else:
        log.warning("No external test_csv provided in config; external test will be skipped.")

    return X_train, y_train, X_val_full, y_val_full, X_test, y_test


# =====================================================================
# 2) BASELINE
# =====================================================================

def run_baseline_phase(X_train, y_train, X_val, y_val, X_test, y_test, svm_cfg, cfg, log):
    """Liczy baseline na walidacji + (opcjonalnie) na teście."""
    metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")

    res_val = train_eval_svm(
        X_train, y_train, X_val, y_val,
        svm_cfg,
        metric=metric,
        return_dict=True,
    )

    metrics = res_val["all"]
    mlflow.log_metric("baseline.fit_metric", res_val["metric"])
    mlflow.log_metric("baseline.fit_time", res_val["fit_time"])
    mlflow.log_metric("baseline.score_time", res_val["score_time"])

    for name, val in metrics.items():
        mlflow.log_metric(f"baseline.{name}", val)

    log.info(
        f"Baseline SVM ({svm_cfg.kernel}) VAL {metric}: {res_val['metric']:.4f} | "
        f"all={res_val['all']} | fit={res_val['fit_time']:.3f}s, score={res_val['score_time']:.3f}s"
    )

    res_test = None
    if X_test is not None:
        res_test = train_eval_svm(
            X_train, y_train, X_test, y_test,
            svm_cfg,
            metric=metric,
            return_dict=True,
        )
        log.info(
            f"Baseline SVM ({svm_cfg.kernel}) TEST {metric}: {res_test['metric']:.4f} | "
            f"all={res_test['all']} | fit={res_test['fit_time']:.3f}s, score={res_test['score_time']:.3f}s"
        )
    return res_val, res_test


# =====================================================================
# 3) PĘTLA CYKLI: TRAIN-GA + VAL-GA + K-SCHEDULE + KUMULACJA WALIDACJI
# =====================================================================

def build_ga_config_from_cfg(base_key: str, cfg) -> GAConfig:
    """Pomocniczo: GAConfig z sekcji cfg[base_key] (np. 'ga' lub 'val_ga')."""
    section = cfg.get(base_key, {})
    gcfg = GAConfig(
        population_size=section.get("population_size", 60),
        generations=section.get("generations", 20),
        tournament_k=section.get("tournament_k", 3),
        crossover_rate=section.get("crossover_rate", 0.8),
        mutation_rate=section.get("mutation_rate", 0.05),
        elitism=section.get("elitism", 2),
        early_stopping=section.get(
            "early_stopping",
            {"patience": 0, "min_delta": 0.0},
        ),
    )
    if isinstance(gcfg.early_stopping, dict):
        gcfg.early_stopping = EarlyStopping(**gcfg.early_stopping)
    return gcfg


def run_train_val_cycles(
    X_train,
    y_train,
    X_val_full,
    y_val_full,
    X_test,
    y_test,
    svm_cfg,
    cfg,
    log,
):
    """Główna pętla eksperymentu:
       - kumulacja walidacji
       - harmonogram k
       - w każdym cyklu: TRAIN-GA (k-per-class) + Val-GA (adwersarialna walidacja)
       - ewaluacja na VAL i TEST
    """

    # --- fitness config ---
    baseline_metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")
    fcfg = FitnessConfig(
        metric=baseline_metric,
        class_imbalance_penalty=cfg.get("fitness", {}).get("class_imbalance_penalty", 0.0),
    )

    # --- harmonogram k ---
    ks_cfg = cfg.get("k_schedule", {})
    num_cycles = ks_cfg.get("num_cycles", 1)
    k_start = ks_cfg.get("k_start", cfg.get("encoding", {}).get("k_per_class", 20))
    k_mult = ks_cfg.get("k_multiplier", 1.5)

    # --- kumulacja walidacji ---
    val_cum_cfg = cfg.get("val_cumulation", {})
    initial_val_fraction = val_cum_cfg.get("initial_fraction", 0.5)

    n_val_full = X_val_full.shape[0]
    n_init = max(1, int(round(initial_val_fraction * n_val_full)))

    rng = np.random.default_rng(cfg.get("seed", 42))
    global_val_mask = np.zeros(n_val_full, dtype=bool)
    init_idx = rng.choice(n_val_full, size=n_init, replace=False)
    global_val_mask[init_idx] = True

    log.info(
        f"Początkowa walidacja: {global_val_mask.sum()} / {n_val_full} próbek "
        f"(initial_fraction={initial_val_fraction})"
    )

    current_k = k_start
    global_best_val = -1.0
    global_best_info = {}

    for cycle in range(num_cycles):
        log.info("=" * 60)
        log.info(f"FAZA TRAIN-GA (cykl {cycle+1}/{num_cycles}), k_per_class≈{int(round(current_k))}")

        # aktualna (skumulowana) walidacja
        X_val_curr = X_val_full[global_val_mask]
        y_val_curr = y_val_full[global_val_mask]
        log.info(
            f"[cykl {cycle+1}] aktualny rozmiar walidacji: "
            f"{X_val_curr.shape[0]} / {X_val_full.shape[0]} próbek"
        )

        # --- TRAIN-GA ---
        kcfg = KPerClassConfig(
            k_per_class=int(round(current_k)),
            random_state=cfg.get("seed", 42) + cycle,
        )
        gcfg_train = build_ga_config_from_cfg("ga", cfg)

        history_train, best_train_mask = run_ga_k_per_class(
            X_pool=X_train,
            y_pool=y_train,
            X_val=X_val_curr,
            y_val=y_val_curr,
            kcfg=kcfg,
            svm_cfg=svm_cfg,
            fcfg=fcfg,
            gcfg=gcfg_train,
            seed=cfg.get("seed", 42) + cycle,
        )

        X_tr_best = X_train[best_train_mask]
        y_tr_best = y_train[best_train_mask]

        best_res_val = train_eval_svm(
            X_tr_best, y_tr_best, X_val_curr, y_val_curr,
            svm_cfg,
            metric=baseline_metric,
            return_dict=True,
        )
        log.info(
            f"[TRAIN-GA cykl {cycle+1}] best VAL {baseline_metric}: "
            f"{best_res_val['metric']:.4f} | all={best_res_val['all']} | "
            f"fit={best_res_val['fit_time']:.3f}s, score={best_res_val['score_time']:.3f}s"
        )

        # aktualizacja globalnego best
        if best_res_val["metric"] > global_best_val:
            global_best_val = best_res_val["metric"]
            global_best_info = {
                "cycle": cycle + 1,
                "k_per_class": int(round(current_k)),
                "train_mask": best_train_mask.copy(),
                "metrics_val": best_res_val,
            }

        # ewaluacja TEŚCIOWA dla tego cyklu
        if X_test is not None:
            best_res_test = train_eval_svm(
                X_tr_best, y_tr_best, X_test, y_test,
                svm_cfg,
                metric=baseline_metric,
                return_dict=True,
            )
            log.info(
                f"[TRAIN-GA cykl {cycle+1}] TEST {baseline_metric}: "
                f"{best_res_test['metric']:.4f} | all={best_res_test['all']} | "
                f"fit={best_res_test['fit_time']:.3f}s, score={best_res_test['score_time']:.3f}s"
            )

        # --- VAL-GA: adwersarialna walidacja na ustalonym SVM ---
        log.info(f"FAZA VAL-GA (cykl {cycle+1}/{num_cycles}), k_per_class≈{int(round(current_k))}")
        clf_fixed = fit_svm_model(X_tr_best, y_tr_best, svm_cfg)

        vcfg = ValEncodingConfig(
            subset_fraction=cfg.get("val_ga", {}).get("subset_fraction", 0.5)
        )
        gcfg_val = build_ga_config_from_cfg("val_ga", cfg)

        # kandydaci do inkrementu: te próbki walidacji, które JESZCZE nie są w global_val_mask
        candidate_mask = ~global_val_mask
        if not np.any(candidate_mask):
            log.info(f"[VAL-GA cykl {cycle+1}] brak kandydatów do inkrementu walidacji.")
            history_val_ga = np.array([0.0])
            adv_val_mask_global = np.zeros_like(global_val_mask, dtype=bool)
        else:
            X_val_candidates = X_val_full[candidate_mask]
            y_val_candidates = y_val_full[candidate_mask]

            history_val_ga, adv_val_mask_local = run_val_ga(
                X_val=X_val_candidates,
                y_val=y_val_candidates,
                clf=clf_fixed,
                vcfg=vcfg,
                gcfg=gcfg_val,
                metric=baseline_metric,
                seed=cfg.get("seed", 42) + 1000 + cycle,
            )

            # mapujemy maskę lokalną (nad kandydatami) na maskę globalną
            adv_val_mask_global = np.zeros_like(global_val_mask, dtype=bool)
            adv_val_mask_global[candidate_mask] = adv_val_mask_local

        # ewaluacja adwersarialnego przyrostu (raport)
        if np.any(adv_val_mask_global):
            X_val_adv = X_val_full[adv_val_mask_global]
            y_val_adv = y_val_full[adv_val_mask_global]
            adv_metric = eval_fixed_svm(clf_fixed, X_val_adv, y_val_adv, metric=baseline_metric)
        else:
            adv_metric = None

        if history_val_ga.size > 0:
            log.info(
                f"[VAL-GA cykl {cycle+1}] best adversarial fitness: {history_val_ga[-1]:.4f} "
                f"(~ min {baseline_metric} ≈ {1.0 - history_val_ga[-1]:.4f})"
            )
        if adv_metric is not None:
            log.info(
                f"[VAL-GA cykl {cycle+1}] adversarial INCREMENT {baseline_metric}: {adv_metric:.4f} "
                f"vs normal VAL (po TRAIN-GA) {baseline_metric}: {best_res_val['metric']:.4f}"
            )

        # --- KUMULACJA WALIDACJI ---
        before = int(global_val_mask.sum())
        global_val_mask = np.logical_or(global_val_mask, adv_val_mask_global)
        after = int(global_val_mask.sum())
        log.info(
            f"[VAL-GA cykl {cycle+1}] kumulacja walidacji: {before} -> {after} próbek "
            f"(dodano {after - before})"
        )

        # --- PRZEŁĄCZNIK FAZ: zwiększamy k na kolejny cykl ---
        prev_k = int(round(current_k))
        current_k = current_k * k_mult
        log.info(
            f"Przejście VAL -> TRAIN: k_per_class {prev_k} -> ~{int(round(current_k))} "
            f"(mnożnik={k_mult})"
        )

    # podsumowanie globalnie najlepszego wyniku na walidacji
    if global_best_info:
        log.info(
            f"GLOBALNIE najlepszy VAL {baseline_metric}: "
            f"{global_best_info['metrics_val']['metric']:.4f} "
            f"w cyklu {global_best_info['cycle']} przy k_per_class={global_best_info['k_per_class']}"
        )


# =====================================================================
# 4) GŁÓWNA FUNKCJA – TERAZ BARDZO CIENKA
# =====================================================================
def _flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _log_config_to_mlflow(cfg: dict, config_path: str):
    flat = _flatten_dict(cfg)
    for k, v in flat.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            pass  # wartości złożone/logów nie logujemy jako param

    # log sam YAML jako artefakt
    mlflow.log_artifact(config_path, artifact_path="config")


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
def main(config):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    experiment_name = cfg.get("experiment_name", Path(config).stem)

    # ----------------------------------------------------------------------
    # START MLflow
    # ----------------------------------------------------------------------
    mlflow.set_experiment("ga_svm_magisterka")

    with mlflow.start_run():
        # log config YAML + flatten paramów
        _log_config_to_mlflow(cfg, config)

        set_global_seed(cfg.get("seed", 42))
        log = get_logger()
        log.info(f"Loaded config: {cfg.get('experiment_name', 'exp')}")

        ensure_dirs(["artifacts/checkpoints", "artifacts/metrics", "artifacts/plots"])

        # ------------------------------------------------------------------
        # 1) dane
        # ------------------------------------------------------------------
        X_train, y_train, X_val_full, y_val_full, X_test, y_test = load_datasets(cfg, log)

        # ------------------------------------------------------------------
        # 2) konfiguracja SVM
        # ------------------------------------------------------------------
        svm_cfg = SVMConfig(
            kernel=cfg["svm"]["kernel"],
            C=cfg["svm"]["C"],
            gamma=cfg["svm"]["gamma"],
            class_weight=cfg["svm"]["class_weight"],
        )

        # ------------------------------------------------------------------
        # 3) baseline
        # ------------------------------------------------------------------
        baseline_res = run_baseline_phase(
            X_train, y_train,
            X_val_full, y_val_full,
            X_test, y_test,
            svm_cfg, cfg, log
        )

        # baseline jako metryki MLflow
        if isinstance(baseline_res, dict):
            mlflow.log_metric("baseline.metric", baseline_res["metric"])
            mlflow.log_metric("baseline.fit_time", baseline_res["fit_time"])
            mlflow.log_metric("baseline.score_time", baseline_res["score_time"])
            for k, v in baseline_res["all"].items():
                mlflow.log_metric(f"baseline.{k}", float(v))

        # ------------------------------------------------------------------
        # 4) cykle TRAIN-GA + VAL-GA
        # ------------------------------------------------------------------
        run_train_val_cycles(
            X_train=X_train,
            y_train=y_train,
            X_val_full=X_val_full,
            y_val_full=y_val_full,
            X_test=X_test,
            y_test=y_test,
            svm_cfg=svm_cfg,
            cfg=cfg,
            log=log,
        )

        # ------------------------------------------------------------------
        # 5) profil kosztu
        # ------------------------------------------------------------------

        _log_svm_profile(log)


if __name__ == "__main__":
    main()
