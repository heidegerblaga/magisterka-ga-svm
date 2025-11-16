import click
import yaml
from pathlib import Path
import sys
import json
import numpy as np

# ensure src on path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.utils.random import set_global_seed, limit_dataset  # zakładam, że masz tę funkcję (u Ciebie już działała)
from ga.utils.io import ensure_dirs
from ga.utils.logger import get_logger
from ga.data.loader import prepare_train_val
from ga.models.svm import SVMConfig, train_eval_svm, fit_svm_model, eval_fixed_svm
from ga.loop.engine import GAConfig, EarlyStopping, run_ga_k_per_class, ValEncodingConfig, run_val_ga

# >>> WAŻNE: używamy trybu "k na klasę" + stała walidacja
from ga.encoding import KPerClassConfig
from ga.fitness import FitnessConfig


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
def main(config):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))

    set_global_seed(cfg.get("seed", 42))
    log = get_logger()
    log.info(f"Loaded config: {cfg.get('experiment_name','exp')}")

    ensure_dirs(["artifacts/checkpoints", "artifacts/metrics", "artifacts/plots"])

    # 1) Dane: pula do TRAIN oraz STAŁA walidacja
    X_train, y_train, X_val, y_val = prepare_train_val(
        cfg["data"]["train_csv"],
        cfg["data"].get("validation_csv", ""),
        cfg["data"]["target"],
        stratify=cfg["data"].get("stratify", True),
        seed=cfg.get("seed", 42),
        header=cfg["data"].get("header", None),
        target_candidates=cfg["data"].get("target_candidates", []),
    )

    # (opcjonalnie) ograniczenie rozmiaru danych do szybkich testów
    try:
        X_train, y_train, X_val, y_val = limit_dataset(X_train, X_val, y_train, y_val)
    except Exception:
        pass

    # --- zewnętrzny zbiór testowy (nie używany przez GA) ---
    import pandas as pd

    test_csv = cfg["data"].get("test_csv", "")
    X_test, y_test = None, None
    if test_csv:
        # test.csv: brak nagłówka, ostatnia kolumna = label
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

    # 2) Baseline SVM (referencja)
    svm_cfg = SVMConfig(
        kernel=cfg["svm"]["kernel"],
        C=cfg["svm"]["C"],
        gamma=cfg["svm"]["gamma"],
        class_weight=cfg["svm"]["class_weight"],
    )
    baseline_metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")

    baseline_res = train_eval_svm(
        X_train, y_train, X_val, y_val,
        svm_cfg,
        metric=baseline_metric,
        return_dict=True,
    )
    log.info(
        f"Baseline SVM ({svm_cfg.kernel}) VAL {baseline_metric}: {baseline_res['metric']:.4f} | "
        f"all={baseline_res['all']} | "
        f"fit={baseline_res['fit_time']:.3f}s, score={baseline_res['score_time']:.3f}s"
    )

    if X_test is not None:
        baseline_test = train_eval_svm(
            X_train, y_train, X_test, y_test,
            svm_cfg,
            metric=baseline_metric,
            return_dict=True,
        )
        log.info(
            f"Baseline SVM ({svm_cfg.kernel}) TEST {baseline_metric}: {baseline_test['metric']:.4f} | "
            f"all={baseline_test['all']} | "
            f"fit={baseline_test['fit_time']:.3f}s, score={baseline_test['score_time']:.3f}s"
        )

    # 3) Konfiguracje GA + fitness + k-per-class
    kcfg = KPerClassConfig(
        k_per_class=cfg.get("encoding", {}).get("k_per_class", 50),
        random_state=cfg.get("seed", 42),
    )
    fcfg = FitnessConfig(
        metric=baseline_metric,
        class_imbalance_penalty=cfg.get("fitness", {}).get("class_imbalance_penalty", 0.0),
    )
    gcfg = GAConfig(
        population_size=cfg.get("ga", {}).get("population_size", 60),
        generations=cfg.get("ga", {}).get("generations", 20),
        tournament_k=cfg.get("ga", {}).get("tournament_k", 3),
        crossover_rate=cfg.get("ga", {}).get("crossover_rate", 0.8),
        mutation_rate=cfg.get("ga", {}).get("mutation_rate", 0.05),
        elitism=cfg.get("ga", {}).get("elitism", 2),
        early_stopping=cfg.get("ga", {}).get("early_stopping", {"patience": 0, "min_delta": 0.0}),
    )
    if isinstance(gcfg.early_stopping, dict):
        gcfg.early_stopping = EarlyStopping(**gcfg.early_stopping)

    # 4) URUCHOMIENIE GA (k-per-class, stała walidacja!)
    #    GA wybiera maskę TRAIN (True=do trenowania), a walidacja X_val,y_val jest stała w całym biegu.
    history, best_train_mask = run_ga_k_per_class(
        X_pool=X_train, y_pool=y_train,
        X_val=X_val, y_val=y_val,
        kcfg=kcfg, svm_cfg=svm_cfg, fcfg=fcfg, gcfg=gcfg,
        seed=cfg.get("seed", 42)
    )

    # 5) Artefakty + ocena best split
    Path("artifacts/metrics/history.json").write_text(
        json.dumps([float(x) for x in history]), encoding="utf-8"
    )
    np.save("artifacts/checkpoints/best_train_mask.npy", best_train_mask)

    # ewaluacja najlepszego splitu na stałej walidacji
    X_tr_best = X_train[best_train_mask]
    y_tr_best = y_train[best_train_mask]

    # mierz wszystkie metryki + czasy
    best_res = train_eval_svm(
        X_tr_best, y_tr_best, X_val, y_val,
        svm_cfg,
        metric=baseline_metric,
        return_dict=True,
    )
    log.info(f"GA(k/klasę) finished. Best gen fitness: {history[-1]:.4f}")
    log.info(
        f"Best-split SVM ({svm_cfg.kernel}) {baseline_metric}: {best_res['metric']:.4f} | "
        f"all={best_res['all']} | "
        f"fit={best_res['fit_time']:.3f}s, score={best_res['score_time']:.3f}s"
    )


    # --- faza Val-GA: druga populacja na walidacji, adwersarialne dobieranie walidacji ---

    # 1) ustalony SVM trenowany na best TRAIN
    clf_fixed = fit_svm_model(X_tr_best, y_tr_best, svm_cfg)

    # 2) konfiguracja dla Val-GA
    val_subset_fraction = cfg.get("val_ga", {}).get("subset_fraction", 0.5)
    vcfg = ValEncodingConfig(subset_fraction=val_subset_fraction)

    val_ga_cfg = GAConfig(
        population_size=cfg.get("val_ga", {}).get("population_size", 40),
        generations=cfg.get("val_ga", {}).get("generations", 20),
        tournament_k=cfg.get("val_ga", {}).get("tournament_k", 3),
        crossover_rate=cfg.get("val_ga", {}).get("crossover_rate", 0.8),
        mutation_rate=cfg.get("val_ga", {}).get("mutation_rate", 0.05),
        elitism=cfg.get("val_ga", {}).get("elitism", 2),
        early_stopping=cfg.get("val_ga", {}).get(
            "early_stopping",
            {"patience": 5, "min_delta": 1e-3},
        ),
    )
    if isinstance(val_ga_cfg.early_stopping, dict):
        val_ga_cfg.early_stopping = EarlyStopping(**val_ga_cfg.early_stopping)

    # 3) odpal Val-GA na X_val, y_val i ustalonym clf_fixed
    baseline_metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")
    history_val_ga, adv_val_mask = run_val_ga(
        X_val=X_val,
        y_val=y_val,
        clf=clf_fixed,
        vcfg=vcfg,
        gcfg=val_ga_cfg,
        metric=baseline_metric,
        seed=cfg.get("seed", 42),
    )

    # 4) raport: jak bardzo można zepsuć wynik SVM na walidacji
    X_val_adv = X_val[adv_val_mask]
    y_val_adv = y_val[adv_val_mask]
    adv_metric = eval_fixed_svm(clf_fixed, X_val_adv, y_val_adv, metric=baseline_metric)

    log.info(
        f"Val-GA finished. Best adversarial fitness: {history_val_ga[-1]:.4f} "
        f"(czyli minimalna {baseline_metric} ≈ {1.0 - history_val_ga[-1]:.4f})"
    )
    log.info(
        f"Adversarial VAL subset ({val_subset_fraction:.2f} frakcji walidacji) "
        f"{baseline_metric}: {adv_metric:.4f} "
        f"vs normal VAL (po GA-train) best-split {baseline_metric}: {best_res['metric']:.4f}"
    )


    if X_test is not None:
        best_test = train_eval_svm(
            X_tr_best, y_tr_best, X_test, y_test,
            svm_cfg,
            metric=baseline_metric,
            return_dict=True,
        )
        log.info(
            f"Best-split SVM ({svm_cfg.kernel}) TEST {baseline_metric}: {best_test['metric']:.4f} | "
            f"all={best_test['all']} | "
            f"fit={best_test['fit_time']:.3f}s, score={best_test['score_time']:.3f}s"
        )

        # --- 6) FAZA VAL-GA: adwersarialna walidacja ---



if __name__ == "__main__":
    main()
