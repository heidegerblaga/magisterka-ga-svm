import click
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.utils.random import set_global_seed,limit_dataset
from ga.utils.io import ensure_dirs
from ga.utils.logger import get_logger
from ga.data.loader import prepare_train_val
from ga.models.svm import SVMConfig, train_eval_svm
from ga.encoding import EncodingConfig
from ga.fitness import FitnessConfig
from ga.loop.engine import GAConfig, run_ga
from ga.utils.artifacts import save_ga_artifacts


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
def main(config):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))

    set_global_seed(cfg.get("seed", 42))
    log = get_logger()
    log.info(f"Loaded config: {cfg.get('experiment_name','exp')}")
    ensure_dirs(["artifacts/checkpoints", "artifacts/metrics", "artifacts/plots"])
    X_train, y_train, X_val, y_val = prepare_train_val(
        cfg["data"]["train_csv"],
        cfg["data"].get("validation_csv", ""),
        cfg["data"]["target"],
        stratify=cfg["data"].get("stratify", True),
        seed=cfg.get("seed", 42),
        header=cfg["data"].get("header", None),
        target_candidates=cfg["data"].get("target_candidates", []),
    )

    X_train,y_train,X_val,y_val = limit_dataset(X_train,X_val,y_train,y_val)

    svm_cfg = SVMConfig(
        kernel=cfg["svm"]["kernel"],
        C=cfg["svm"]["C"],
        gamma=cfg["svm"]["gamma"],
        class_weight=cfg["svm"]["class_weight"],
    )

    score = train_eval_svm(X_train, y_train, X_val, y_val, svm_cfg, metric=cfg["fitness"]["metric"])
    log.info(f"Baseline SVM ({svm_cfg.kernel}) {cfg['fitness']['metric']}: {score:.4f}")

    enc_cfg = EncodingConfig(
        val_fraction=cfg.get("encoding", {}).get("val_fraction", 0.2),
        random_state=cfg.get("seed", 42),
    )
    fcfg = FitnessConfig(
        metric=cfg.get("fitness", {}).get("metric", "balanced_accuracy"),
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
        from ga.loop.engine import EarlyStopping
        gcfg.early_stopping = EarlyStopping(**gcfg.early_stopping)

    # uruchom GA
    history, best_mask = run_ga(
        X_train, y_train, enc_cfg=enc_cfg, svm_cfg=svm_cfg, fcfg=fcfg, gcfg=gcfg, seed=cfg.get("seed", 42)
    )

    save_ga_artifacts(
        history=history,
        best_mask=best_mask,
        X=X_train,
        y=y_train,
        svm_cfg=svm_cfg,
        metric=cfg["fitness"]["metric"],
        log=log,
    )

    log.info("Project scaffold OK. Implement GA loop in src/ga/...")

if __name__ == "__main__":
    main()
