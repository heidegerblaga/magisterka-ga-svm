import click
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.utils.random import set_global_seed
from ga.utils.io import ensure_dirs
from ga.utils.logger import get_logger
from ga.data.loader import prepare_train_val
from ga.models.svm import SVMConfig, train_eval_svm


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
    svm_cfg = SVMConfig(
        kernel=cfg["svm"]["kernel"],
        C=cfg["svm"]["C"],
        gamma=cfg["svm"]["gamma"],
        class_weight=cfg["svm"]["class_weight"],
    )
    score = train_eval_svm(X_train, y_train, X_val, y_val, svm_cfg, metric=cfg["fitness"]["metric"])
    log.info(f"Baseline SVM ({svm_cfg.kernel}) {cfg['fitness']['metric']}: {score:.4f}")

    log.info("Project scaffold OK. Implement GA loop in src/ga/...")

if __name__ == "__main__":
    main()
