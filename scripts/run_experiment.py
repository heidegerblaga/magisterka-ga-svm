import click
import yaml
from pathlib import Path
from ga.utils.random import set_global_seed
from ga.utils.io import ensure_dirs
from ga.utils.logger import get_logger

@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
def main(config):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    set_global_seed(cfg.get("seed", 42))
    log = get_logger()
    log.info(f"Loaded config: {cfg.get('experiment_name','exp')}")
    ensure_dirs(["artifacts/checkpoints", "artifacts/metrics", "artifacts/plots"])
    # TODO: hook up data load, GA loop and SVM training once modules are implemented
    log.info("Project scaffold OK. Implement GA loop in src/ga/...")

if __name__ == "__main__":
    main()
