#!/usr/bin/env python3
"""Create GA-SVM project scaffold in the current directory."""
import os, pathlib, textwrap

ROOT = pathlib.Path.cwd()

def write(path, content=""):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")

def main():
    # directories
    for d in [
        "configs", "data/raw", "data/processed", "notebooks",
        "artifacts/checkpoints", "artifacts/metrics", "artifacts/plots",
        "scripts", "src/ga/data", "src/ga/models", "src/ga/metrics",
        "src/ga/operators", "src/ga/utils", "tests",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    # files (minimal)
    write("pyproject.toml", """[build-system]
requires = ["setuptools", "wheel"]

[project]
name = "ga-svm-project"
version = "0.1.0"
description = "Genetic algorithm for opposing evolution of train/val sets for SVM"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pandas",
    "scikit-learn",
    "pyyaml",
    "matplotlib",
    "click",
    "joblib",
]

[tool.black]
line-length = 100
""")
    write("requirements.txt", """numpy
pandas
scikit-learn
pyyaml
matplotlib
click
joblib
""")
    write(".gitignore", """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*.so

# Virtual environments
.venv/
env/
venv/

# OS
.DS_Store

# Project artifacts
artifacts/
*.log

# Data (keep only small samples tracked)
data/processed/
data/raw/
!data/.gitkeep
""")
    write("README.md", """# GA-SVM Project

This repository implements a **genetic algorithm** to build training/validation splits for SVM
with an *adversarial (opposing) evolution* of the validation set.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/base.yaml
```

## Layout

- `configs/` — YAML configs for experiments
- `src/ga/` — source code (GA, operators, SVM utilities)
- `data/` — raw/processed data (ignored by git)
- `artifacts/` — outputs: checkpoints, metrics, plots
- `notebooks/` — exploration notebooks
- `tests/` — unit tests
- `scripts/` — CLI scripts
""")
    write("configs/base.yaml", """seed: 42
experiment_name: "baseline_ga_svm"
data:
  train_csv: "data/raw/train.csv"
  validation_csv: "data/raw/validation.csv"
  test_csv: "data/raw/test.csv"  # optional
  target: "label"
  stratify: true

svm:
  kernel: "rbf"        # rbf|linear|poly|sigmoid
  C: 1.0
  gamma: "scale"
  class_weight: "balanced"

ga:
  population_size: 50
  generations: 50
  elitism: 2
  tournament_k: 3
  crossover_prob: 0.8
  mutation_prob: 0.2
  val_fraction: 0.2           # fraction of data used for validation in a chromosome
  constraints:
    min_class_fraction: 0.05  # each class must have at least this fraction in train/val
  early_stopping:
    patience: 7
    min_delta: 1e-4

fitness:
  metric: "roc_auc"           # roc_auc|f1|balanced_accuracy|pr_auc
  penalties:
    val_size_penalty: 0.0
    class_imbalance_penalty: 0.5
    sv_ratio_penalty: 0.5     # penalty for too high/low % of support vectors
""")
    write("scripts/run_experiment.py", """import click
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
""")
    write("src/ga/__init__.py", "__all__ = []\n")
    write("src/ga/utils/io.py", "from pathlib import Path\n\n"
          "def ensure_dirs(paths):\n    for p in paths:\n        Path(p).mkdir(parents=True, exist_ok=True)\n")
    write("src/ga/utils/random.py", "import random, numpy as np\n\n"
          "def set_global_seed(seed: int):\n    random.seed(seed)\n    np.random.seed(seed)\n")
    write("src/ga/utils/logger.py", "import logging\n\n"
          "def get_logger():\n    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')\n    return logging.getLogger('ga')\n")
    write("tests/test_sanity.py", "def test_sanity():\n    assert 1+1==2\n")
    (ROOT/"data/.gitkeep").write_text("", encoding="utf-8")
    print("Scaffold created in", ROOT)

if __name__ == "__main__":
    main()
