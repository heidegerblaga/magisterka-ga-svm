# GA-SVM Project

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
