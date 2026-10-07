# GA-SVM Project

This repository implements a **genetic algorithm** to select compact SVM training subsets (k-per-class)
and (optionally) evolve an adversarial validation subset.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/base.yaml
```

## Modes

`scripts/run_experiment.py` supports:

- `full_svm` — baseline SVM on full training set
- `gasvm` — GA-SVM with fixed `k_per_class`
- `dagasvm` — **adaptive DAGASVM**: starts from small `k_per_class` and increases K during a single run based on SV ratio and stability counters
- `dagasvm_phase` — phased T'/V' variant: optimize `T'` at fixed `K`, expand `V'` with hard samples, then increase `K`
- `dagasvm_valga` — alias for the phased T'/V' variant

Example (adaptive DAGASVM config fragment):

```yaml
algorithm_mode: dagasvm
daga:
  k_start: 10
  TSV: 0.25
  Ts: 3
  epsilon: 0.5
  k_max: 80
```

## Visualizations (single run)

After a run finishes, you can generate plots and per-generation frames for the best individual:

```bash
python scripts/plot_run_trace.py --summary artifacts/runs/<run_id>/metrics/summary.json
```

Outputs are saved under `artifacts/runs/<run_id>/plots/`.

## Layout

- `configs/` — YAML configs for experiments
- `src/ga/` — source code (GA, operators, SVM utilities)
- `data/` — raw/processed data (raw ignored by git)
- `artifacts/` — outputs: checkpoints, metrics, plots (ignored by git)
- `tests/` — unit tests
- `scripts/` — CLI scripts

## Badanie doboru V' (ewolucyjny vs losowy, pamięć V', zbiory 2D i wielowymiarowe)

Rozdział ról zbiorów w trybie `dagasvm_phase`:

- **V'** (aktualny podzbiór walidacyjny) — jedyne źródło metryk *sterujących* (fitness T', decyzje o
  rozszerzeniu V', wzroście K i zatrzymaniu).
- **V** (pełny zbiór walidacyjny) — pula kandydatów, z której wybierane są nowe elementy V', oraz źródło
  metryk *obserwacyjnych* liczonych po decyzji cyklu (nie sterują niczym). Test to wyłącznie obserwacja.
- K (`k_per_class`) rośnie najwyżej raz na cykl T'; przy 100% na V' K nie rośnie automatycznie.

Konfiguracja V' (sekcja `vprime`, wsteczna zgodność z `daga.val_selection_strategy`):

```yaml
vprime:
  selection: evolutionary   # evolutionary | heuristic | random
  memory: true              # czy V' pamięta elementy wybrane w poprzednich fazach
  mode: growing             # growing | fixed (fixed: losowe stałe V' o rozmiarze fixed_size, tylko selection=random)
  fixed_size: null
observation:
  full_validation: true     # obserwacja pełnego V po decyzji cyklu (nie steruje algorytmem)
  test: true
```

Odtworzenie badania (patrz `scripts/make_study_configs.py` — warianty: `evo_mem`, `evo_nomem`,
`rand_fixed`, `rand_growth`, `heur_mem`):

```bash
python scripts/generate_study_datasets.py && python scripts/tune_study_svm.py
python scripts/make_study_configs.py
python scripts/run_study_queue.py -d chessboard -d moons -d circles -d spiral -d classification2d -d digits \
    -v evo_mem -v evo_nomem -v rand_growth -v heur_mem -w 10
python scripts/run_study_queue.py -d chessboard -d moons -d circles -d spiral -d classification2d -d digits \
    -v rand_fixed -w 10          # po ukończeniu evo_mem (rozmiar = średni końcowy |V'| z evo_mem)
python scripts/aggregate_vprime_study.py && python scripts/plot_vprime_study.py
python scripts/generate_vprime_study_report.py   # -> output/pdf/raport_vprime_badanie.pdf
python -m pytest tests -q
```
