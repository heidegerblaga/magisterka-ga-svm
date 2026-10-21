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

## Badanie uzupełniające: reguła stopu BA(V') = 1,0 (`followup_stop_rule`)

Oddzielone od badania V' (własne dane, konfiguracje, artefakty, tabele, wykresy i raport PDF):
`data/followup_stop_rule/`, `configs/followup_stop_rule/`, `artifacts/followup_stop_rule/`, `reports/followup_stop_rule/`,
`output/pdf/raport_followup_regula_stopu.pdf`. Skrypty: `scripts/followup_stop_rule/`.

Nowe parametry fazowego DAGASVM (sekcja `daga`, domyślnie `0` = zachowanie historyczne, zweryfikowane na 24 starych przebiegach):

```yaml
daga:
  min_vprime_per_class: 50              # minimalna liczba próbek V' w KAŻDEJ klasie, zanim BA(V')=1,0 może zatrzymać przebieg
  perfect_recheck_expansions: 1         # obowiązkowe rozszerzenia V' po osiągnięciu minimum (i ponowna ocena T') przed takim stopem
  perfect_recheck_min_add_per_class: 50 # (opcjonalnie) minimalny przyrost V' na klasę przy takim ponownym sprawdzeniu
vprime:
  pool_budget_factor: null              # (opcjonalnie) wspólny budżet doboru: podpula kandydatów <= czynnik × liczba wybieranych elementów
```

BA(V') = 1,0 nadal nigdy nie zwiększa K; `cycles.csv` i `summary.json` zawierają wymagane minimum, jego osiągnięcie, liczbę ponownych
sprawdzeń, powód zablokowania stopu, kategorię zakończenia i rozbicie kosztu doboru V' (faktyczne wywołania modelu vs nominalne oceny Val-GA).

Odtworzenie badania:

```bash
python scripts/followup_stop_rule/generate_datasets.py && python scripts/followup_stop_rule/generate_datasets.py -s 2026 -d magic_full
python scripts/followup_stop_rule/tune_svm.py && python scripts/followup_stop_rule/tune_svm.py -s 2026 -d magic_full
python scripts/followup_stop_rule/make_configs.py
python -m pytest tests -q
python scripts/followup_stop_rule/verify_legacy.py            # stare konfiguracje => identyczne przebiegi
python scripts/followup_stop_rule/smoke.py -d moons --save     # smoke wszystkich 8 wariantów
python scripts/followup_stop_rule/run_queue.py --all -w 10     # pełny plan (wznawialny; rand_fixed czeka na evo_mem)
python scripts/followup_stop_rule/run_queue.py --include-pilot -d magic_full -w 3   # opcjonalny pilotaż pełnego MAGIC
python scripts/followup_stop_rule/progress.py                  # postęp i szacunek czasu
python scripts/followup_stop_rule/aggregate.py && python scripts/followup_stop_rule/plot.py
python scripts/followup_stop_rule/report.py                    # tylko gdy wszystkie serie kompletne (inaczej --allow-incomplete)
```

## Badanie uzupełniające: automat fazowy T′ ↔ V′ (`followup_phase_switch`)

Nowy tryb `algorithm_mode: dagasvm_phase_switch` (`src/ga/phase_switch.py`). Tryb `dagasvm_phase` i badanie `followup_stop_rule`
pozostają bez zmian. Jawne fazy `optimize_T_prime` / `optimize_V_prime` / `stop`:
* BA(V′) = 1,0 przełącza do fazy V′ i nie kończy algorytmu;
* faza V′ trwa, dopóki rośnie trudność przyrostów ΔV′;
* zastój T′ daje jedną próbę V′ na poziom K;
* rozmiar V′ ma jawny limit.

**Wersje.** v1 (smoke z 2026-10-14; serie bez prefiksu `v2_`) jest zamrożona: `AUDYT.md`, `RAPORT_SMOKE.md`, tabele i szablony v1 nie są nadpisywane.
v2 (serie `v2_*`) poprawia semantykę próby V′ (T′ dostaje szansę po zwiększeniu K), dodaje `vprime_difficulty_metric`, `max_vprime_expansions`,
liczniki kosztu i kolumny fazowe. Audyt: `reports/followup_phase_switch/AUDYT_v2.md`; raport: `reports/followup_phase_switch/v2/RAPORT_V2.md`.
Przebieg v1 odtwarza się na kodzie v2 flagami `retry_resolution: first_step` i `vprime_difficulty_metric: one_minus_metric`.

```yaml
algorithm_mode: dagasvm_phase_switch
fitness: {tprime_metric: balanced_accuracy, vprime_metric: balanced_accuracy}
phase_switch:
  vprime_increment_per_class: 10      # stały rozmiar przyrostu ΔV' (niezależny od K)
  max_vprime_fraction: 0.25           # limit |V'| = min(ułamek·|V|, max_vprime_rows)
  max_vprime_rows: null
  max_vprime_expansions: null         # globalny limit liczby rozszerzeń V'
  vprime_difficulty_metric: error_rate  # trudność ΔV' = odsetek błędów (albo one_minus_metric = 1 − vprime_metric)
  vprime_difficulty_min_delta: 0.01   # koniec fazy V', gdy trudność ΔV' przestaje rosnąć...
  vprime_difficulty_patience: 2       # ...przez tyle kolejnych przyrostów (średnia z okna vprime_difficulty_window)
  max_no_improvement_k_increases: 3   # zastój T'
  fitness_min_delta: 0.0005
  allow_one_vprime_retry_after_t_stall: true
  retry_resolution: after_k_increases # po próbie V' T' dostaje zwiększenia K; stop dopiero po ich wyczerpaniu (first_step = v1)
  retry_reference: same_vprime        # poziom odniesienia próby: ten sam, rozszerzony V' (literal = fitness sprzed próby)
  k_growth_requires_sv_signal: true   # istniejąca polityka K (wzrost tylko przy ρ_SV >= TSV)
  schedule_replay: null               # ścieżka do cycles.csv: odtworzenie harmonogramu (wariant rand_replay)
```

```bash
./.venv/bin/pytest -q                                        # cały zestaw (pythonpath ustawione w pyproject.toml)
python scripts/followup_phase_switch/run_queue.py --series v2_smoke_moons -d moons -v evo_switch -v rand_switch -v rand_replay \
    --seeds 42 --seeds 43 --seeds 44 -w 4     # każdy seed = osobna flaga; rand_replay startuje po evo_switch
python scripts/followup_phase_switch/run_queue.py --series v2_smoke_chess_r1000 -d chessboard -v evo_switch -v rand_switch -v rand_replay \
    --seeds 42 --seeds 43 --set max_vprime_rows=1000 -w 4
# podziały dodatkowe (tylko zbiory zależne od podziału; Szachownica ma stały plik data/raw):
python scripts/followup_phase_switch/run_queue.py --series v2_split2027 --split 2027 -d moons -d circles -d spiral -d classification2d -d digits -d magic \
    -v evo_switch -v rand_switch -v rand_replay --seeds 42 --seeds 43 --seeds 44 --seeds 45 --seeds 46 -w 4
python scripts/followup_phase_switch/aggregate.py --version v2 && python scripts/followup_phase_switch/make_configs.py
python scripts/followup_phase_switch/report_v2.py --pytest "<wynik pytest>"   # -> reports/followup_phase_switch/v2/RAPORT_V2.md
python scripts/followup_phase_switch/report_full.py   --pytest "<wynik pytest>"   # -> .../v2/RAPORT_PELNE_V2.md (seria v2_full)
python scripts/followup_phase_switch/report_splits.py --pytest "<wynik pytest>"   # -> .../v2/RAPORT_PODZIALY_V2.md (podziały 2026–2029)
```
