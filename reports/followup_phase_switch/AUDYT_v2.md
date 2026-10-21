# Audyt fazowego DAGASVM — stan na 2026-10-20 (v2 automatu T′ ↔ V′)

Dokument uzupełnia `AUDYT.md` (v1, 2026-10-14), którego nie zmieniam. Tu: (1) weryfikacja diagnozy starego
algorytmu na danych, (2) inwentarz użycia BA (pkt 12), (3) co zmieniło się względem v1 i dlaczego, (4) mapa
„punkt specyfikacji → kod → test”, (5) pytania do promotora.

Stan wyjściowy: tryb `dagasvm_phase_switch` **już istniał** (v1, `src/ga/phase_switch.py`). Przebudowa v2 poprawia go;
tryb `dagasvm_phase`, jego konfiguracje, artefakty i raport `followup_stop_rule` są nietknięte
(`src/ga/phase_engine.py`, `src/ga/vprime.py`, `src/ga/loop/engine.py`, `src/ga/models/svm.py` nie zmieniły się w v2).

## 1. Diagnoza starego algorytmu (`dagasvm_phase`, `src/ga/phase_engine.py`)

| pytanie | odpowiedź (miejsce w kodzie) |
|---|---|
| Gdzie jest optymalizacja T′? | `run_phase_experiment`, wywołanie `ga_runner(...)` (l. 432): jedno GA przy **stałym K** na cykl; populacja startowa losowa w każdym cyklu (brak warm startu — T′ z poprzedniego cyklu nie trafia do populacji). |
| Gdzie jest dobór nowych próbek V′? | `expand_vprime` (l. 550; `src/ga/vprime.py:301`) → `select_new_ids` → heurystyka / Val-GA (`run_val_ga`, `loop/engine.py:736`) / losowanie. Pula kandydatów = V \ V′. |
| Co powoduje rozszerzenie V′? | `decide_cycle` (l. 209) zwraca `expand_validation_only`, gdy: (a) BA(V′)=1 i V′ nie było „utwardzone” dla bieżącego K; (b) jest sygnał wzrostu K (ρ_SV ≥ TSV), ale V′ nie było jeszcze utwardzone dla tego K; (c) ρ_SV stabilne po utwardzeniu (do `Ts` razy). Flaga `validation_hardened_for_current_k` jest zerowana **przy każdym wzroście K** (l. 641), więc rytm to „rozszerz V′ → zwiększ K → rozszerz V′ …”. |
| Rozmiar przyrostu | `add_per_class = max(1, next_k − K)` (l. 533): przyrost jest **sprzężony z K**; nie ma limitu rozmiaru V′ ani budżetu. |
| Rola BA(V′) | `perfect_on_current_val` (l. 474): BA ≥ 1,0. Reguła historyczna: najpierw jedno rozszerzenie, potem **stop**. Reguła followup_stop_rule: bramka „minimalny rozmiar V′ + obowiązkowe ponowne sprawdzenie” (odrzucona przez promotora). K nie rośnie przy BA=1. |
| Rola ρ_SV, TSV | `has_growth_signal = ρ_SV ≥ TSV` (l. 475). Brak sygnału ⇒ **stop** (`stop_no_growth_signal`). Wielkość kroku K: `next_k_from_sv_ratio` (l. 155), `ΔK = 1 + |ρ_SV − TSV|/(1 − TSV)` (wartość bezwzględna — wzór ma sens tylko dla ρ_SV ≥ TSV). |
| Rola Ts, K | `Ts` = cierpliwość GA dla T′ (`EarlyStopping`) **oraz** liczba cykli „stabilnego ρ_SV”, przez które rozszerza się V′ zamiast zwiększać K. K = liczba próbek/klasę w T′. |
| Warunek przełączania faz | Brak jawnych faz. Faza wynika z kombinacji flag w `decide_cycle`. |
| Warunek zakończenia | Prawie zawsze ρ_SV < TSV albo K na limicie klasy; BA(V′)=1 praktycznie nigdy. |

### Liczby (przeliczone z `reports/followup_stop_rule/tables/per_cycle.csv` i `per_run.csv`; podział 2026, reguła `extended`, 4 warianty rosnące: evo_mem, evo_nomem, heur_mem, rand_growth)

| wielkość | wartość |
|---|---|
| przebiegi / cykle | 566 / 21 061 |
| cykle z rozszerzeniem V′ | 13 147 (62,4 %) |
| cykle z BA(V′) ≥ 1,0 | 235 (1,12 %); 72 % z nich to cykl 1; mediana \|V′\| w nich = 20 |
| rozszerzenia po BA(V′)=1 | 235 z 13 147 |
| mediana dodanych próbek na rozszerzenie | 28 (średnia 54,6) |
| końcowe \|V′\|, mediana evo_mem | Szachownica 3606, MAGIC 2253, Circles 1211, Digits 705, Spiral 465, Moons 264 |
| kategorie stopu (566 przebiegów) | no_k_growth_signal 400, k_limit 164, no_candidates 2, perfect_ba 0 |

Uwaga do `AUDYT.md` v1: tam podano „706 przebiegów: 499 / 204 / 2 / 1”. To liczba z 5 wariantami **razem z `rand_fixed`**
(140 przebiegów, stop mechaniczny). Przy 4 wariantach rosnących wartości są jak wyżej. Wnioski się nie zmieniają.

**Dlaczego V′ rosło do ≈3500 (Szachownica)?** (1) Rozszerzenie jest wymuszane flagą „utwardzenia dla K”, a nie sukcesem T′
(tylko 1,1 % cykli ma BA(V′)=1). (2) Przyrost = `next_k − K`, więc rośnie razem z K. (3) Nic nie ogranicza rozmiaru V′.
(4) Stop następuje dopiero przy ρ_SV < TSV albo K na limicie. Efekt: mediana 45 rozszerzeń i 17 wzrostów K na przebieg.

**`rand_fixed` jest wariantem uprzywilejowanym.** Konfiguracja (`scripts/followup_stop_rule/make_configs.py:43`) ma
`fixed_size_source_variant: evo_mem`; w wynikach \|V′\| = 3528 dla wszystkich seedów Szachownicy. Rozmiar pochodzi więc z wyniku
ewolucji. W v2 **nie używam `rand_fixed`**; porównanie ewolucja ↔ losowanie robi `rand_replay` (jawnie ten sam harmonogram).

## 2. Inwentarz użycia metryki (BA) — pkt 12

Notacja: **steruje** = wpływa na przebieg algorytmu; **obserwacja** = tylko zapis. T′ = fitness/ocena T′, V′/ΔV′ = zbiory sterujące.

| # | miejsce | metryka (konfiguracja) | zbiór | steruje? |
|---|---|---|---|---|
| 1 | fitness osobników GA dla T′: `loop/engine.py::_evaluate_population_k` → `fitness.py::evaluate_individual_fixed_train` → `svm.py::train_eval_svm(all_metrics=False)` | `fitness.tprime_metric` (BA domyślnie; ścieżka szybka tylko dla BA/accuracy) | V′ skumulowane | tak (selekcja GA, wczesne zatrzymanie GA, `target_fitness`) |
| 2 | wartość kroku T′ `f`: `phase_switch._metric(clf, X_vp, y_vp, tprime_metric)` | `tprime_metric` | V′ | tak (opanowanie V′, poprawa, zastój, postęp próby V′) |
| 3 | warunek T′ → V′: `perfect = f ≥ perfect_threshold` (1,0) | `tprime_metric` | V′ | tak |
| 4 | dobór ΔV′ (Val-GA): `loop/engine.py::_evaluate_population_val`, `_metric_from_preds` | `fitness.vprime_metric`; fitness osobnika = 1 − metryka na podzbiorze ΔV′ | ΔV′ (podzbiór puli V \ V′) | tak; heurystyka i losowanie nie używają metryki |
| 5 | trudność ΔV′ (koniec fazy V′) | `vprime_difficulty_metric`: `error_rate` = 1 − accuracy (domyślnie) albo `one_minus_metric` = 1 − `vprime_metric` | ΔV′ | tak |
| 6 | warunki zastoju i stopu: `fitness_min_delta`, `max_no_improvement_k_increases` | `tprime_metric` | V′ | tak |
| 7 | wybór modelu reprezentatywnego (`best_on_final_vprime`) | `tprime_metric` | V′ | tak (po zakończeniu, tylko V′) |
| 8 | kolumny `observed_full_validation_metric`, `observed_test_metric`, `global_best.metrics_*` | `tprime_metric` | pełne V, test | **nie** (obserwacja; test: `test_req10_*`) |
| 9 | `summary.objective_metric_name` (`scripts/run_experiment.py`) | v2: `tprime_metric` dla tego trybu (w v1 błędnie czytało `fitness.metric`) | — | nie |
| 10 | `reporting.py::build_report_row` | `fitness.metric`; wiersz raportu używa precision/recall/F1, nie tej wartości | pełny train, V, test | nie |

### Ciche podstawienia BA znalezione w kodzie (ryzyko dla późniejszego badania metryk)

| miejsce | zachowanie | zabezpieczenie w trybie `dagasvm_phase_switch` |
|---|---|---|
| `engine.py:687 _metric_from_preds` | nieznana metryka (np. `roc_auc`, `f1_macro`) → po cichu BA | `fitness.vprime_metric` ∈ {balanced_accuracy, accuracy, f1}, inne → `ValueError` |
| `svm.py:75–76 compute_prediction_metrics` | `roc_auc`/`pr_auc` bez `predict_proba` → BA | `roc_auc`/`pr_auc` wymagają `svm.probability=true`, inaczej `ValueError` |
| `svm.py:70` | wieloklasowe `pr_auc` → BA także przy `probability=true` | **v2:** `pr_auc` dla >2 klas → `ValueError` (test `test_multiclass_pr_auc_*`) |
| `engine.py:691` | `f1` w Val-GA jest binarne (`average="binary"`) | **v2:** `vprime_metric=f1` dla >2 klas → `ValueError` |

Stary tryb `dagasvm_phase` zachowuje te zachowania bez zmian (semantyka niezmieniona celowo).

### Ocena metodologiczna BA

* **Poprawność użycia:** BA jest właściwą domyślną metryką sterującą, bo V′ budowane adwersarialnie nie ma proporcji klas pełnego V
  (Val-GA wybiera najtrudniejsze próbki bez ograniczenia na klasę, patrz `delta_per_class`). BA waży klasy równo niezależnie od składu V′.
* **Próg 1,0:** BA = 1 ⇔ zero błędów na V′ (każda klasa obecna w V′ od startu), więc BA, accuracy i macro-F1 mają tu identyczny próg „opanowania”.
  **Dla ROC AUC tak nie jest** (AUC = 1 nie oznacza zera błędów przy progu decyzyjnym SVM): przy badaniu metryk `perfect_threshold`
  trzeba zdefiniować osobno dla każdej metryki, inaczej „opanowanie V′” znaczyłoby co innego.
* **Trudność przyrostu na błędzie klasyfikacji (accuracy), nie na BA:** wskazane w specyfikacji (`error_rate`). Przy ΔV′ zdominowanym
  przez jedną klasę odsetek błędów zależy od składu przyrostu, a BA byłoby równe recallowi tej klasy. Obie wartości są zapisywane.
* **Nieobciążoność:** BA(V′) to ocena na zbiorze, który zarazem steruje T′ (fitness) i jest budowany przeciw T′ (V′). To nie jest
  estymata uogólniania — dlatego test i pełne V są wyłącznie obserwacją (testy `test_req10_*`: śmieciowy obserwator i odwrócone etykiety testu nie zmieniają trajektorii).
* **Nasycenie:** fitness = 1,0 nie różnicuje modeli; GA dla T′ kończy się wtedy natychmiast (`ga_stop_at_perfect`).

## 3. Co zmieniło się w v2 względem v1 (`AUDYT.md` §4)

| zmiana | uzasadnienie (punkt specyfikacji) |
|---|---|
| **Próba V′ po zastoju: T′ ma szansę „także po zwiększeniu K”** (`retry_resolution: after_k_increases`, domyślnie). v1 kończyło algorytm już przy pierwszym kroku T′ bez postępu (`first_step`, nadal dostępne). | §8: „kontynuuj do osiągnięcia BA=1,0 albo wykrycia braku poprawy”; §9.3: „nawet po zwiększeniu K”. Brak postępu = `max_no_improvement_k_increases` zwiększeń K od próby V′ bez osiągnięcia poziomu odniesienia albo K nie może rosnąć. |
| `retry_reference` (`same_vprime` domyślnie / `literal`) | §9.2–9.3 dosłownie: „co najmniej zapamiętany fitness” (`literal`, na starym V′). Domyślnie porównanie na tym samym, rozszerzonym V′ (uzasadnienie S1/S2 w `AUDYT.md`: dosłowna reguła kończyłaby algorytm prawie zawsze, bo V′ po próbie jest trudniejsze). |
| `vprime_difficulty_metric` (`error_rate` domyślnie; v1: 1 − BA) | §6. |
| `max_vprime_expansions` (globalny limit) | §9, §10. |
| kolumny: `prev_phase`, `vprime_expansions_total`, `k_increases_total`, `vprime_limit_reason`, `retry_pending`, `retry_fail_count`, `model_metric_on_vprime`, koszt | §4, §7, §13. |
| koszt przez `CountingModel`: wywołania `predict` / `decision_function`, treningi SVM, nominalny vs faktyczny koszt T′ i Val-GA | §7. Zgodność liczników z faktycznymi wywołaniami `SVC.predict`/`decision_function` sprawdza test z podsłuchem. |
| `vprime.score_candidate_pool` domyślnie `false` w tym trybie | diagnostyka błędności całej puli nie steruje niczym, a przy doborze losowym kosztowała `predict` po całej puli; wyłączona, by koszt losowego i ewolucyjnego był porównywalny. |
| ochrona przed ciszą metryk (`pr_auc` >2 klas, `f1` >2 klas); `objective_metric_name` z `tprime_metric` | §12. |
| `pyproject.toml`: `[tool.pytest.ini_options] pythonpath` | `./.venv/bin/pytest -q` przerywało się na 2 plikach z samego powodu braku katalogu głównego w `sys.path` (nie testy). |

**Odtwarzalność v1:** przebieg v1 (Moons, evo, seed 42, seria `smoke_moons_f025_svgate`) uruchomiony kodem v2 z
`retry_resolution=first_step` i `vprime_difficulty_metric=one_minus_metric` daje `cycles.csv` identyczny we wszystkich 60 wspólnych kolumnach
(45 kroków; poza czasem). Artefakty i tabele v1 nie były nadpisywane; wyniki v2 są w seriach `v2_*` i w `reports/followup_phase_switch/v2/`.

## 4. Mapa: punkt specyfikacji → kod → test

| § | wymaganie | kod (`src/ga/phase_switch.py`) | test (`tests/test_phase_switch.py`) |
|---|---|---|---|
| 4 | jawne fazy, rekord cyklu | `PHASE_*`, `PHASE_SWITCH_COLUMNS`, `PhaseSwitchController` | `test_cli_phase_switch_writes_columns_and_vprime_fraction` |
| 5 | BA(V′)=1 → faza V′, nie stop; K ≤ 1×/cykl | `decide_after_tprime` | `test_req01_*`, `test_perfect_vprime_*`, `test_req05_*` |
| 6 | V′ → T′ po plateau trudności ΔV′ | `DifficultyTracker`, `decide_after_vprime` | `test_difficulty_*`, `test_req02_*`, `test_difficulty_metric_option_*` |
| 7 | ΔV′ tylko z V \ V′, fitness na ΔV′, T′ na pełnym V′, koszt | `expand_vprime` (pula = kandydaci), `CountingModel`, `ga_runner(X_val=V′)` | `test_req03_*`, `test_req03b_*`, `test_req04_*`, `test_cost_counters_*` |
| 8–9 | zwiększ K ≤ 1×, zastój, jedna próba V′, stop po próbie | `PhaseSwitchController.on_tprime` | `test_three_fruitless_*`, `test_retry_resolved_*`, `test_literal_retry_*`, `test_req08_*` |
| 9 | brak zapętlenia, limity cykli / rozszerzeń / kandydatów | `max_cycles`, `max_vprime_expansions`, `expansion_block_reason` | `test_req09_*`, `test_scripted_controller_always_terminates` |
| 10 | kontrola rozmiaru V′ | `vprime_cap_rows`, `plan_increment_per_class` | `test_req13_*`, `test_vprime_never_exceeds_*`, `test_vprime_cap_and_increment_planning` |
| 11 | `\|V′\|/\|V\|` | `vprime_fraction` | `test_req11_*` |
| 12 | metryki, brak testu w sterowaniu | `fitness.tprime_metric/vprime_metric` | `test_settings_*`, `test_req10_*`, `test_multiclass_*`, `test_cli_objective_metric_name_*` |
| 3 | ten sam harmonogram evo ↔ losowanie | `load_schedule_replay`, `_replay_tprime` | `test_req06_*`, `test_req12_*`, `test_random_replay_reproduces_*` |
| 2 | stary tryb działa | — (`phase_engine.py` nietknięty) | `test_req14_*`, `test_phase_engine.py`, `test_stop_rule.py` |

## 5. Pytania wymagające decyzji promotora

1. **Próg „opanowania V′”.** W smoke v2 (Moons + Szachownica, warianty z własnym automatem: 68 wejść do fazy V′) „opanowanie” (BA(V′)=1) wywołało 29 wejść,
   zastój T′ (próba po 3 zwiększeniach K bez poprawy) — 39. Przy doborze ewolucyjnym BA(V′)=1 zdarza się tylko przy małych V′: Moons wyłącznie na V′ startowym (20 próbek),
   Szachownica przy 20–140 próbkach; potem już nie. Czy próg ma pozostać 1,0, czy ma być < 1 (np. 1 − ε), tak by „opanowanie” zdarzało się także przy większych V′?
2. **Rola ρ_SV.** Domyślnie `k_growth_requires_sv_signal: true`: K rośnie tylko przy ρ_SV ≥ TSV (polityka DAGA); ρ_SV < TSV traktowane jest jako
   pomocniczy sygnał braku poprawy (§11) i daje próbę V′ bez dalszego zwiększania K (a po nieudanej próbie — stop, bo K nie może rosnąć).
   Smoke v2, Szachownica, limit \|V′\| = 1000, ewolucja, n = 2: **z bramką** K ≈ 890, \|V′\| 320–720, BA test 0,9745, ≈ 170 s; **bez bramki** K ≈ 5500, \|V′\| 740–1000,
   BA test 0,9868, ≈ 600 s (pełny SVM: 0,9885). Czyli bramka obcina jakość o ≈ 0,012 BA, ale kosztuje 3,5× mniej czasu. Bez bramki wzór ΔK działa odwrotnie poniżej TSV
   (`AUDYT.md` S3: wartość bezwzględna |ρ_SV − TSV|). Czy zachować bramkę, czy zmienić wzór na ΔK?
3. **Dosłowne „co najmniej zapamiętany fitness” (§9.3).** Domyślnie porównuję na tym samym, rozszerzonym V′ (`retry_reference: same_vprime`).
   Wariant dosłowny (`literal`) jest zaimplementowany; czy promotor woli go?
4. **Wybór modelu reprezentatywnego.** `best_on_final_vprime` (najlepszy fitness na końcowym V′) vs `last_tprime_step`.
5. **Limity rozmiaru V′ dla różnych zbiorów.** Jedno `max_vprime_fraction` nie daje porównywalnego \|V′\| (|V| od 719 do 27 853); obok niego jest `max_vprime_rows`.
