# followup_phase_switch: automat T′ ↔ V′ — etap smoke i krótkich serii

Raport generowany przez `scripts/followup_phase_switch/report.py` z tabel w `reports/followup_phase_switch/tables/`.
**Nie jest to pełny eksperyment.** Wykonano wyłącznie testy jednostkowe, smoke testy (Moons, Szachownica, 2 seedy na
pozostałych zbiorach) i krótką serię dla Moons (10 seedów). Przebiegów: 111, kroków automatu: 4664.
Audyt i lista sprzeczności specyfikacji: `reports/followup_phase_switch/AUDYT.md`.

## 1. Pliki

| plik | zmiana |
|---|---|
| `src/ga/phase_switch.py` | **nowy**: tryb `dagasvm_phase_switch` (ustawienia, czyste funkcje decyzyjne, kontroler automatu, pętla, replay) |
| `src/ga/loop/engine.py` | `run_ga_k_per_class_with_trace`: opcjonalne `seed_masks` (warm start T′) i `target_fitness`; przy `None` identyczne zachowanie |
| `scripts/run_experiment.py` | dispatch `algorithm_mode: dagasvm_phase_switch`; `cycles.csv` z kolumnami `PHASE_SWITCH_COLUMNS` |
| `src/ga/reporting.py` | nowy tryb dodany do zbiorów trybów (maska T′ w wierszu raportu) |
| `tests/test_phase_switch.py` | **nowy**: 30 funkcji testowych (część parametryzowana) |
| `scripts/followup_phase_switch/*` | **nowe**: `ps_common.py`, `run_queue.py`, `aggregate.py`, `make_configs.py`, `report.py` |
| `configs/followup_phase_switch/` | **nowe**: szablony konfiguracji i `study_spec.yaml` |
| `artifacts/`, `reports/followup_phase_switch/` | **nowe**: przebiegi, tabele, ten raport, `AUDYT.md` |

`ga.phase_engine` (tryb `dagasvm_phase`), konfiguracje, artefakty i raport `followup_stop_rule` są nietknięte.
Przebieg followup_stop_rule (Moons, evo_mem, seed 42) uruchomiony nowym kodem odtwarza `cycles.csv` bit w bit (poza czasami).

## 2. Automat fazowy

Jeden krok = jedno działanie: **jedno** uruchomienie GA dla T′ albo **jedno** rozszerzenie V′.

| z fazy | warunek (wyłącznie V′/ΔV′, T′, K, pula kandydatów) | do fazy | K |
|---|---|---|---|
| `optimize_T_prime` | BA(V′) ≥ 1,0 i V′ można rozszerzyć | `optimize_V_prime` | bez zmian |
| `optimize_T_prime` | BA(V′) ≥ 1,0 i V′ na limicie / brak kandydatów | **stop** `stop_perfect_on_max_vprime` | bez zmian |
| `optimize_T_prime` | BA(V′) < 1,0, brak zastoju, ρ_SV ≥ TSV | `optimize_T_prime` | **+1 wzrost** (wzór DAGA) |
| `optimize_T_prime` | zastój: `max_no_improvement_k_increases` kolejnych wzrostów K bez poprawy o `fitness_min_delta`, albo ρ_SV < TSV, albo K na limicie klasy; próba przy tym K niewykonana i V′ rozszerzalne | `optimize_V_prime` (próba po zastoju) | bez zmian |
| `optimize_T_prime` | zastój, próba przy tym K już była / V′ nierozszerzalne | **stop** `stop_tprime_no_progress` / `stop_no_k_growth_signal` / `stop_k_limit` | — |
| `optimize_T_prime` (1. krok po próbie) | brak poprawy względem najlepszego T′ sprzed próby **na tym samym V′** | **stop** `stop_no_progress_after_vprime_retry` | — |
| `optimize_V_prime` | trudność ΔV′ (średnia z okna) nadal rośnie o ≥ `vprime_difficulty_min_delta` | `optimize_V_prime` | bez zmian |
| `optimize_V_prime` | `vprime_difficulty_patience` przyrostów bez wzrostu / limit |V′| / brak kandydatów / `max_vprime_expansions_per_phase` | `optimize_T_prime` (pełne, skumulowane V′, warm start) | bez zmian |

Trudność przyrostu = 1 − BA(ΔV′) modelu T′ ustalonego na czas fazy V′. Zapisywane są: BA(ΔV′), odsetek błędów,
liczba próbek w każdej klasie, zmiana trudności względem poprzedniego przyrostu, koszt oceny (wiersze).
Limit |V′| = min(⌊`max_vprime_fraction`·|V|⌋, `max_vprime_rows`). Pełne V i test są wyłącznie obserwowane.

Parametry domyślne (sekcja `phase_switch`; `fitness.tprime_metric` / `fitness.vprime_metric` domyślnie BA):

| parametr | wartość domyślna |
|---|---|
| vprime_increment_per_class | 10 |
| max_vprime_fraction | 0.25 |
| max_vprime_rows | null |
| vprime_difficulty_min_delta | 0.01 |
| vprime_difficulty_patience | 2 |
| vprime_difficulty_window | 2 |
| vprime_min_difficulty | 0.0 |
| max_vprime_expansions_per_phase | 10 |
| max_no_improvement_k_increases | 3 |
| fitness_min_delta | 0.0005 |
| allow_one_vprime_retry_after_t_stall | true |
| k_growth_requires_sv_signal | true |
| warm_start_tprime | true |
| ga_stop_at_perfect | true |
| representative | "best_on_final_vprime" |
| max_cycles | 300 |

## 3. Testy

`tests/test_phase_switch.py`: 30 funkcji. Pokryte: każda reguła decyzyjna (w tym tabela kombinacji
„BA = 1 nigdy nie zwiększa K”); skryptowany kontroler (T′ → V′ → T′, próba po zastoju z sukcesem i bez,
najwyżej jedna próba na poziom K); 50 losowych przebiegów, które muszą się zakończyć; reguła trudności (stały wzrost,
koniec wzrostu, zerowa trudność, okno); planowanie przyrostu (bardzo małe V′, limit V′, brak kandydatów); integracja
z GA/SVM dla 3 strategii doboru (niezmienniki każdego kroku, limit |V′| dla kilku wartości ułamka i `max_vprime_rows`,
wyczerpanie kandydatów, brak wpływu pełnego V i testu na przebieg, powtarzalność, wspólny start ewolucji i losowania);
replay harmonogramu; zgodność wsteczna GA; CLI.

## 4. Kontrola niezmienników na wszystkich zapisanych krokach

| sprawdzenie | wynik |
|---|---|
| kroki V′ bez poprzedzającej decyzji przełączenia | 0 |
| kroki, w których jednocześnie wzrosło K i rozszerzono V′ | 0 |
| zmiana K w kroku V′ | 0 |
| BA(V′) ≥ 1 i wzrost K w tym samym kroku (warianty z własnym automatem) | 0 |
| j.w. w `rand_replay` (K narzucone harmonogramem evo; losowe V′ bywa opanowane tam, gdzie ewolucyjne nie) | 35 (informacyjnie) |
| |V′| powyżej limitu | 0 |
| więcej niż jedna próba V′ po zastoju przy tym samym K | 0 |
| przebiegi zakończone bezpiecznikiem `stop_max_cycles` | 0 |
| pary evo_switch / rand_replay z identycznym końcowym |V′| | 39 / 39 |

## 5. Wyniki smoke

### 5.1. Moons — kilka wartości `max_vprime_fraction`, bez i z bramką ρ_SV (3 seedy)

| seria | zbiór | wariant | n | |V'| (śr.) | |V'| min–max | |V| | |V'|/|V| | limit |V'| | K końc. | kroki T'/V' | fazy V' (opanowanie/zastój) | BA test | BA V' (model) | trudność ΔV' | czas [s] | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke_moons_f005 | moons | evo_switch | 3 | 240 | 240–240 | 4800 | 0,050 | 240 | 81 | 18,0/11,0 | 1,0/2,0 | 0,9654 ± 0,0050 | 0,9045 | 0,244 | 10 | tprime_no_progress:3 |
| smoke_moons_f005 | moons | rand_replay | 3 | 240 | 240–240 | 4800 | 0,050 | 240 | 81 | 18,0/11,0 | 1,0/2,0 | 0,9635 ± 0,0021 | 0,9808 | 0,040 | 4 | replay_end:3 |
| smoke_moons_f005 | moons | rand_switch | 3 | 200 | 120–240 | 4800 | 0,042 | 240 | 45 | 9,7/9,0 | 1,0/1,7 | 0,9635 ± 0,0039 | 0,9805 | 0,038 | 3 | no_progress_after_vprime_retry:3 |
| smoke_moons_f010 | moons | evo_switch | 3 | 480 | 480–480 | 4800 | 0,100 | 480 | 113 | 32,7/23,0 | 1,0/4,3 | 0,9600 ± 0,0025 | 0,8980 | 0,234 | 23 | tprime_no_progress:3 |
| smoke_moons_f010 | moons | rand_replay | 3 | 480 | 480–480 | 4800 | 0,100 | 480 | 113 | 32,7/23,0 | 1,0/4,3 | 0,9667 ± 0,0027 | 0,9772 | 0,037 | 8 | replay_end:3 |
| smoke_moons_f010 | moons | rand_switch | 3 | 253 | 120–360 | 4800 | 0,053 | 480 | 55 | 11,3/11,7 | 1,0/2,0 | 0,9653 ± 0,0030 | 0,9826 | 0,037 | 4 | no_progress_after_vprime_retry:3 |
| smoke_moons_f025 | moons | evo_switch | 3 | 833 | 720–940 | 4800 | 0,174 | 1200 | 484 | 51,7/40,7 | 1,0/8,3 | 0,9683 ± 0,0017 | 0,8940 | 0,216 | 43 | no_progress_after_vprime_retry:3 |
| smoke_moons_f025 | moons | rand_replay | 3 | 833 | 720–940 | 4800 | 0,174 | 1200 | 484 | 51,7/40,7 | 1,0/8,3 | 0,9676 ± 0,0017 | 0,9773 | 0,032 | 20 | replay_end:3 |
| smoke_moons_f025 | moons | rand_switch | 3 | 253 | 120–360 | 4800 | 0,053 | 1200 | 55 | 11,3/11,7 | 1,0/2,0 | 0,9653 ± 0,0030 | 0,9826 | 0,037 | 4 | no_progress_after_vprime_retry:3 |
| smoke_moons_f010_svgate | moons | evo_switch | 3 | 407 | 280–480 | 4800 | 0,085 | 480 | 90 | 22,7/19,3 | 1,0/3,7 | 0,9660 ± 0,0024 | 0,8965 | 0,235 | 13 | no_k_growth_signal:2, no_progress_after_vprime_retry:1 |
| smoke_moons_f010_svgate | moons | rand_replay | 3 | 407 | 280–480 | 4800 | 0,085 | 480 | 90 | 22,7/19,3 | 1,0/3,7 | 0,9650 ± 0,0048 | 0,9805 | 0,035 | 5 | replay_end:3 |
| smoke_moons_f010_svgate | moons | rand_switch | 3 | 253 | 120–360 | 4800 | 0,053 | 480 | 55 | 11,3/11,7 | 1,0/2,0 | 0,9653 ± 0,0030 | 0,9826 | 0,037 | 3 | no_progress_after_vprime_retry:3 |
| smoke_moons_f025_svgate | moons | evo_switch | 3 | 427 | 280–540 | 4800 | 0,089 | 1200 | 90 | 22,7/20,3 | 1,0/3,7 | 0,9669 ± 0,0024 | 0,8952 | 0,234 | 13 | no_k_growth_signal:3 |
| smoke_moons_f025_svgate | moons | rand_replay | 3 | 427 | 280–540 | 4800 | 0,089 | 1200 | 90 | 22,7/20,3 | 1,0/3,7 | 0,9650 ± 0,0048 | 0,9814 | 0,033 | 5 | replay_end:3 |
| smoke_moons_f025_svgate | moons | rand_switch | 3 | 253 | 120–360 | 4800 | 0,053 | 1200 | 55 | 11,3/11,7 | 1,0/2,0 | 0,9653 ± 0,0030 | 0,9826 | 0,037 | 3 | no_progress_after_vprime_retry:3 |

Bez bramki ρ_SV (serie `smoke_moons_f0xx`) K rośnie poniżej TSV coraz szybciej (AUDYT S3), a |V′| zatrzymuje się dopiero na limicie.
Z bramką (`_svgate`, wartość domyślna) |V′| ustala się na podobnym poziomie przy limicie 0,10 i 0,25, czyli rozmiar wynika z automatu, a nie z limitu.

### 5.2. Szachownica (2 seedy)

| seria | zbiór | wariant | n | |V'| (śr.) | |V'| min–max | |V| | |V'|/|V| | limit |V'| | K końc. | kroki T'/V' | fazy V' (opanowanie/zastój) | BA test | BA V' (model) | trudność ΔV' | czas [s] | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke_chess_r1000 | chessboard | evo_switch | 2 | 790 | 660–920 | 27853 | 0,028 | 1000 | 4728 | 61,0/38,5 | 1,5/8,5 | 0,9818 ± 0,0096 | 0,9597 | 0,287 | 608 | no_progress_after_vprime_retry:2 |
| smoke_chess_r1000 | chessboard | rand_switch | 2 | 420 | 400–440 | 27853 | 0,015 | 1000 | 824 | 24,5/20,0 | 3,5/2,0 | 0,9653 ± 0,0036 | 0,9945 | 0,127 | 90 | no_progress_after_vprime_retry:2 |
| smoke_chess_r1000_svgate | chessboard | evo_switch | 2 | 470 | 420–520 | 27853 | 0,017 | 1000 | 860 | 30,0/22,5 | 1,5/4,0 | 0,9706 ± 0,0025 | 0,9506 | 0,341 | 145 | no_progress_after_vprime_retry:1, no_k_growth_signal:1 |
| smoke_chess_r1000_svgate | chessboard | rand_replay | 2 | 470 | 420–520 | 27853 | 0,017 | 1000 | 860 | 30,0/22,5 | 3,0/2,5 | 0,9710 ± 0,0046 | 0,9880 | 0,097 | 90 | replay_end:2 |
| smoke_chess_r1000_svgate | chessboard | rand_switch | 2 | 420 | 400–440 | 27853 | 0,015 | 1000 | 824 | 24,5/20,0 | 3,5/2,0 | 0,9653 ± 0,0036 | 0,9945 | 0,127 | 93 | no_progress_after_vprime_retry:2 |
| smoke_chess_f025_svgate | chessboard | evo_switch | 2 | 470 | 420–520 | 27853 | 0,017 | 6963 | 860 | 30,0/22,5 | 1,5/4,0 | 0,9706 ± 0,0025 | 0,9506 | 0,341 | 149 | no_k_growth_signal:1, no_progress_after_vprime_retry:1 |
| smoke_chess_f025_svgate | chessboard | rand_replay | 2 | 470 | 420–520 | 27853 | 0,017 | 6963 | 860 | 30,0/22,5 | 3,0/2,5 | 0,9710 ± 0,0046 | 0,9880 | 0,097 | 76 | replay_end:2 |
| smoke_chess_f025_svgate | chessboard | rand_switch | 2 | 420 | 400–440 | 27853 | 0,015 | 6963 | 824 | 24,5/20,0 | 3,5/2,0 | 0,9653 ± 0,0036 | 0,9945 | 0,127 | 102 | no_progress_after_vprime_retry:2 |

### 5.3. Krótka seria Moons (10 seedów, wartości domyślne)

| seria | zbiór | wariant | n | |V'| (śr.) | |V'| min–max | |V| | |V'|/|V| | limit |V'| | K końc. | kroki T'/V' | fazy V' (opanowanie/zastój) | BA test | BA V' (model) | trudność ΔV' | czas [s] | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| short_moons | moons | evo_switch | 10 | 404 | 280–540 | 4800 | 0,084 | 1200 | 92 | 22,9/19,2 | 1,0/4,0 | 0,9672 ± 0,0026 | 0,8953 | 0,244 | 18 | no_k_growth_signal:8, no_progress_after_vprime_retry:2 |
| short_moons | moons | rand_replay | 10 | 404 | 280–540 | 4800 | 0,084 | 1200 | 92 | 22,9/19,2 | 1,1/3,9 | 0,9652 ± 0,0044 | 0,9770 | 0,046 | 8 | replay_end:10 |
| short_moons | moons | rand_switch | 10 | 286 | 120–500 | 4800 | 0,060 | 1200 | 58 | 11,5/13,3 | 1,7/1,7 | 0,9645 ± 0,0035 | 0,9790 | 0,046 | 4 | no_progress_after_vprime_retry:10 |

Porównania sparowane (te same seedy; `rand_replay` = identyczne fazy, K i |ΔV′| jak evo_switch):

| a | b | metric | n | mean_a | mean_b | mean_diff | sd_diff | a_larger | b_larger |
|---|---|---|---|---|---|---|---|---|---|
| evo_switch | rand_replay | test_ba | 10 | 0,9672 | 0,9652 | 0,0019 | 0,0062 | 6 | 4 |
| evo_switch | rand_replay | fullv_ba | 10 | 0,9656 | 0,9650 | 0,0005 | 0,0045 | 6 | 4 |
| evo_switch | rand_replay | vprime_rows | 10 | 404,0000 | 404,0000 | 0,0000 | 0,0000 | 0 | 0 |
| evo_switch | rand_replay | steps | 10 | 42,1000 | 42,1000 | 0,0000 | 0,0000 | 0 | 0 |
| evo_switch | rand_replay | final_k | 10 | 92,5000 | 92,5000 | 0,0000 | 0,0000 | 0 | 0 |
| evo_switch | rand_replay | mean_delta_difficulty | 10 | 0,2440 | 0,0460 | 0,1979 | 0,0146 | 10 | 0 |
| evo_switch | rand_switch | test_ba | 10 | 0,9672 | 0,9645 | 0,0027 | 0,0038 | 8 | 1 |
| evo_switch | rand_switch | fullv_ba | 10 | 0,9656 | 0,9637 | 0,0019 | 0,0030 | 7 | 3 |
| evo_switch | rand_switch | vprime_rows | 10 | 404,0000 | 286,0000 | 118,0000 | 153,3188 | 7 | 2 |
| evo_switch | rand_switch | steps | 10 | 42,1000 | 24,8000 | 17,3000 | 12,2932 | 9 | 0 |
| evo_switch | rand_switch | final_k | 10 | 92,5000 | 57,9000 | 34,6000 | 13,5499 | 10 | 0 |
| evo_switch | rand_switch | mean_delta_difficulty | 10 | 0,2440 | 0,0455 | 0,1984 | 0,0163 | 10 | 0 |

### 5.4. |V′|/|V| na wszystkich zbiorach (wartości domyślne) i porównanie z poprzednim algorytmem

| zbiór | seria | n | |V| | limit |V'| | |V'| nowy (śr.) | |V'|/|V| nowy | |V'| stary evo_mem (mediana) | |V'|/|V| stary | BA test nowy | BA test stary evo_mem |
|---|---|---|---|---|---|---|---|---|---|---|
| moons | short_moons | 10 | 4800 | 1200 | 404 | 0,084 | 264 | 0,055 | 0,9672 | 0,9648 |
| circles | smoke_all | 2 | 4800 | 1200 | 400 | 0,083 | 1211 | 0,252 | 0,9794 | 0,9772 |
| classification2d | smoke_all | 2 | 4800 | 1200 | 410 | 0,085 | 403 | 0,084 | 0,9421 | 0,9417 |
| digits | smoke_all | 2 | 719 | 179 | 170 | 0,236 | 705 | 0,981 | 0,9538 | 0,9643 |
| magic | smoke_all | 2 | 3200 | 800 | 240 | 0,075 | 2253 | 0,704 | 0,8375 | 0,8382 |
| spiral | smoke_all | 2 | 4800 | 1200 | 510 | 0,106 | 465 | 0,097 | 0,9152 | 0,9123 |
| chessboard | smoke_chess_f025_svgate | 2 | 27853 | 6963 | 470 | 0,017 | 3606 | 0,129 | 0,9706 | 0,9732 |
| chessboard | smoke_chess_r1000_svgate | 2 | 27853 | 1000 | 470 | 0,017 | 3606 | 0,129 | 0,9706 | 0,9732 |

### 5.5. Obserwacje wymagające decyzji

* **Szachownica: jakość ogranicza polityka K, a nie V′.** Pełny SVM (cały zbiór treningowy) ma BA test 0,9885. Z bramką ρ_SV K kończy się na ≈860 i BA test = 0,9706 (stary algorytm też kończył przy ρ_SV < TSV: K ≈ 837, BA 0,9732). Bez bramki K rośnie do 8450, 1005 i BA test = 0,9886, 0,9751 (seedy 43, 42), czyli przy dużym K zbliża się do pełnego SVM, kosztem czasu (śr. 608 s vs 145 s). Wybór TSV / bramki to decyzja o kompromisie rozmiar T′ ↔ jakość, niezależna od reguł V′.
* **Wybór modelu reprezentatywnego.** Domyślnie wybierane jest najlepsze T′ na końcowym V′ (`best_on_final_vprime`, tylko V′). W 22 z 72 przebiegów jest to inne T′ niż ostatni krok. Średnia różnica BA test (wybrany − ostatni) = -0,0021; najgorzej Digits (-0,0112): przy ~17 próbkach V′ na klasę wybór po fitnessie V′ jest zaszumiony. Alternatywa `representative: last_tprime_step` jest zaimplementowana.
* **Digits (10 klas, |V| = 719):** limit 0,25·|V| = 179 jest osiągany (|V′| = 170), a K dochodzi do limitu klasy (`stop_k_limit`). Przy wielu klasach i małym V ułamek 0,25 może być zbyt ciasny; w pełnym eksperymencie warto sprawdzić 0,5 dla Digits.

## 6. Warunki z punktu 4.8 przed pełnym eksperymentem

| warunek | stan |
|---|---|
| fazy przełączają się zgodnie z regułą | spełniony (sekcja 4) |
| V′ nie rośnie automatycznie do ≈3500 | spełniony: maks. |V′| na Szachownicy = 920 (smoke_chess_r1000: |V'| = 790, smoke_chess_r1000_svgate: |V'| = 470, smoke_chess_f025_svgate: |V'| = 470) |
| |V′|/|V| raportowane | spełniony (`vprime_rows`, `full_validation_rows`, `vprime_fraction` w summary.json, cycles.csv, per_run.csv) |
| warunek stopu nie zapętla algorytmu | spełniony: 0 stopów bezpiecznikiem |
| wariant losowy i ewolucyjny porównywalne | spełniony dla `rand_replay` (identyczny harmonogram); `rand_switch` ma inny rozmiar V′ |

## 7. Odpowiedzi

**1. Dlaczego wcześniej V′ rosło do ≈3500?**
W starym trybie V′ było rozszerzane przez logikę ρ_SV, a nie przez sukces T′. Po każdym wzroście K flaga „V′ utwardzone
dla tego K” się zerowała, co wymuszało rozszerzenie. Dodatkowo przy stabilnym ρ_SV rozszerzano V′ zamiast K (do Ts razy).
Przyrost wynosił 2·(K_next − K), czyli rósł razem z K.
Szachownica (podział bazowy, evo_mem + rand_growth): mediana 45 rozszerzeń i 17
wzrostów K na przebieg, średnio 79,2 próbek na rozszerzenie.
Tylko 5 z 1775 rozszerzeń nastąpiło po BA(V′)=1,0.
Końcowe |V′| = mediana 3606. Przebieg kończył się, gdy ρ_SV < TSV albo K osiągało limit; rozmiar V′ nie miał żadnego ograniczenia.

**2. Jak nowy algorytm ogranicza ten wzrost?**
(a) V′ rośnie tylko w fazie `optimize_V_prime`, do której wchodzi się wyłącznie po opanowaniu V′ (BA = 1) albo jednorazowo
na poziom K po zastoju T′. (b) Przyrost ma stały, jawny rozmiar (`vprime_increment_per_class`, domyślnie 10/klasę),
niezależny od K. (c) Faza V′ kończy się, gdy trudność przyrostów przestaje rosnąć. (d) Twardy limit
min(`max_vprime_fraction`·|V|, `max_vprime_rows`). (e) Brak poprawy T′ po próbie V′ kończy cały algorytm.
Szachownica: smoke_chess_r1000: |V'| = 790, smoke_chess_r1000_svgate: |V'| = 470, smoke_chess_f025_svgate: |V'| = 470. Moons: |V′| = 404
(|V′|/|V| = 0,084).

**3. Kiedy dokładnie przechodzimy z T′ do V′?** Po kroku T′, gdy (i) najlepsze T′ ma BA(V′) ≥ `perfect_threshold` (1,0), albo
(ii) wystąpił zastój T′, czyli 3 kolejne wzrosty K bez poprawy fitnessu
o ≥ 0.0005, ρ_SV < TSV albo K na limicie, i przy tym K nie było jeszcze próby V′.
W obu przypadkach V′ musi dać się rozszerzyć. W smoke (b) jest ścieżką dominującą: przy doborze ewolucyjnym BA = 1
występuje praktycznie tylko na startowym V′ (AUDYT S4).

**4. Kiedy dokładnie wracamy z V′ do T′?** Po kroku V′, gdy trudność przyrostów (1 − BA(ΔV′), średnia z okna
2) przez 2 kolejne przyrosty nie wzrosła
o ≥ 0.01 (przyrost bez błędów liczy się jako brak wzrostu); albo gdy V′ osiągnęło limit
lub zabrakło kandydatów; albo po 10 przyrostach w fazie. T′ jest wtedy optymalizowane
na pełnym, skumulowanym V′ przy niezmienionym K.

**5. Kiedy zwiększamy K?** Tylko po kroku T′, w którym BA(V′) < 1,0, nie ma zastoju i ρ_SV ≥ TSV (istniejąca polityka DAGA;
wielkość kroku z dotychczasowego wzoru). Najwyżej raz na krok. Nigdy w kroku V′ i nigdy przy BA(V′) = 1,0.

**6. Kiedy kończymy cały algorytm?** `stop_perfect_on_max_vprime`: T′ opanowało V′, a V′ nie da się już rozszerzyć.
`stop_no_progress_after_vprime_retry`: po próbie V′ T′ nie poprawiło się na tym samym V′.
`stop_tprime_no_progress` / `stop_no_k_growth_signal` / `stop_k_limit`: zastój T′, a próba V′ przy tym K była już wykonana lub V′ jest na limicie.
`stop_max_cycles`: bezpiecznik (0 wystąpień). Samo nieopanowanie V′ nie kończy algorytmu.

**7. Jak porównujemy ewolucję i losowanie przy tym samym rozmiarze V′?** Wariant `rand_replay` odtwarza z przebiegu
evo_switch z tym samym seedem sekwencję faz, K w każdym kroku T′ i |ΔV′| w każdym kroku V′. Różni się wyłącznie tym,
które próbki trafiają do ΔV′ (losowo zamiast Val-GA). Start (V′ startowe, pierwszy krok T′) jest wspólny.
Moons, 10 seedów: BA test evo − rand_replay = 0,0019
(SD różnic 0,0062; evo lepsze w 6/10),
przy trudności przyrostów 0,244 vs
0,046. Pomocniczo `rand_switch` (losowanie we własnym automacie)
daje mniejsze V′ (286 vs 404).

**8. |V′|/|V| dla każdego zbioru:** tabela 5.4. Zbiory inne niż Moons i Szachownica mają tylko 2 seedy (smoke).

## 8. Proponowany następny krok (do decyzji)

1. Pełny eksperyment w układzie evo_switch / rand_replay (+ rand_switch pomocniczo) z wartościami domyślnymi z sekcji 2,
   przy jawnym `max_vprime_rows` tylko tam, gdzie limit ułamkowy jest zbyt luźny.
2. Do rozstrzygnięcia z promotorem: czy próg „opanowania V′” ma pozostać 1,0 (AUDYT S4), czy zastój T′ ma być pełnoprawną
   ścieżką do fazy V′ (obecnie dominującą).
3. Porównanie metryk (BA / F1 / ROC AUC) dopiero po ustaleniu automatu; kod jest przygotowany (`fitness.tprime_metric`,
   `fitness.vprime_metric`), ROC AUC dla doboru V′ wymaga jeszcze implementacji w Val-GA.
