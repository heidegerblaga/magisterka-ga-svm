# Audyt fazowego DAGASVM przed przebudową (followup_phase_switch)

Data: 2026-10-14. Stan kodu: `b38f226` + niezatwierdzone zmiany followup_stop_rule
(`src/ga/phase_engine.py`, `src/ga/vprime.py`, `scripts/run_experiment.py`, `README.md`).
Testy bazowe: `tests/test_stop_rule.py`, `tests/test_phase_engine.py`, `tests/test_vprime.py`, 48/48 zaliczonych (`.venv`).
Liczby poniżej pochodzą z `reports/followup_stop_rule/tables/per_run.csv` i `per_cycle.csv`
(podział 2026, warianty rosnące evo_mem / evo_nomem / heur_mem / rand_growth, reguła `extended`).

## 1. Jak algorytm działa dziś (`src/ga/phase_engine.py::run_phase_experiment`)

Jeden **cykl** = dokładnie jedno uruchomienie GA dla T′ przy stałym K (`run_ga_k_per_class_with_trace`)
+ jedna decyzja `decide_cycle(...)` ∈ {`expand_validation_only`, `increase_k`, `stop_*`}
+ co najwyżej jeden krok doboru przyrostu V′ (`expand_vprime`).

Decyzja zależy od flag: `perfect_on_current_val`, `has_growth_signal` (ρ_SV ≥ TSV),
`validation_hardened_for_current_k`, `sv_ratio_stable_since_hardening`, licznika cierpliwości Ts
oraz (od followup_stop_rule) bramki `perfect_stop_gate` (min. rozmiar V′ + obowiązkowe „ponowne sprawdzenia”).

## 2. Ustalenia

### A1. Nie ma jawnych faz. „Faza V′” to jeden krok doboru, a nie optymalizacja
* W kodzie nie istnieje stan „faza T′ / faza V′”. Faza wynika pośrednio z kombinacji flag w `decide_cycle`.
* Krok V′ to pojedyncze wywołanie `expand_vprime` (heurystyka, Val-GA albo losowanie). Kod nie sprawdza,
  czy nowe V′ faktycznie „łamie” T′. Trudność przyrostu (`selected_error_rate`) jest tylko logowana,
  nie steruje algorytmem.
* T′ ocenia się na nowym V′ dopiero w kolejnym cyklu, w nowym, pełnym przebiegu GA.

### A2. Przełączenie T′ → V′ nie jest wywoływane sukcesem T′ na V′
* 13 147 rozszerzeń V′ na 21 061 cykli. Tylko 235 cykli (1,1%) ma BA(V′) ≥ 1,0.
  Niemal wszystkie rozszerzenia wynikają więc z logiki ρ_SV / „utwardzenia dla bieżącego K”,
  a nie z tego, że T′ opanowało V′.
* W typowym przebiegu (Moons, evo_mem, seed 42) cykle układają się w sztywny rytm:
  rozszerz V′ → zwiększ K → rozszerz V′ → … BA(V′) spada przy tym monotonicznie z 0,94 do 0,88.
  T′ nigdy nie „dogania” V′.
* Średnie BA(V′) w ostatnim cyklu: evo_mem 0,895, heur_mem 0,816, evo_nomem 0,937, rand_growth 0,947.
  Algorytm kończy się z T′, które nie klasyfikuje bezbłędnie aktualnego V′.

### A3. Warunek stopu w praktyce = ρ_SV, nie BA(V′)
Kategorie stopu (podział bazowy, 706 przebiegów): `no_k_growth_signal` 499,
`k_limit` 204, `no_candidates` 2, `perfect_ba` 1.
Przebieg kończy się prawie zawsze, gdy ρ_SV < TSV albo K dochodzi do limitu klasy.
Stop „BA(V′) = 1,0” praktycznie nie występuje.

### A4. Reguła „minimalnego V′” obsługuje głównie zdegenerowany start
* 72% cykli z BA(V′) = 1,0 to **cykl 1**. Mediana |V′| w takich cyklach = **20** (= 2 klasy × `k_start` = 10).
* Bramka `min_size` + `recheck` zadziałała w 220 z 235 takich cykli i przeskoczyła V′ jednym krokiem
  (mediana +80 próbek, np. 20 → 100). Zwykłe rozszerzenie dodaje medianowo 28 próbek.
* Jest to dokładnie mechanizm zakwestionowany przez promotora: BA(V′) = 1,0 „naprawia się”
  sztucznym dobiciem V′ do minimum, zamiast potraktować je jako sygnał przejścia do fazy V′.
  Stara wersja (bez bramki) kończyła zaś w tej sytuacji cały algorytm po jednym rozszerzeniu
  (57/700 przebiegów, w raporcie 54 określono jako przedwczesne).

### A5. Rozmiar V′ nie jest kontrolowany. Jest sprzężony z K
* Przyrost na klasę = `next_k − K` (wzór DAGA na ρ_SV), minimum 1. Rozmiar V′ rośnie więc tak jak K,
  bez własnego limitu ani budżetu.
* Końcowe |V′| (mediany): Szachownica ≈ 3600, MAGIC 8000 ≈ 2250 (z 3200 w V), Circles ≈ 1200, Moons ≈ 260.
* `memory=false` odtwarza ten sam harmonogram rozmiaru.
* `rand_fixed` bierze rozmiar z końcowego |V′| evo_mem (`fixed_size_source_variant`).
  Wariant bazowy „wie” więc, ile urosło V′ w wariancie porównywanym.

### A6. T′ startuje od zera w każdym cyklu
`run_ga_k_per_class_with_trace` inicjalizuje populację losowo (`init_k_per_class`, seed+i).
Najlepsze T′ z poprzedniego cyklu nie trafia do populacji, nawet gdy K się nie zmieniło
(po samym rozszerzeniu V′). „Ponowna optymalizacja T′” jest więc w praktyce nową optymalizacją.
Wyjaśnia to część skoków BA(V′) między cyklami.

### A7. GA dla T′ nie zatrzymuje się po osiągnięciu BA(V′) = 1,0
Wczesne zatrzymanie to tylko cierpliwość `Ts` + `phase_min_delta`. Po osiągnięciu fitness 1,0
GA wykonuje jeszcze Ts generacji bez możliwości poprawy. Nie jest to błąd, tylko zmarnowany budżet.
Zdarzenie „T′ opanowało V′” nie jest też jawnie raportowane z wnętrza GA.

### A8. Model reprezentatywny = ostatni cykl
`selection_rule = "last_completed_cycle"`. Przy obecnym stopie (ρ_SV) jest to T′ dopasowane
do najtrudniejszego V′, na którym BA(V′) < 1.

### A9. Elementy poprawne, do zachowania
* Rozdział ról: V′ steruje, pełne V jest pulą kandydatów + obserwacją, test jest tylko obserwacją
  (`FullValidationObserver`, testy `test_full_validation_metric_never_controls_*`).
* `VPrimeSet` (brak duplikatów, inwarianty `memory`) i `expand_vprime` (strategie, budżet puli `pool_budget_factor`).
* Inwariant „BA(V′) = 1,0 nigdy nie zwiększa K” i „≤ 1 wzrost K na cykl”.
* Niezatwierdzone zmiany w `vprime.py` (budżet puli, liczniki kosztu) i nowe kolumny w `run_experiment.py`
  są kompletne i potrzebne do odtworzenia raportu followup_stop_rule. **Nie usuwać.**
  Bramka `min_vprime_per_class` / `perfect_recheck_*` zostaje jako tryb historyczny (domyślnie 0/0 = reguła sprzed followupu).

## 3. Wnioski dla przebudowy

1. Potrzebny jest jawny automat fazowy (`optimize_T_prime` / `optimize_V_prime` / `stop`) jako **nowy tryb**
   (np. `algorithm_mode: dagasvm_phase_switch`), obok obecnego `dagasvm_phase`.
   Dzięki temu stare konfiguracje i raport pozostają odtwarzalne bit w bit.
2. Przejście T′ → V′ ma być wyzwalane przez BA(V′) ≥ 1,0 (oraz prawdopodobnie przez stagnację T′;
   **do potwierdzenia w dalszej części specyfikacji**), a nie przez flagi „utwardzenia dla K”.
3. Faza V′ powinna być rzeczywistą optymalizacją. Faza ta trwa, dopóki nowe V′ nie przestanie być
   bezbłędnie klasyfikowane przez bieżące T′ (albo do wyczerpania budżetu / kandydatów).
4. Rozmiar V′ musi mieć własną, jawną kontrolę (przyrost, limit, budżet), oddzieloną od K.
5. Ponowna optymalizacja T′ powinna móc startować z poprzedniego najlepszego T′ (warm start).
6. Stop powinien wynikać z jawnych warunków automatu, a nie z wyczerpania ρ_SV jako efektu ubocznego.

## 4. Uzupełnienie po implementacji: sprzeczności specyfikacji i przyjęte rozwiązania

Implementacja: `src/ga/phase_switch.py` (tryb `dagasvm_phase_switch`). Stary tryb `dagasvm_phase` jest bez zmian.
Kontrola: przebieg followup_stop_rule (Moons, evo_mem, seed 42) uruchomiony nowym kodem daje identyczny `cycles.csv`
(poza kolumnami czasu).

### S1. Porównanie „po próbie V′ ≥ najlepszy fitness sprzed próby” porównuje BA na dwóch różnych zbiorach
Fitness po próbie liczony jest na V′ powiększonym o trudny przyrost, a fitness sprzed próby na mniejszym, łatwiejszym V′.
Dosłowna reguła kończyłaby więc algorytm niemal zawsze, ale nie z powodu braku postępu T′.
**Rozwiązanie:** punktem odniesienia jest najlepsze T′ sprzed próby **ocenione na tym samym, rozszerzonym V′**
(`retry_reference_same_vprime`). Porównanie dosłowne jest zapisywane jako diagnostyka (`retry_progress_raw`).
Różnica jest duża w praktyce. We wszystkich seriach smoke/short (warianty z własnym automatem) wykonano 193 sprawdzenia po próbie V′: postęp na tym samym V′ w 153, a w porównaniu dosłownym tylko w 39 (short_moons: 45/57 vs 12/57).

### S2. „Co najmniej” przy warm starcie jest spełnione zawsze
Ponowna optymalizacja T′ startuje z najlepszym poprzednim T′ w populacji (warm start), więc wynik nigdy nie jest gorszy.
Warunek „≥” nie mógłby więc nigdy zakończyć algorytmu.
**Rozwiązanie:** postęp = poprawa o co najmniej `fitness_min_delta` (domyślnie 0,0005) albo osiągnięcie progu BA(V′)=1.

### S3. Wzór na K działa odwrotnie poniżej TSV
`ΔK = 1 + |ρ_SV − TSV| / (1 − TSV)` zawiera wartość bezwzględną. W starym algorytmie wzór był używany tylko przy ρ_SV ≥ TSV
(przy ρ_SV < TSV następował stop). Bez tej bramki: im niższe ρ_SV, tym szybciej rośnie K
(Moons: 418 → 1117 w 4 krokach; Szachownica: K ≈ 3800).
**Rozwiązanie (domyślne):** `k_growth_requires_sv_signal: true`, czyli istniejąca polityka K (wzrost tylko przy ρ_SV ≥ TSV).
ρ_SV < TSV **nie kończy** jednak algorytmu od razu: jest zastojem T′, który daje jedną próbę V′ na poziom K,
i dopiero brak postępu po niej kończy algorytm (`stop_no_k_growth_signal`).
Wariant bez bramki pozostaje dostępny (`false`); serie smoke_*_f0xx i smoke_chess_r1000 uruchomiono właśnie z nim.

### S4. BA(V′)=1,0 jest osiągalne praktycznie tylko na startowym V′
Przy doborze ewolucyjnym ΔV′ trafia w obszar nakładania się klas lub szumu etykiet, gdzie BA = 1 jest nieosiągalne.
Przejście „T′ opanowało V′” następuje więc raz (na V′ startowym, 2·K_start próbek), a wszystkie kolejne fazy V′
wynikają z reguły zastoju (4.4). Specyfikacja to dopuszcza, ale główną ścieżką automatu jest w praktyce zastój → próba V′.
Jeżeli promotor chce, aby główną ścieżką było „opanowanie V′”, próg `perfect_threshold` musiałby być < 1
(np. 1 − ε albo próg względny). Parametr istnieje; wartość < 1 nie była testowana.

### S5. Reguła „trudność przestaje rosnąć” przy ustalonym modelu
W fazie V′ model T′ jest ustalony, a dobór adwersarialny zbiera najpierw najtrudniejsze próbki. Trudność kolejnych przyrostów
powinna więc maleć, a faza powinna kończyć się po `patience + 1` przyrostach. Tak się dzieje przy doborze
heurystycznym. Val-GA jest stochastyczny, więc trudność jego przyrostów fluktuuje (0,13–0,50 przy 20 próbkach), a faza trwa dłużej.
Średnia z okna (`vprime_difficulty_window`, domyślnie 2) wygładza te fluktuacje tylko częściowo.

### S6. Jeden ułamek `max_vprime_fraction` nie daje porównywalnego |V′| między zbiorami
|V| wynosi od 719 (Digits) do 27 853 (Szachownica). 0,25 to 6963 próbki dla Szachownicy (więcej niż stare ≈ 3600)
i 180 dla Digits. Dlatego obok ułamka jest bezwzględny limit `max_vprime_rows` (efektywny limit = minimum obu).
Przy bramce ρ_SV (S3) limit 0,25 w ogóle nie jest osiągany: Moons kończy przy |V′|/|V| ≈ 0,08.

### S7. Wariant losowy „z takim samym harmonogramem”
Losowy dobór ΔV′ we własnym automacie (`rand_switch`) daje łatwe przyrosty (trudność ≈ 0,05), szybki koniec fazy V′
i wcześniejszy stop, więc mniejsze |V′| (Moons: 286 vs 404). Porównanie z ewolucją miesza wtedy skutek doboru ze skutkiem rozmiaru V′.
**Rozwiązanie:** wariant `rand_replay` odtwarza z przebiegu evo_switch z tym samym seedem **dokładnie** sekwencję faz,
K w każdym kroku T′ i |ΔV′| w każdym kroku V′. Różni się tylko wyborem próbek. To jest porównanie główne;
`rand_switch` jest raportowany pomocniczo.

### S8. Ciche zastępowanie metryk (przygotowanie do porównania BA / F1 / ROC AUC)
* `_metric_from_preds` (Val-GA) dla nieznanej metryki (np. `roc_auc`, `f1_macro`) po cichu liczy BA.
* `evaluate_fitted_svm` dla `roc_auc` / `pr_auc` przy `svm.probability=false` po cichu zwraca BA.

Nowy tryb odrzuca takie konfiguracje jawnym błędem. `fitness.tprime_metric` przyjmuje metryki obsługiwane przez
`evaluate_fitted_svm`; `fitness.vprime_metric` (na razie) tylko `balanced_accuracy`, `accuracy`, `f1`.

### S9. `memory=false` nie ma sensu w automacie z przyrostem ΔV′
Przy `memory=false` V′ jest regenerowane i „przyrost” nie istnieje. Tryb wymaga `vprime.memory=true` i `vprime.mode=growing`.
