# followup_phase_switch v2 — podziały dodatkowe 2027–2029

Raport generowany przez `scripts/followup_phase_switch/report_splits.py`. Uzupełnia `RAPORT_PELNE_V2.md` (podział bazowy 2026, 20 seedów).
Zakres nowych przebiegów: podziały 2027, 2028, 2029 × 6 zbiorów zależnych od podziału (circles, classification2d, digits, magic, moons, spiral) × seedy 42–46 ×
3 warianty (`evo_switch`, `rand_replay`, `rand_switch`) = **270 przebiegów**; ustawienia domyślne automatu, parametry SVM (C, γ) osobne dla każdego podziału
(jak w `followup_stop_rule`). Szachownica nie bierze udziału (stały plik `data/raw`, nie zależy od podziału). Analizowany układ zrównoważony: 4 podziały (2026 z serii
`v2_full`, tylko seedy 42–46) × 5 seedów = 20 obserwacji na zbiór i wariant (komplet).
`pytest`: 140 passed (0 błędów). Poprzednie wyniki (v1, smoke, `v2_full`) i stary tryb są nietknięte.

BA na teście i na pełnym V to wyłącznie obserwacje. Trudność przyrostu ΔV′ i jakość modelu są osobnymi właściwościami; raport nie wnioskuje z jednej o drugiej.

## 1. Wyniki per zbiór i podział

| zbiór | podział | n | \|V\| | \|V′\| evo (śr.) | \|V′\|/\|V\| | K końc. evo | BA test evo | BA test rand_replay | BA test rand_switch | evo − replay | pełny SVM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| circles | 2026 | 5 | 4800 | 580 | 0,121 | 320 | 0,9779 | 0,9784 | 0,9786 | -0,0005 | 0,9804 |
| circles | 2027 | 5 | 4800 | 604 | 0,126 | 346 | 0,9812 | 0,9795 | 0,9775 | 0,0017 | 0,9792 |
| circles | 2028 | 5 | 4800 | 584 | 0,122 | 343 | 0,9833 | 0,9840 | 0,9834 | -0,0007 | 0,9842 |
| circles | 2029 | 5 | 4800 | 728 | 0,152 | 366 | 0,9801 | 0,9808 | 0,9778 | -0,0008 | 0,9821 |
| classification2d | 2026 | 5 | 4800 | 540 | 0,112 | 194 | 0,9436 | 0,9439 | 0,9406 | -0,0002 | 0,9479 |
| classification2d | 2027 | 5 | 4800 | 412 | 0,086 | 178 | 0,9381 | 0,9355 | 0,9327 | 0,0027 | 0,9459 |
| classification2d | 2028 | 5 | 4800 | 564 | 0,117 | 172 | 0,9419 | 0,9430 | 0,9371 | -0,0011 | 0,9475 |
| classification2d | 2029 | 5 | 4800 | 552 | 0,115 | 168 | 0,9345 | 0,9323 | 0,9296 | 0,0022 | 0,9388 |
| digits | 2026 | 5 | 719 | 170 | 0,236 | 70 | 0,9584 | 0,9557 | 0,9539 | 0,0027 | 0,9578 |
| digits | 2027 | 5 | 719 | 170 | 0,236 | 70 | 0,9657 | 0,9628 | 0,9595 | 0,0029 | 0,9725 |
| digits | 2028 | 5 | 719 | 170 | 0,236 | 70 | 0,9554 | 0,9465 | 0,9444 | 0,0089 | 0,9582 |
| digits | 2029 | 5 | 719 | 170 | 0,236 | 70 | 0,9654 | 0,9675 | 0,9586 | -0,0022 | 0,9747 |
| magic | 2026 | 5 | 3200 | 396 | 0,124 | 1125 | 0,8386 | 0,8411 | 0,8295 | -0,0024 | 0,8443 |
| magic | 2027 | 5 | 3200 | 292 | 0,091 | 1125 | 0,8361 | 0,8368 | 0,8366 | -0,0007 | 0,8378 |
| magic | 2028 | 5 | 3200 | 368 | 0,115 | 1125 | 0,8247 | 0,8267 | 0,8200 | -0,0020 | 0,8266 |
| magic | 2029 | 5 | 3200 | 404 | 0,126 | 1125 | 0,8293 | 0,8320 | 0,8293 | -0,0027 | 0,8337 |
| moons | 2026 | 5 | 4800 | 380 | 0,079 | 83 | 0,9671 | 0,9674 | 0,9646 | -0,0003 | 0,9700 |
| moons | 2027 | 5 | 4800 | 376 | 0,078 | 91 | 0,9620 | 0,9611 | 0,9610 | 0,0009 | 0,9637 |
| moons | 2028 | 5 | 4800 | 376 | 0,078 | 95 | 0,9603 | 0,9634 | 0,9501 | -0,0031 | 0,9642 |
| moons | 2029 | 5 | 4800 | 368 | 0,077 | 79 | 0,9667 | 0,9630 | 0,9649 | 0,0037 | 0,9708 |
| spiral | 2026 | 5 | 4800 | 452 | 0,094 | 228 | 0,9137 | 0,9144 | 0,9148 | -0,0007 | 0,9317 |
| spiral | 2027 | 5 | 4800 | 460 | 0,096 | 211 | 0,9111 | 0,9184 | 0,9168 | -0,0073 | 0,9275 |
| spiral | 2028 | 5 | 4800 | 412 | 0,086 | 210 | 0,9083 | 0,9096 | 0,9042 | -0,0013 | 0,9200 |
| spiral | 2029 | 5 | 4800 | 464 | 0,097 | 243 | 0,9145 | 0,9111 | 0,9120 | 0,0034 | 0,9329 |

## 2. Ewolucja a losowanie przy tym samym harmonogramie (evo_switch − rand_replay), 4 podziały × 5 seedów

Różnica sparowana po parach (podział, seed), n = 20 na zbiór; 95% CI z bootstrapu. Obserwacje w obrębie jednego podziału nie są w pełni niezależne,
więc przedziały są orientacyjne. Korekta Bonferroniego na 6 zbiorów (osobno dla BA test).

BA test:

| zbiór | n | evo_switch | rand_replay | różnica | 95% CI | CI wyklucza 0 | po korekcie Bonferroniego | evo_switch większe/mniejsze |
|---|---|---|---|---|---|---|---|---|
| circles | 20 | 0,9806 | 0,9807 | -0,0001 | [-0,0008; 0,0007] | nie | nie | 7/10 |
| classification2d | 20 | 0,9395 | 0,9387 | 0,0009 | [-0,0018; 0,0037] | nie | nie | 12/8 |
| digits | 20 | 0,9612 | 0,9581 | 0,0031 | [-0,0006; 0,0074] | nie | nie | 10/9 |
| magic | 20 | 0,8322 | 0,8342 | -0,0020 | [-0,0053; 0,0013] | nie | nie | 9/11 |
| moons | 20 | 0,9640 | 0,9637 | 0,0003 | [-0,0014; 0,0020] | nie | nie | 9/8 |
| spiral | 20 | 0,9119 | 0,9134 | -0,0015 | [-0,0059; 0,0026] | nie | nie | 9/11 |

BA pełnego V (obserwacja):

| zbiór | n | evo_switch | rand_replay | różnica | 95% CI | CI wyklucza 0 | po korekcie Bonferroniego | evo_switch większe/mniejsze |
|---|---|---|---|---|---|---|---|---|
| circles | 20 | 0,9815 | 0,9803 | 0,0012 | [0,0006; 0,0018] | tak | tak | 16/4 |
| classification2d | 20 | 0,9412 | 0,9393 | 0,0019 | [-0,0002; 0,0041] | nie | nie | 12/8 |
| digits | 20 | 0,9697 | 0,9680 | 0,0016 | [-0,0026; 0,0058] | nie | nie | 13/6 |
| magic | 20 | 0,8385 | 0,8416 | -0,0031 | [-0,0058; -0,0008] | tak | tak | 6/14 |
| moons | 20 | 0,9674 | 0,9669 | 0,0005 | [-0,0009; 0,0019] | nie | nie | 12/7 |
| spiral | 20 | 0,9170 | 0,9159 | 0,0011 | [-0,0020; 0,0041] | nie | nie | 12/8 |

CI różnicy BA test wyklucza zero w 0 z 6 zbiorów; po korekcie Bonferroniego w 0.

Pomocniczo, evo_switch − rand_switch (różne harmonogramy, więc różnica miesza efekt doboru z efektem rozmiaru V′):

| zbiór | n | evo_switch | rand_switch | różnica | 95% CI | CI wyklucza 0 | po korekcie Bonferroniego | evo_switch większe/mniejsze |
|---|---|---|---|---|---|---|---|---|
| circles | 20 | 0,9806 | 0,9793 | 0,0013 | [-0,0001; 0,0028] | nie | nie | 11/8 |
| classification2d | 20 | 0,9395 | 0,9350 | 0,0045 | [0,0019; 0,0072] | tak | tak | 14/6 |
| digits | 20 | 0,9612 | 0,9541 | 0,0071 | [0,0010; 0,0139] | tak | nie | 12/7 |
| magic | 20 | 0,8322 | 0,8288 | 0,0034 | [-0,0005; 0,0071] | nie | nie | 15/5 |
| moons | 20 | 0,9640 | 0,9601 | 0,0039 | [0,0003; 0,0081] | tak | nie | 15/5 |
| spiral | 20 | 0,9119 | 0,9119 | -0,0000 | [-0,0040; 0,0039] | nie | nie | 9/10 |

## 3. Wpływ podziału danych na poziom BA

Odch. std. BA test między średnimi podziałów (4 wartości) wobec średniego odch. std. po seedach w obrębie podziału:

| zbiór | evo_switch: między podziałami | evo_switch: w obrębie podziału | rand_replay: między podziałami | rand_replay: w obrębie podziału | rand_switch: między podziałami | rand_switch: w obrębie podziału |
|---|---|---|---|---|---|---|
| circles | 0,0023 | 0,0010 | 0,0024 | 0,0013 | 0,0028 | 0,0027 |
| classification2d | 0,0041 | 0,0037 | 0,0057 | 0,0042 | 0,0049 | 0,0053 |
| digits | 0,0051 | 0,0085 | 0,0092 | 0,0080 | 0,0069 | 0,0129 |
| magic | 0,0064 | 0,0075 | 0,0062 | 0,0039 | 0,0068 | 0,0059 |
| moons | 0,0034 | 0,0029 | 0,0027 | 0,0014 | 0,0069 | 0,0065 |
| spiral | 0,0028 | 0,0066 | 0,0039 | 0,0083 | 0,0055 | 0,0064 |

## 4. Stary algorytm (evo_mem) a nowy (evo_switch), te same podziały i seedy

Porównanie opisowe (różne automaty faz i polityka rozszerzania V′).

| zbiór | n | \|V′\| stary | \|V′\| nowy | stosunek | BA test stary | BA test nowy | różnica | 95% CI | CI wyklucza 0 | po korekcie Bonferroniego |
|---|---|---|---|---|---|---|---|---|---|---|
| circles | 20 | 1202 | 624 | 0,52 | 0,9805 | 0,9806 | 0,0001 | [-0,0009; 0,0011] | nie | nie |
| classification2d | 20 | 399 | 517 | 1,30 | 0,9371 | 0,9395 | 0,0024 | [-0,0003; 0,0051] | nie | nie |
| digits | 20 | 707 | 170 | 0,24 | 0,9652 | 0,9612 | -0,0040 | [-0,0086; 0,0004] | nie | nie |
| magic | 20 | 2442 | 365 | 0,15 | 0,8358 | 0,8322 | -0,0036 | [-0,0067; -0,0008] | tak | nie |
| moons | 20 | 276 | 375 | 1,36 | 0,9642 | 0,9640 | -0,0001 | [-0,0022; 0,0022] | nie | nie |
| spiral | 20 | 530 | 447 | 0,84 | 0,9113 | 0,9119 | 0,0006 | [-0,0036; 0,0051] | nie | nie |

## 5. Powody zakończenia (nowe 270 przebiegów)

| powód zakończenia | evo_switch | rand_replay | rand_switch |
|---|---|---|---|
| stop_k_limit | 17 | 0 | 13 |
| stop_no_k_growth_signal | 40 | 0 | 3 |
| stop_no_progress_after_vprime_retry | 31 | 0 | 70 |
| stop_perfect_on_max_vprime | 0 | 0 | 2 |
| stop_replay_end | 0 | 90 | 0 |
| stop_tprime_no_progress | 2 | 0 | 2 |

## 6. Niezmienniki na wszystkich 10519 cyklach nowych przebiegów

| sprawdzenie | wynik |
|---|---|
| cykle V′ bez poprzedzającej decyzji przełączenia (lub kontynuacji fazy V′) | 0 |
| faza V′ zakończona decyzją „do T′”, a następny cykl nie jest T′ | 0 |
| `prev_phase` niezgodne z poprzednim cyklem | 0 |
| jednoczesne zwiększenie K i rozszerzenie V′ w jednym cyklu | 0 |
| zmiana K w cyklu V′ / zmiana V′ w cyklu T′ / V′ urosło inaczej niż o jedno ΔV′ | 0 / 0 / 0 |
| BA(V′)=1 i wzrost K w tym samym cyklu (warianty z własnym automatem) | 0 |
| \|V′\| powyżej limitu / `vprime_fraction` ≠ \|V′\|/\|V\| | 0 / 0 |
| więcej niż jedna próba V′ po zastoju przy tym samym K | 0 |
| przebiegi zakończone bezpiecznikiem `stop_max_cycles` | 0 |
| pary `rand_replay` z identycznym końcowym \|V′\| co `evo_switch` | 90 / 90 |

Zachowanie zgodne ze specyfikacją na wszystkich cyklach: TAK.

## 7. Ograniczenia

* Tylko 5 seedów na podział; różnice rzędu 0,001 BA są poniżej rozdzielczości tego układu.
* Szachownica nie była badana na podziałach dodatkowych (jej dane nie zależą od podziału).
* Wartości domyślne automatu nie były strojone; bramka ρ_SV jest włączona (`RAPORT_V2.md`).
