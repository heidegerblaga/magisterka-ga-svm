# followup_phase_switch v2 — pełne badanie (seria `v2_full`)

Raport generowany przez `scripts/followup_phase_switch/report_full.py` z tabel `reports/followup_phase_switch/v2/tables/`.
Zakres: podział bazowy 2026 (jak w badaniu `followup_stop_rule`), 7 zbiorów × 20 seedów (42–61) × 3 warianty
(`evo_switch`, `rand_replay` — główne porównanie, `rand_switch` — pomocniczo) = **420 przebiegów**, ustawienia domyślne automatu
(`RAPORT_V2.md` §3), `max_vprime_rows: null`, `k_growth_requires_sv_signal: true`. Dodatkowe podziały danych (2027–2029) **nie** były uruchamiane.
Cykli (kroków automatu) w serii: 17152. Opis automatu i warunków: `RAPORT_V2.md` §2; audyt: `AUDYT_v2.md`.
`pytest`: 140 passed (0 błędów). Wyniki v1, smoke v2 i stary tryb `dagasvm_phase` są nietknięte.

Trudność przyrostu ΔV′ i jakość modelu są osobnymi właściwościami; raport nie wnioskuje z jednej o drugiej. BA na teście i na pełnym V to wyłącznie obserwacje.
`rand_replay` ma identyczny z `evo_switch` (ten sam seed) harmonogram faz, K i \|ΔV′\| — różni się tylko wyborem próbek do ΔV′, więc \|V′\|, K i liczba cykli są
z konstrukcji równe; `rand_switch` stosuje te same reguły przy własnym harmonogramie, więc jego \|V′\| bywa mniejszy.

## 1. Wyniki per zbiór i wariant (n = 20 seedów w każdej komórce)

Pełny SVM (cały zbiór treningowy) ma BA test: chessboard 0,9885, circles 0,9804, classification2d 0,9479, digits 0,9578, magic 0,8443, moons 0,9700, spiral 0,9317.

| seria | zbiór | wariant | n | \|V′\| (śr.) | min–max | \|V\| | \|V′\|/\|V\| | rozszerzenia V′ | cykle (T′/V′) | wejścia do V′ (opanowanie/próba) | K końc. | BA na V′ (model rep.) | BA na ΔV′ (śr. po krokach V′) | BA pełne V (obserwacja) | BA test (obserwacja) | czas [s] | powód zakończenia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_full | chessboard | evo_switch | 20 | 498 | 320–760 | 27853 | 0,018 | 23,9 | 32,0/23,9 | 1,8/3,8 | 876 | 0,9572 | 0,6725 | 0,9691 | 0,9732 ± 0,0038 | 199 | no_k_growth_signal:18, no_progress_after_vprime_retry:2 |
| v2_full | chessboard | rand_replay | 20 | 498 | 320–760 | 27853 | 0,018 | 23,9 | 32,0/23,9 | 2,4/3,2 | 876 | 0,9898 | 0,9039 | 0,9661 | 0,9700 ± 0,0033 | 139 | replay_end:20 |
| v2_full | chessboard | rand_switch | 20 | 443 | 240–800 | 27853 | 0,016 | 21,1 | 29,1/21,1 | 2,9/2,8 | 844 | 0,9928 | 0,8755 | 0,9636 | 0,9678 ± 0,0072 | 128 | no_progress_after_vprime_retry:18, no_k_growth_signal:2 |
| v2_full | circles | evo_switch | 20 | 568 | 320–760 | 4800 | 0,118 | 27,4 | 31,9/27,4 | 1,1/5,7 | 330 | 0,9161 | 0,7944 | 0,9799 | 0,9777 ± 0,0015 | 46 | no_progress_after_vprime_retry:10, no_k_growth_signal:10 |
| v2_full | circles | rand_replay | 20 | 568 | 320–760 | 4800 | 0,118 | 27,4 | 31,9/27,4 | 1,4/5,3 | 330 | 0,9863 | 0,9673 | 0,9790 | 0,9781 ± 0,0017 | 26 | replay_end:20 |
| v2_full | circles | rand_switch | 20 | 314 | 140–540 | 4800 | 0,065 | 14,7 | 16,9/14,7 | 1,9/2,1 | 232 | 0,9885 | 0,9492 | 0,9776 | 0,9766 ± 0,0029 | 10 | no_progress_after_vprime_retry:20 |
| v2_full | classification2d | evo_switch | 20 | 464 | 260–740 | 4800 | 0,097 | 22,2 | 28,4/22,2 | 0,7/4,8 | 172 | 0,8412 | 0,6904 | 0,9400 | 0,9422 ± 0,0027 | 28 | no_k_growth_signal:11, no_progress_after_vprime_retry:9 |
| v2_full | classification2d | rand_replay | 20 | 464 | 260–740 | 4800 | 0,097 | 22,2 | 28,4/22,2 | 0,7/4,8 | 172 | 0,9546 | 0,9263 | 0,9376 | 0,9418 ± 0,0035 | 16 | replay_end:20 |
| v2_full | classification2d | rand_switch | 20 | 400 | 120–720 | 4800 | 0,083 | 19,0 | 22,6/19,0 | 0,7/3,8 | 140 | 0,9555 | 0,9262 | 0,9375 | 0,9411 ± 0,0038 | 11 | no_progress_after_vprime_retry:19, no_k_growth_signal:1 |
| v2_full | digits | evo_switch | 20 | 170 | 170–170 | 719 | 0,236 | 2,0 | 7,0/2,0 | 0,5/0,5 | 70 | 0,9607 | 0,7377 | 0,9660 | 0,9600 ± 0,0099 | 13 | k_limit:19, tprime_no_progress:1 |
| v2_full | digits | rand_replay | 20 | 170 | 170–170 | 719 | 0,236 | 2,0 | 7,0/2,0 | 0,5/0,5 | 70 | 0,9810 | 0,9212 | 0,9656 | 0,9585 ± 0,0109 | 12 | replay_end:20 |
| v2_full | digits | rand_switch | 20 | 170 | 170–170 | 719 | 0,236 | 2,0 | 6,8/2,0 | 0,5/0,5 | 67 | 0,9799 | 0,9216 | 0,9650 | 0,9549 ± 0,0106 | 11 | k_limit:17, tprime_no_progress:1, perfect_on_max_vprime:1, no_progress_after_vprime_retry:1 |
| v2_full | magic | evo_switch | 20 | 392 | 240–680 | 3200 | 0,123 | 18,6 | 20,0/18,6 | 0,4/3,4 | 1125 | 0,7000 | 0,4403 | 0,8392 | 0,8371 ± 0,0074 | 118 | no_progress_after_vprime_retry:10, k_limit:10 |
| v2_full | magic | rand_replay | 20 | 392 | 240–680 | 3200 | 0,123 | 18,6 | 20,0/18,6 | 0,4/3,4 | 1125 | 0,8696 | 0,8042 | 0,8419 | 0,8392 ± 0,0037 | 106 | replay_end:20 |
| v2_full | magic | rand_switch | 20 | 315 | 180–540 | 3200 | 0,098 | 14,8 | 19,9/14,8 | 0,4/2,9 | 1114 | 0,8784 | 0,7921 | 0,8355 | 0,8338 ± 0,0094 | 95 | no_progress_after_vprime_retry:17, k_limit:3 |
| v2_full | moons | evo_switch | 20 | 361 | 200–540 | 4800 | 0,075 | 17,1 | 20,4/17,1 | 0,9/3,5 | 88 | 0,8944 | 0,7529 | 0,9657 | 0,9676 ± 0,0030 | 16 | no_k_growth_signal:13, no_progress_after_vprime_retry:7 |
| v2_full | moons | rand_replay | 20 | 361 | 200–540 | 4800 | 0,075 | 17,1 | 20,4/17,1 | 1,1/3,4 | 88 | 0,9768 | 0,9529 | 0,9656 | 0,9663 ± 0,0027 | 7 | replay_end:20 |
| v2_full | moons | rand_switch | 20 | 279 | 140–580 | 4800 | 0,058 | 12,9 | 14,8/12,9 | 1,6/1,9 | 73 | 0,9813 | 0,9553 | 0,9631 | 0,9649 ± 0,0034 | 5 | no_progress_after_vprime_retry:20 |
| v2_full | spiral | evo_switch | 20 | 435 | 180–900 | 4800 | 0,091 | 20,8 | 36,1/20,8 | 0,7/4,7 | 234 | 0,8573 | 0,5931 | 0,9192 | 0,9155 ± 0,0065 | 142 | no_k_growth_signal:11, no_progress_after_vprime_retry:9 |
| v2_full | spiral | rand_replay | 20 | 435 | 180–900 | 4800 | 0,091 | 20,8 | 36,1/20,8 | 0,7/4,7 | 234 | 0,9507 | 0,8700 | 0,9159 | 0,9145 ± 0,0066 | 119 | replay_end:20 |
| v2_full | spiral | rand_switch | 20 | 407 | 240–920 | 4800 | 0,085 | 19,4 | 28,1/19,4 | 0,7/4,0 | 176 | 0,9446 | 0,8426 | 0,9110 | 0,9098 ± 0,0101 | 71 | no_progress_after_vprime_retry:17, no_k_growth_signal:3 |

Rozmiar V′ względem V (wszystkie warianty): chessboard śr. 0,017, maks. 0,029, circles śr. 0,101, maks. 0,158, classification2d śr. 0,092, maks. 0,154, digits śr. 0,236, maks. 0,236, magic śr. 0,114, maks. 0,212, moons śr. 0,070, maks. 0,121, spiral śr. 0,089, maks. 0,192.
Maks. \|V′\| na Szachownicy = 800 (stary algorytm: ≈ 3600).

## 2. Ewolucja a losowanie przy tym samym harmonogramie (evo_switch − rand_replay)

Różnice sparowane po seedach (n = 20), 95% CI z bootstrapu (10 000 prób). Różnica liczby próbek/K/cykli jest zerowa z konstrukcji.

| zbiór | metryka | n | evo_switch | rand_replay | różnica | 95% CI | CI wyklucza 0 | CI po korekcie Bonferroniego | evo_switch większe/mniejsze |
|---|---|---|---|---|---|---|---|---|---|
| chessboard | test_ba | 20 | 0,9732 | 0,9700 | 0,0031 | [0,0013; 0,0049] | tak | tak | 14/6 |
| chessboard | fullv_ba | 20 | 0,9691 | 0,9661 | 0,0029 | [0,0010; 0,0047] | tak | tak | 14/6 |
| chessboard | vprime_rows | 20 | 498,0000 | 498,0000 | 0,0000 | [0,0000; 0,0000] | nie | nie | 0/0 |
| chessboard | mean_delta_difficulty | 20 | 0,2802 | 0,0982 | 0,1820 | [0,1718; 0,1929] | tak | tak | 20/0 |
| circles | test_ba | 20 | 0,9777 | 0,9781 | -0,0004 | [-0,0015; 0,0006] | nie | nie | 9/10 |
| circles | fullv_ba | 20 | 0,9799 | 0,9790 | 0,0009 | [0,0003; 0,0014] | tak | tak | 15/4 |
| circles | vprime_rows | 20 | 568,0000 | 568,0000 | 0,0000 | [0,0000; 0,0000] | nie | nie | 0/0 |
| circles | mean_delta_difficulty | 20 | 0,1705 | 0,0334 | 0,1372 | [0,1313; 0,1432] | tak | tak | 20/0 |
| classification2d | test_ba | 20 | 0,9422 | 0,9418 | 0,0004 | [-0,0015; 0,0023] | nie | nie | 10/10 |
| classification2d | fullv_ba | 20 | 0,9400 | 0,9376 | 0,0024 | [0,0007; 0,0042] | tak | nie | 16/4 |
| classification2d | vprime_rows | 20 | 464,0000 | 464,0000 | 0,0000 | [0,0000; 0,0000] | nie | nie | 0/0 |
| classification2d | mean_delta_difficulty | 20 | 0,2703 | 0,0744 | 0,1959 | [0,1872; 0,2047] | tak | tak | 20/0 |
| digits | test_ba | 20 | 0,9600 | 0,9585 | 0,0015 | [-0,0033; 0,0075] | nie | nie | 7/9 |
| digits | fullv_ba | 20 | 0,9660 | 0,9656 | 0,0003 | [-0,0029; 0,0035] | nie | nie | 9/9 |
| digits | vprime_rows | 20 | 170,0000 | 170,0000 | 0,0000 | [0,0000; 0,0000] | nie | nie | 0/0 |
| digits | mean_delta_difficulty | 20 | 0,1985 | 0,0837 | 0,1148 | [0,1008; 0,1283] | tak | tak | 20/0 |
| magic | test_ba | 20 | 0,8371 | 0,8392 | -0,0021 | [-0,0050; 0,0007] | nie | nie | 7/13 |
| magic | fullv_ba | 20 | 0,8392 | 0,8419 | -0,0027 | [-0,0045; -0,0007] | tak | tak | 6/14 |
| magic | vprime_rows | 20 | 392,0000 | 392,0000 | 0,0000 | [0,0000; 0,0000] | nie | nie | 0/0 |
| magic | mean_delta_difficulty | 20 | 0,3697 | 0,1808 | 0,1890 | [0,1722; 0,2073] | tak | tak | 20/0 |
| moons | test_ba | 20 | 0,9676 | 0,9663 | 0,0014 | [-0,0003; 0,0031] | nie | nie | 10/8 |
| moons | fullv_ba | 20 | 0,9657 | 0,9656 | 0,0002 | [-0,0012; 0,0016] | nie | nie | 9/9 |
| moons | vprime_rows | 20 | 361,0000 | 361,0000 | 0,0000 | [0,0000; 0,0000] | nie | nie | 0/0 |
| moons | mean_delta_difficulty | 20 | 0,2111 | 0,0479 | 0,1632 | [0,1558; 0,1704] | tak | tak | 20/0 |
| spiral | test_ba | 20 | 0,9155 | 0,9145 | 0,0010 | [-0,0027; 0,0045] | nie | nie | 12/8 |
| spiral | fullv_ba | 20 | 0,9192 | 0,9159 | 0,0034 | [0,0005; 0,0058] | tak | nie | 15/5 |
| spiral | vprime_rows | 20 | 435,0000 | 435,0000 | 0,0000 | [0,0000; 0,0000] | nie | nie | 0/0 |
| spiral | mean_delta_difficulty | 20 | 0,3708 | 0,1317 | 0,2391 | [0,2268; 0,2511] | tak | tak | 20/0 |

Przedział ufności różnicy BA test wyklucza zero w **1** z 7 zbiorów: chessboard (0,0031); po korekcie Bonferroniego na 7 zbiorów (przedziały
99,3 %) — w **1**. Przedziały w tabeli są liczone osobno dla każdej metryki i zbioru, bez korekty na wielokrotność porównań
(tabela ma 28 wierszy), więc pojedyncze „tak” przy BA pełnego V (obserwacja) nie jest mocnym dowodem. Poza zbiorami wskazanymi wyżej nie ma podstaw,
by twierdzić o przewadze któregokolwiek doboru. Ewolucja buduje przyrosty o wyższym odsetku błędów modelu (wiersz `mean_delta_difficulty`) — to jest własność przyrostu, a nie dowód lepszego modelu.

Pomocniczo, evo_switch − rand_switch (różne harmonogramy; różnica miesza efekt doboru z efektem rozmiaru V′):

| zbiór | metryka | n | evo_switch | rand_switch | różnica | 95% CI | CI wyklucza 0 | CI po korekcie Bonferroniego | evo_switch większe/mniejsze |
|---|---|---|---|---|---|---|---|---|---|
| chessboard | test_ba | 20 | 0,9732 | 0,9678 | 0,0054 | [0,0022; 0,0089] | tak | tak | 13/7 |
| chessboard | vprime_rows | 20 | 498,0000 | 443,0000 | 55,0000 | [-6,0000; 116,0000] | nie | nie | 12/6 |
| circles | test_ba | 20 | 0,9777 | 0,9766 | 0,0011 | [-0,0002; 0,0024] | nie | nie | 12/5 |
| circles | vprime_rows | 20 | 568,0000 | 314,0000 | 254,0000 | [190,0000; 317,0000] | tak | tak | 19/1 |
| classification2d | test_ba | 20 | 0,9422 | 0,9411 | 0,0011 | [-0,0011; 0,0035] | nie | nie | 9/11 |
| classification2d | vprime_rows | 20 | 464,0000 | 400,0000 | 64,0000 | [-25,0000; 151,0000] | nie | nie | 11/8 |
| digits | test_ba | 20 | 0,9600 | 0,9549 | 0,0051 | [-0,0009; 0,0114] | nie | nie | 12/6 |
| digits | vprime_rows | 20 | 170,0000 | 170,0000 | 0,0000 | [0,0000; 0,0000] | nie | nie | 0/0 |
| magic | test_ba | 20 | 0,8371 | 0,8338 | 0,0033 | [-0,0021; 0,0090] | nie | nie | 12/8 |
| magic | vprime_rows | 20 | 392,0000 | 315,0000 | 77,0000 | [12,0000; 145,0000] | tak | nie | 14/4 |
| moons | test_ba | 20 | 0,9676 | 0,9649 | 0,0027 | [0,0006; 0,0049] | tak | nie | 16/4 |
| moons | vprime_rows | 20 | 361,0000 | 279,0000 | 82,0000 | [22,0000; 145,0000] | tak | nie | 13/6 |
| spiral | test_ba | 20 | 0,9155 | 0,9098 | 0,0057 | [0,0011; 0,0106] | tak | nie | 16/4 |
| spiral | vprime_rows | 20 | 435,0000 | 407,0000 | 28,0000 | [-39,0000; 95,0000] | nie | nie | 13/7 |

## 3. Czy mniejsze V′ pogarsza jakość? Porównanie ze starym algorytmem (evo_mem, ten sam podział i seedy)

Stary: `dagasvm_phase`, reguła rozszerzona, `followup_stop_rule`. Nowy: `evo_switch` (v2). To porównanie opisowe (różne algorytmy przełączania faz i K), nie izoluje wpływu rozmiaru V′.

| zbiór | n | \|V′\| stary (śr.) | \|V′\| nowy (śr.) | stosunek nowy/stary | K końc. stary | K końc. nowy | BA test stary | BA test nowy | różnica (nowy − stary) | 95% CI | CI wyklucza 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chessboard | 20 | 3528 | 498 | 0,14 | 835 | 876 | 0,9732 | 0,9732 | -0,0000 | [-0,0025; 0,0027] | nie |
| circles | 20 | 1188 | 568 | 0,48 | 303 | 330 | 0,9772 | 0,9777 | 0,0006 | [-0,0004; 0,0015] | nie |
| classification2d | 20 | 400 | 464 | 1,16 | 135 | 172 | 0,9417 | 0,9422 | 0,0005 | [-0,0013; 0,0023] | nie |
| digits | 20 | 705 | 170 | 0,24 | 53 | 70 | 0,9643 | 0,9600 | -0,0043 | [-0,0099; 0,0003] | nie |
| magic | 20 | 2290 | 392 | 0,17 | 994 | 1125 | 0,8382 | 0,8371 | -0,0011 | [-0,0039; 0,0017] | nie |
| moons | 20 | 256 | 361 | 1,41 | 76 | 88 | 0,9648 | 0,9676 | 0,0028 | [0,0001; 0,0059] | tak |
| spiral | 20 | 463 | 435 | 0,94 | 175 | 234 | 0,9123 | 0,9155 | 0,0032 | [-0,0003; 0,0069] | nie |

## 4. Skąd pochodzą wejścia do fazy V′ i jak kończą się próby

Wejścia do fazy V′ (warianty z własnym automatem): **1182**, z czego z opanowania V′ (BA=1): **293** (24,8 %),
z zastoju T′ (próba): 889. Kontrole po próbie V′: 1271, z czego poziom odniesienia osiągnięty: 730.

| zbiór | wariant | wejścia do V′ | z opanowania (BA(V′)=1) | z zastoju T′ (próba) | rozmiary V′ przy opanowaniu |
|---|---|---|---|---|---|
| chessboard | evo_switch | 111 | 36 | 75 | 20, 80, 120, 140, 180 |
| chessboard | rand_switch | 114 | 58 | 56 | 20, 80, 100, 120, 140, 160, 180, 200 |
| circles | evo_switch | 135 | 21 | 114 | 20, 120 |
| circles | rand_switch | 81 | 38 | 43 | 20, 60, 80, 100, 120, 180, 220, 260 |
| classification2d | evo_switch | 111 | 14 | 97 | 20 |
| classification2d | rand_switch | 90 | 14 | 76 | 20 |
| digits | evo_switch | 20 | 10 | 10 | 50 |
| digits | rand_switch | 20 | 10 | 10 | 50 |
| magic | evo_switch | 76 | 8 | 68 | 20 |
| magic | rand_switch | 66 | 8 | 58 | 20 |
| moons | evo_switch | 89 | 19 | 70 | 20 |
| moons | rand_switch | 70 | 31 | 39 | 20, 60, 80, 100, 120, 140, 180, 220 |
| spiral | evo_switch | 107 | 13 | 94 | 20 |
| spiral | rand_switch | 92 | 13 | 79 | 20 |

### Powody zakończenia

| zbiór | wariant | n | no_progress_after_vprime_retry | no_k_growth_signal | k_limit | tprime_no_progress | perfect_on_max_vprime | replay_end | max_cycles |
|---|---|---|---|---|---|---|---|---|---|
| chessboard | evo_switch | 20 | 2 | 18 | 0 | 0 | 0 | 0 | 0 |
| chessboard | rand_replay | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 0 |
| chessboard | rand_switch | 20 | 18 | 2 | 0 | 0 | 0 | 0 | 0 |
| circles | evo_switch | 20 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| circles | rand_replay | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 0 |
| circles | rand_switch | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| classification2d | evo_switch | 20 | 9 | 11 | 0 | 0 | 0 | 0 | 0 |
| classification2d | rand_replay | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 0 |
| classification2d | rand_switch | 20 | 19 | 1 | 0 | 0 | 0 | 0 | 0 |
| digits | evo_switch | 20 | 0 | 0 | 19 | 1 | 0 | 0 | 0 |
| digits | rand_replay | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 0 |
| digits | rand_switch | 20 | 1 | 0 | 17 | 1 | 1 | 0 | 0 |
| magic | evo_switch | 20 | 10 | 0 | 10 | 0 | 0 | 0 | 0 |
| magic | rand_replay | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 0 |
| magic | rand_switch | 20 | 17 | 0 | 3 | 0 | 0 | 0 | 0 |
| moons | evo_switch | 20 | 7 | 13 | 0 | 0 | 0 | 0 | 0 |
| moons | rand_replay | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 0 |
| moons | rand_switch | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| spiral | evo_switch | 20 | 9 | 11 | 0 | 0 | 0 | 0 | 0 |
| spiral | rand_replay | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 0 |
| spiral | rand_switch | 20 | 17 | 3 | 0 | 0 | 0 | 0 | 0 |

## 5. Koszt obliczeń (średnia na przebieg)

| seria | zbiór | wariant | n | treningi SVM | wywołania predict | wywołania decision_function | T′: wiersze V′ ocenione (faktycznie) | T′: nominalnie (bez wcześn. zatrzymania GA) | ΔV′: wiersze ocenione | kandydaci (suma po doborach) | V′: wiersze predict | Val-GA nominalnie [wiersze] | obserwacja [wiersze] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_full | chessboard | evo_switch | 20 | 6577 | 6652 | 24 | 1549530 | 7392570 | 478 | 659389 | 660345 | 301140 | 1603205 |
| v2_full | chessboard | rand_replay | 20 | 5093 | 5141 | 24 | 1183950 | 7392570 | 478 | 659389 | 956 | 0 | 1603205 |
| v2_full | chessboard | rand_switch | 20 | 5159 | 5204 | 21 | 1205160 | 7059630 | 423 | 583901 | 846 | 0 | 1453139 |
| v2_full | circles | evo_switch | 20 | 5610 | 5698 | 27 | 1647810 | 8643420 | 548 | 123346 | 124442 | 345240 | 229680 |
| v2_full | circles | rand_replay | 20 | 4149 | 4204 | 27 | 1211400 | 8643420 | 548 | 123346 | 1096 | 0 | 229680 |
| v2_full | circles | rand_switch | 20 | 2525 | 2556 | 15 | 544470 | 3311730 | 294 | 68015 | 588 | 0 | 122040 |
| v2_full | classification2d | evo_switch | 20 | 5151 | 5222 | 22 | 1261560 | 6532320 | 444 | 100987 | 101875 | 279720 | 204480 |
| v2_full | classification2d | rand_replay | 20 | 4239 | 4283 | 22 | 1035900 | 6532320 | 444 | 100987 | 888 | 0 | 204480 |
| v2_full | classification2d | rand_switch | 20 | 3483 | 3525 | 19 | 805380 | 4940160 | 380 | 86865 | 760 | 0 | 163080 |
| v2_full | digits | evo_switch | 20 | 1254 | 1260 | 2 | 141525 | 654720 | 120 | 1238 | 1478 | 75600 | 7553 |
| v2_full | digits | rand_replay | 20 | 1158 | 1162 | 2 | 124845 | 654720 | 120 | 1238 | 240 | 0 | 7553 |
| v2_full | digits | rand_switch | 20 | 1118 | 1123 | 2 | 118575 | 631005 | 120 | 1238 | 240 | 0 | 7391 |
| v2_full | magic | evo_switch | 20 | 4847 | 4906 | 19 | 818700 | 3042960 | 372 | 55662 | 56406 | 234360 | 96000 |
| v2_full | magic | rand_replay | 20 | 4476 | 4514 | 19 | 760860 | 3042960 | 372 | 55662 | 744 | 0 | 96000 |
| v2_full | magic | rand_switch | 20 | 4485 | 4518 | 15 | 679770 | 2683050 | 295 | 44644 | 590 | 0 | 95520 |
| v2_full | moons | evo_switch | 20 | 3623 | 3678 | 17 | 742260 | 3809280 | 341 | 78473 | 79155 | 214830 | 146520 |
| v2_full | moons | rand_replay | 20 | 2806 | 2840 | 17 | 574500 | 3809280 | 341 | 78473 | 682 | 0 | 146520 |
| v2_full | moons | rand_switch | 20 | 2179 | 2207 | 13 | 443250 | 2693280 | 259 | 60008 | 518 | 0 | 106920 |
| v2_full | spiral | evo_switch | 20 | 6972 | 7039 | 21 | 1488030 | 7499520 | 415 | 94434 | 95264 | 261450 | 259920 |
| v2_full | spiral | rand_replay | 20 | 6365 | 6406 | 21 | 1323900 | 7499520 | 415 | 94434 | 830 | 0 | 259920 |
| v2_full | spiral | rand_switch | 20 | 5043 | 5085 | 19 | 1026270 | 5589300 | 387 | 88323 | 774 | 0 | 201960 |

## 6. Niezmienniki na wszystkich 17152 cyklach

| sprawdzenie | wynik |
|---|---|
| cykle V′ bez poprzedzającej decyzji przełączenia (lub kontynuacji fazy V′) | 0 |
| faza V′ zakończona decyzją „do T′”, a następny cykl nie jest T′ | 0 |
| `prev_phase` niezgodne z poprzednim cyklem | 0 |
| cykle, w których jednocześnie wzrosło K i rozszerzono V′ | 0 |
| zmiana K w cyklu V′ / zmiana V′ w cyklu T′ | 0 / 0 |
| cykl V′, w którym V′ urosło inaczej niż o jedno ΔV′ | 0 |
| BA(V′)=1 i wzrost K w tym samym cyklu (warianty z własnym automatem) | 0 |
| j.w. w `rand_replay` (K narzucone harmonogramem evo) | 141 (informacyjnie) |
| \|V′\| powyżej limitu / `vprime_fraction` ≠ \|V′\|/\|V\| | 0 / 0 |
| więcej niż jedna próba V′ po zastoju przy tym samym K | 0 |
| przebiegi zakończone bezpiecznikiem `stop_max_cycles` | 0 |
| pary `rand_replay` z identycznym końcowym \|V′\| co `evo_switch` | 140 / 140 |
| kontrole po próbie V′ bez osiągnięcia poziomu: dalej zwiększenie K / stop | 382 / 159 (z 541) |

Zachowanie zgodne ze specyfikacją na wszystkich cyklach: TAK.

## 7. Ograniczenia

* Jeden podział danych (2026); wariancja między podziałami nie była badana.
* Wartości domyślne automatu nie były strojone; wrażliwość (przyrost ΔV′, progi trudności, `max_no_improvement_k_increases`) nie była badana.
* `k_growth_requires_sv_signal: true` (bramka ρ_SV) wyznacza zakończenie w 69 przebiegach i ogranicza K; wpływ bramki na jakość był sprawdzony tylko na Szachownicy (smoke, `RAPORT_V2.md`).
* Digits: limit 0,25·\|V\| (179 próbek) jest praktycznie osiągany i K dochodzi do limitu klasy (`stop_k_limit`).
* Skład klasowy ΔV′ przy doborze ewolucyjnym i losowym nie jest wymuszany (zob. `RAPORT_V2.md` §9).
* Koszt doboru ewolucyjnego obejmuje ocenę całej puli kandydatów raz na rozszerzenie.
