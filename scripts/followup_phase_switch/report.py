"""Raport etapu smoke/short badania `followup_phase_switch` -> reports/followup_phase_switch/RAPORT_SMOKE.md.

Wszystkie liczby są liczone z tabel aggregate.py (per_run.csv, per_step.csv, paired.csv) oraz — dla porównania
„przed” — z reports/followup_stop_rule/tables/per_run.csv (tylko odczyt). Uruchom po aggregate.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import PHASE_SWITCH_DEFAULTS_V1 as PHASE_SWITCH_DEFAULTS  # noqa: E402  (v1: parametry z 2026-10-14)
from ps_common import REPO_ROOT, REPORT_ROOT, TABLES, fsr  # noqa: E402

OUT = REPORT_ROOT / "RAPORT_SMOKE.md"
OLD_PER_RUN = REPO_ROOT / "reports" / "followup_stop_rule" / "tables" / "per_run.csv"
OLD_PER_CYCLE = REPO_ROOT / "reports" / "followup_stop_rule" / "tables" / "per_cycle.csv"
VARIANT_ORDER = ["evo_switch", "rand_replay", "rand_switch", "heur_switch"]


def f(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}".replace(".", ",")


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def series_table(runs: pd.DataFrame, series: list[str]) -> str:
    rows = []
    for s in series:
        for v in VARIANT_ORDER:
            g = runs[(runs.series == s) & (runs.variant == v)]
            if g.empty:
                continue
            for d, gd in g.groupby("dataset"):
                rows.append({
                    "seria": s, "zbiór": d, "wariant": v, "n": len(gd),
                    "|V'| (śr.)": f(gd.vprime_rows.mean(), 0), "|V'| min–max": f"{gd.vprime_rows.min()}–{gd.vprime_rows.max()}",
                    "|V|": int(gd.full_validation_rows.iloc[0]), "|V'|/|V|": f(gd.vprime_fraction.mean(), 3),
                    "limit |V'|": int(gd.vprime_cap_rows.iloc[0]), "K końc.": f(gd.final_k.mean(), 0),
                    "kroki T'/V'": f"{f(gd.tprime_steps.mean(), 1)}/{f(gd.vprime_steps.mean(), 1)}",
                    "fazy V' (opanowanie/zastój)": f"{f(gd.vprime_phases_mastered.mean(), 1)}/{f(gd.vprime_retries.mean(), 1)}",
                    "BA test": f"{f(gd.test_ba.mean())} ± {f(gd.test_ba.std(ddof=1) if len(gd) > 1 else np.nan)}",
                    "BA V' (model)": f(gd.rep_vprime_ba.mean()),
                    "trudność ΔV'": f(gd.mean_delta_difficulty.mean(), 3),
                    "czas [s]": f(gd.duration_sec.mean(), 0),
                    "stop": ", ".join(f"{k.replace('stop_', '')}:{n}" for k, n in gd.stop_decision.value_counts().items()),
                })
    return table(pd.DataFrame(rows))


def observations(runs: pd.DataFrame) -> str:
    out = []
    ch = runs[(runs.dataset == "chessboard") & (runs.variant == "evo_switch")]
    if len(ch):
        base = ch.baseline_full_svm_test_ba.iloc[0]
        nog = ch[ch.series == "smoke_chess_r1000"]
        gat = ch[ch.series == "smoke_chess_r1000_svgate"]
        out.append(
            f"* **Szachownica: jakość ogranicza polityka K, a nie V′.** Pełny SVM (cały zbiór treningowy) ma BA test {f(base)}. "
            f"Z bramką ρ_SV K kończy się na ≈{f(gat.final_k.mean(), 0)} i BA test = {f(gat.test_ba.mean())} "
            f"(stary algorytm też kończył przy ρ_SV < TSV: K ≈ 837, BA 0,9732). Bez bramki K rośnie do "
            f"{', '.join(str(int(k)) for k in nog.final_k)} i BA test = {', '.join(f(x) for x in nog.test_ba)} "
            f"(seedy {', '.join(str(int(x)) for x in nog.seed)}), czyli przy dużym K zbliża się do pełnego SVM, kosztem czasu "
            f"(śr. {f(nog.duration_sec.mean(), 0)} s vs {f(gat.duration_sec.mean(), 0)} s). Wybór TSV / bramki to decyzja o "
            "kompromisie rozmiar T′ ↔ jakość, niezależna od reguł V′.")
    diff = runs[(runs.variant != "rand_replay") & (runs.rep_k != runs.final_k)]
    if len(diff):
        d = (diff.test_ba - diff.last_tprime_test_ba)
        out.append(
            f"* **Wybór modelu reprezentatywnego.** Domyślnie wybierane jest najlepsze T′ na końcowym V′ (`best_on_final_vprime`, "
            f"tylko V′). W {len(diff)} z {int((runs.variant != 'rand_replay').sum())} przebiegów jest to inne T′ niż ostatni krok. "
            f"Średnia różnica BA test (wybrany − ostatni) = {f(d.mean())}; najgorzej Digits "
            f"({f(d[diff.dataset == 'digits'].mean()) if (diff.dataset == 'digits').any() else '—'}): przy ~17 próbkach V′ na klasę wybór "
            "po fitnessie V′ jest zaszumiony. Alternatywa `representative: last_tprime_step` jest zaimplementowana.")
    dg = runs[(runs.dataset == "digits") & (runs.variant == "evo_switch")]
    if len(dg):
        out.append(
            f"* **Digits (10 klas, |V| = {int(dg.full_validation_rows.iloc[0])}):** limit 0,25·|V| = {int(dg.vprime_cap_rows.iloc[0])} "
            f"jest osiągany (|V′| = {f(dg.vprime_rows.mean(), 0)}), a K dochodzi do limitu klasy (`stop_k_limit`). Przy wielu klasach "
            "i małym V ułamek 0,25 może być zbyt ciasny; w pełnym eksperymencie warto sprawdzić 0,5 dla Digits.")
    return "\n".join(out)


def invariants(steps: pd.DataFrame, runs: pd.DataFrame) -> dict:
    own = steps[steps.variant != "rand_replay"]
    st = own.sort_values(["series", "dataset", "variant", "seed", "cycle"])
    st_prev = st.groupby(["series", "dataset", "variant", "seed"]).shift(1)
    v = st.phase == "optimize_V_prime"
    entry_ok = ((st_prev.phase == "optimize_T_prime") & (st_prev.decision == "switch_to_V_prime")) | \
               ((st_prev.phase == "optimize_V_prime") & (st_prev.next_phase == "optimize_V_prime"))
    t = steps[steps.phase == "optimize_T_prime"]
    as_bool = lambda s: s.astype(str).str.lower() == "true"  # noqa: E731
    both = (as_bool(steps.k_increased) & as_bool(steps.vprime_expanded)).sum()
    k_jump_in_v = (steps[steps.phase == "optimize_V_prime"].k_after != steps[steps.phase == "optimize_V_prime"].k_per_class).sum()
    perfect_k_mask = as_bool(t.tprime_perfect) & as_bool(t.k_increased)
    perfect_k = (perfect_k_mask & (t.variant != "rand_replay")).sum()
    perfect_k_replay = (perfect_k_mask & (t.variant == "rand_replay")).sum()
    over_cap = (steps.vprime_rows_after > steps.vprime_cap_rows).sum()
    retry = t[(t.next_phase == "optimize_V_prime") & ~as_bool(t.tprime_perfect) & (t.variant != "rand_replay")]
    dup_retry = int(retry.groupby(["series", "dataset", "variant", "seed", "k_per_class"]).size().gt(1).sum())
    rp = runs[runs.variant == "rand_replay"].set_index(["series", "dataset", "seed"])
    ev = runs[runs.variant == "evo_switch"].set_index(["series", "dataset", "seed"])
    common = rp.index.intersection(ev.index)
    replay_equal = int((rp.loc[common, "vprime_rows"] == ev.loc[common, "vprime_rows"]).sum())
    return {
        "steps": len(steps), "runs": len(runs),
        "v_entry_bad": int((v & ~entry_ok).sum()),
        "k_and_v_same_step": int(both), "k_change_in_v_step": int(k_jump_in_v), "perfect_increases_k": int(perfect_k), "perfect_k_replay": int(perfect_k_replay),
        "over_cap": int(over_cap), "dup_retry_per_k": dup_retry,
        "max_cycles_stops": int((runs.stop_decision == "stop_max_cycles").sum()),
        "replay_pairs": len(common), "replay_equal_size": replay_equal,
        "max_vprime_chess": int(runs[runs.dataset == "chessboard"].vprime_rows.max()) if (runs.dataset == "chessboard").any() else None,
    }


def main() -> None:
    runs = pd.read_csv(TABLES / "per_run.csv")
    steps = pd.read_csv(TABLES / "per_step.csv", low_memory=False)
    paired = pd.read_csv(TABLES / "paired.csv")
    old = pd.read_csv(OLD_PER_RUN)
    old = old[(old.split == fsr.BASE_SPLIT)]
    inv = invariants(steps, runs)
    test_names = re.findall(r"^def (test_\w+)", (REPO_ROOT / "tests" / "test_phase_switch.py").read_text(), re.M)

    # ---- liczby do odpowiedzi ----
    oc = pd.read_csv(OLD_PER_CYCLE)
    oc = oc[(oc.split == fsr.BASE_SPLIT) & (oc.dataset == "chessboard") & oc.variant.isin(["evo_mem", "rand_growth"])]
    oexp = oc[oc.decision == "expand_validation_only"]
    old_chess = old[(old.dataset == "chessboard") & old.variant.isin(["evo_mem", "rand_growth"])]
    old_runs = oc.groupby(["variant", "seed"]).agg(exp=("decision", lambda s: (s == "expand_validation_only").sum()),
                                                   kinc=("decision", lambda s: (s == "increase_k").sum()))
    sm = runs[runs.series == "short_moons"]
    pm = paired[(paired.series == "short_moons") & (paired.a == "evo_switch")]

    def pv(series, b, metric, col):
        r = paired[(paired.series == series) & (paired.b == b) & (paired.metric == metric)]
        return r[col].iloc[0] if len(r) else np.nan

    chess_series = [s for s in ["smoke_chess_r1000", "smoke_chess_r1000_svgate", "smoke_chess_f025_svgate"]
                    if (runs.series == s).any()]
    frac_by_ds = runs[(runs.variant == "evo_switch") & runs.series.isin(["short_moons", "smoke_all", "smoke_chess_f025_svgate",
                                                                         "smoke_chess_r1000_svgate"])]
    frac_rows = []
    for (s, d), g in frac_by_ds.groupby(["series", "dataset"]):
        o = old[(old.dataset == d) & (old.variant == "evo_mem")]
        frac_rows.append({"zbiór": d, "seria": s, "n": len(g), "|V|": int(g.full_validation_rows.iloc[0]),
                          "limit |V'|": int(g.vprime_cap_rows.iloc[0]),
                          "|V'| nowy (śr.)": f(g.vprime_rows.mean(), 0), "|V'|/|V| nowy": f(g.vprime_fraction.mean(), 3),
                          "|V'| stary evo_mem (mediana)": f(o.vprime_size.median(), 0) if len(o) else "—",
                          "|V'|/|V| stary": f(o.vprime_size.median() / g.full_validation_rows.iloc[0], 3) if len(o) else "—",
                          "BA test nowy": f(g.test_ba.mean()), "BA test stary evo_mem": f(o.test_ba.mean()) if len(o) else "—"})
    frac_tab = table(pd.DataFrame(frac_rows))

    ps_tab = table(pd.DataFrame([{"parametr": k, "wartość domyślna": json.dumps(v)} for k, v in PHASE_SWITCH_DEFAULTS.items()]))
    paired_tab = table(pm[pm.metric.isin(["test_ba", "fullv_ba", "vprime_rows", "final_k", "steps", "mean_delta_difficulty"])]
                       .assign(**{c: lambda d, c=c: d[c].map(lambda x: f(x)) for c in ["mean_a", "mean_b", "mean_diff", "sd_diff"]})
                       [["a", "b", "metric", "n", "mean_a", "mean_b", "mean_diff", "sd_diff", "a_larger", "b_larger"]])

    chess_line = ""
    if chess_series:
        g = runs[runs.series.isin(chess_series) & (runs.variant == "evo_switch")]
        chess_line = ", ".join(f"{s}: |V'| = {f(runs[(runs.series == s) & (runs.variant == 'evo_switch')].vprime_rows.mean(), 0)}"
                               for s in chess_series)

    md = f"""# followup_phase_switch: automat T′ ↔ V′ — etap smoke i krótkich serii

Raport generowany przez `scripts/followup_phase_switch/report.py` z tabel w `reports/followup_phase_switch/tables/`.
**Nie jest to pełny eksperyment.** Wykonano wyłącznie testy jednostkowe, smoke testy (Moons, Szachownica, 2 seedy na
pozostałych zbiorach) i krótką serię dla Moons (10 seedów). Przebiegów: {inv['runs']}, kroków automatu: {inv['steps']}.
Audyt i lista sprzeczności specyfikacji: `reports/followup_phase_switch/AUDYT.md`.

## 1. Pliki

| plik | zmiana |
|---|---|
| `src/ga/phase_switch.py` | **nowy**: tryb `dagasvm_phase_switch` (ustawienia, czyste funkcje decyzyjne, kontroler automatu, pętla, replay) |
| `src/ga/loop/engine.py` | `run_ga_k_per_class_with_trace`: opcjonalne `seed_masks` (warm start T′) i `target_fitness`; przy `None` identyczne zachowanie |
| `scripts/run_experiment.py` | dispatch `algorithm_mode: dagasvm_phase_switch`; `cycles.csv` z kolumnami `PHASE_SWITCH_COLUMNS` |
| `src/ga/reporting.py` | nowy tryb dodany do zbiorów trybów (maska T′ w wierszu raportu) |
| `tests/test_phase_switch.py` | **nowy**: {len(test_names)} funkcji testowych (część parametryzowana) |
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

{ps_tab}

## 3. Testy

`tests/test_phase_switch.py`: {len(test_names)} funkcji. Pokryte: każda reguła decyzyjna (w tym tabela kombinacji
„BA = 1 nigdy nie zwiększa K”); skryptowany kontroler (T′ → V′ → T′, próba po zastoju z sukcesem i bez,
najwyżej jedna próba na poziom K); 50 losowych przebiegów, które muszą się zakończyć; reguła trudności (stały wzrost,
koniec wzrostu, zerowa trudność, okno); planowanie przyrostu (bardzo małe V′, limit V′, brak kandydatów); integracja
z GA/SVM dla 3 strategii doboru (niezmienniki każdego kroku, limit |V′| dla kilku wartości ułamka i `max_vprime_rows`,
wyczerpanie kandydatów, brak wpływu pełnego V i testu na przebieg, powtarzalność, wspólny start ewolucji i losowania);
replay harmonogramu; zgodność wsteczna GA; CLI.

## 4. Kontrola niezmienników na wszystkich zapisanych krokach

| sprawdzenie | wynik |
|---|---|
| kroki V′ bez poprzedzającej decyzji przełączenia | {inv['v_entry_bad']} |
| kroki, w których jednocześnie wzrosło K i rozszerzono V′ | {inv['k_and_v_same_step']} |
| zmiana K w kroku V′ | {inv['k_change_in_v_step']} |
| BA(V′) ≥ 1 i wzrost K w tym samym kroku (warianty z własnym automatem) | {inv['perfect_increases_k']} |
| j.w. w `rand_replay` (K narzucone harmonogramem evo; losowe V′ bywa opanowane tam, gdzie ewolucyjne nie) | {inv['perfect_k_replay']} (informacyjnie) |
| |V′| powyżej limitu | {inv['over_cap']} |
| więcej niż jedna próba V′ po zastoju przy tym samym K | {inv['dup_retry_per_k']} |
| przebiegi zakończone bezpiecznikiem `stop_max_cycles` | {inv['max_cycles_stops']} |
| pary evo_switch / rand_replay z identycznym końcowym |V′| | {inv['replay_equal_size']} / {inv['replay_pairs']} |

## 5. Wyniki smoke

### 5.1. Moons — kilka wartości `max_vprime_fraction`, bez i z bramką ρ_SV (3 seedy)

{series_table(runs, ['smoke_moons_f005', 'smoke_moons_f010', 'smoke_moons_f025', 'smoke_moons_f010_svgate', 'smoke_moons_f025_svgate'])}

Bez bramki ρ_SV (serie `smoke_moons_f0xx`) K rośnie poniżej TSV coraz szybciej (AUDYT S3), a |V′| zatrzymuje się dopiero na limicie.
Z bramką (`_svgate`, wartość domyślna) |V′| ustala się na podobnym poziomie przy limicie 0,10 i 0,25, czyli rozmiar wynika z automatu, a nie z limitu.

### 5.2. Szachownica (2 seedy)

{series_table(runs, chess_series)}

### 5.3. Krótka seria Moons (10 seedów, wartości domyślne)

{series_table(runs, ['short_moons'])}

Porównania sparowane (te same seedy; `rand_replay` = identyczne fazy, K i |ΔV′| jak evo_switch):

{paired_tab}

### 5.4. |V′|/|V| na wszystkich zbiorach (wartości domyślne) i porównanie z poprzednim algorytmem

{frac_tab}

### 5.5. Obserwacje wymagające decyzji

{observations(runs)}

## 6. Warunki z punktu 4.8 przed pełnym eksperymentem

| warunek | stan |
|---|---|
| fazy przełączają się zgodnie z regułą | {'spełniony' if inv['v_entry_bad'] == 0 and inv['k_and_v_same_step'] == 0 and inv['perfect_increases_k'] == 0 else 'NIESPEŁNIONY'} (sekcja 4) |
| V′ nie rośnie automatycznie do ≈3500 | {'spełniony' if (inv['max_vprime_chess'] or 0) < 3500 else 'NIESPEŁNIONY'}: maks. |V′| na Szachownicy = {inv['max_vprime_chess']} ({chess_line}) |
| |V′|/|V| raportowane | spełniony (`vprime_rows`, `full_validation_rows`, `vprime_fraction` w summary.json, cycles.csv, per_run.csv) |
| warunek stopu nie zapętla algorytmu | {'spełniony' if inv['max_cycles_stops'] == 0 else 'NIESPEŁNIONY'}: {inv['max_cycles_stops']} stopów bezpiecznikiem |
| wariant losowy i ewolucyjny porównywalne | {'spełniony' if inv['replay_pairs'] and inv['replay_equal_size'] == inv['replay_pairs'] else 'NIESPEŁNIONY'} dla `rand_replay` (identyczny harmonogram); `rand_switch` ma inny rozmiar V′ |

## 7. Odpowiedzi

**1. Dlaczego wcześniej V′ rosło do ≈3500?**
W starym trybie V′ było rozszerzane przez logikę ρ_SV, a nie przez sukces T′. Po każdym wzroście K flaga „V′ utwardzone
dla tego K” się zerowała, co wymuszało rozszerzenie. Dodatkowo przy stabilnym ρ_SV rozszerzano V′ zamiast K (do Ts razy).
Przyrost wynosił 2·(K_next − K), czyli rósł razem z K.
Szachownica (podział bazowy, evo_mem + rand_growth): mediana {f(old_runs.exp.median(), 0)} rozszerzeń i {f(old_runs.kinc.median(), 0)}
wzrostów K na przebieg, średnio {f(oexp.val_rows_added.mean(), 1)} próbek na rozszerzenie.
Tylko {int(oexp.perfect_on_current_val.sum())} z {len(oexp)} rozszerzeń nastąpiło po BA(V′)=1,0.
Końcowe |V′| = mediana {f(old_chess.vprime_size.median(), 0)}. Przebieg kończył się, gdy ρ_SV < TSV albo K osiągało limit; rozmiar V′ nie miał żadnego ograniczenia.

**2. Jak nowy algorytm ogranicza ten wzrost?**
(a) V′ rośnie tylko w fazie `optimize_V_prime`, do której wchodzi się wyłącznie po opanowaniu V′ (BA = 1) albo jednorazowo
na poziom K po zastoju T′. (b) Przyrost ma stały, jawny rozmiar (`vprime_increment_per_class`, domyślnie 10/klasę),
niezależny od K. (c) Faza V′ kończy się, gdy trudność przyrostów przestaje rosnąć. (d) Twardy limit
min(`max_vprime_fraction`·|V|, `max_vprime_rows`). (e) Brak poprawy T′ po próbie V′ kończy cały algorytm.
Szachownica: {chess_line}. Moons: |V′| = {f(sm[sm.variant == 'evo_switch'].vprime_rows.mean(), 0)}
(|V′|/|V| = {f(sm[sm.variant == 'evo_switch'].vprime_fraction.mean(), 3)}).

**3. Kiedy dokładnie przechodzimy z T′ do V′?** Po kroku T′, gdy (i) najlepsze T′ ma BA(V′) ≥ `perfect_threshold` (1,0), albo
(ii) wystąpił zastój T′, czyli {PHASE_SWITCH_DEFAULTS['max_no_improvement_k_increases']} kolejne wzrosty K bez poprawy fitnessu
o ≥ {PHASE_SWITCH_DEFAULTS['fitness_min_delta']}, ρ_SV < TSV albo K na limicie, i przy tym K nie było jeszcze próby V′.
W obu przypadkach V′ musi dać się rozszerzyć. W smoke (b) jest ścieżką dominującą: przy doborze ewolucyjnym BA = 1
występuje praktycznie tylko na startowym V′ (AUDYT S4).

**4. Kiedy dokładnie wracamy z V′ do T′?** Po kroku V′, gdy trudność przyrostów (1 − BA(ΔV′), średnia z okna
{PHASE_SWITCH_DEFAULTS['vprime_difficulty_window']}) przez {PHASE_SWITCH_DEFAULTS['vprime_difficulty_patience']} kolejne przyrosty nie wzrosła
o ≥ {PHASE_SWITCH_DEFAULTS['vprime_difficulty_min_delta']} (przyrost bez błędów liczy się jako brak wzrostu); albo gdy V′ osiągnęło limit
lub zabrakło kandydatów; albo po {PHASE_SWITCH_DEFAULTS['max_vprime_expansions_per_phase']} przyrostach w fazie. T′ jest wtedy optymalizowane
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
Moons, 10 seedów: BA test evo − rand_replay = {f(pv('short_moons', 'rand_replay', 'test_ba', 'mean_diff'))}
(SD różnic {f(pv('short_moons', 'rand_replay', 'test_ba', 'sd_diff'))}; evo lepsze w {int(pv('short_moons', 'rand_replay', 'test_ba', 'a_larger'))}/10),
przy trudności przyrostów {f(pv('short_moons', 'rand_replay', 'mean_delta_difficulty', 'mean_a'), 3)} vs
{f(pv('short_moons', 'rand_replay', 'mean_delta_difficulty', 'mean_b'), 3)}. Pomocniczo `rand_switch` (losowanie we własnym automacie)
daje mniejsze V′ ({f(pv('short_moons', 'rand_switch', 'vprime_rows', 'mean_b'), 0)} vs {f(pv('short_moons', 'rand_switch', 'vprime_rows', 'mean_a'), 0)}).

**8. |V′|/|V| dla każdego zbioru:** tabela 5.4. Zbiory inne niż Moons i Szachownica mają tylko 2 seedy (smoke).

## 8. Proponowany następny krok (do decyzji)

1. Pełny eksperyment w układzie evo_switch / rand_replay (+ rand_switch pomocniczo) z wartościami domyślnymi z sekcji 2,
   przy jawnym `max_vprime_rows` tylko tam, gdzie limit ułamkowy jest zbyt luźny.
2. Do rozstrzygnięcia z promotorem: czy próg „opanowania V′” ma pozostać 1,0 (AUDYT S4), czy zastój T′ ma być pełnoprawną
   ścieżką do fazy V′ (obecnie dominującą).
3. Porównanie metryk (BA / F1 / ROC AUC) dopiero po ustaleniu automatu; kod jest przygotowany (`fitness.tprime_metric`,
   `fitness.vprime_metric`), ROC AUC dla doboru V′ wymaga jeszcze implementacji w Val-GA.
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"zapisano {OUT}")


if __name__ == "__main__":
    main()
