"""Raport v2 badania `followup_phase_switch` -> reports/followup_phase_switch/v2/RAPORT_V2.md.

Liczby w tabelach pochodzą z tabel `aggregate.py --version v2` (reports/followup_phase_switch/v2/tables/). Części
opisowe (automat, warunki, role parametrów, uzasadnienie wartości domyślnych) są stałym tekstem tego pliku i opisują
kod `src/ga/phase_switch.py`. NIE nadpisuje raportu v1 (`RAPORT_SMOKE.md`).

    python scripts/followup_phase_switch/report_v2.py --pytest "137 passed in 100s"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import PHASE_SWITCH_DEFAULTS, REPORT_ROOT_V2, TABLES_V2  # noqa: E402

OUT = REPORT_ROOT_V2 / "RAPORT_V2.md"
VARIANT_ORDER = ["evo_switch", "rand_replay", "rand_switch"]
VARIANT_NOTE = {
    "evo_switch": "ewolucyjny dobór ΔV′ (Val-GA)",
    "rand_replay": "losowy dobór ΔV′, harmonogram (fazy, K, \\|ΔV′\\|) odtworzony z evo_switch — porównanie główne",
    "rand_switch": "losowy dobór ΔV′, własny automat (te same reguły; harmonogram i \\|V′\\| mogą się różnić) — pomocniczo",
}


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


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower() == "true"


def results_table(runs: pd.DataFrame, series: list[str]) -> str:
    rows = []
    for s in series:
        for d, gd_all in runs[runs.series == s].groupby("dataset"):
            for v in VARIANT_ORDER:
                gd = gd_all[gd_all.variant == v]
                if gd.empty:
                    continue
                rows.append({
                    "seria": s, "zbiór": d, "wariant": v, "n": len(gd),
                    "\\|V′\\| (śr.)": f(gd.vprime_rows.mean(), 0),
                    "min–max": f"{gd.vprime_rows.min()}–{gd.vprime_rows.max()}",
                    "\\|V\\|": int(gd.full_validation_rows.iloc[0]),
                    "\\|V′\\|/\\|V\\|": f(gd.vprime_fraction.mean(), 3),
                    "rozszerzenia V′": f(gd.vprime_expansions.mean(), 1),
                    "cykle (T′/V′)": f"{f(gd.tprime_steps.mean(), 1)}/{f(gd.vprime_steps.mean(), 1)}",
                    "wejścia do V′ (opanowanie/próba)": f"{f(gd.vprime_phases_mastered.mean(), 1)}/{f(gd.vprime_retries.mean(), 1)}",
                    "K końc.": f(gd.final_k.mean(), 0),
                    "BA na V′ (model rep.)": f(gd.rep_vprime_ba.mean()),
                    "BA na ΔV′ (śr. po krokach V′)": f(gd.delta_ba_mean.mean()),
                    "BA pełne V (obserwacja)": f(gd.fullv_ba.mean()),
                    "BA test (obserwacja)": f"{f(gd.test_ba.mean())} ± {f(gd.test_ba.std(ddof=1) if len(gd) > 1 else np.nan)}",
                    "czas [s]": f(gd.duration_sec.mean(), 0),
                    "powód zakończenia": ", ".join(f"{k.replace('stop_', '')}:{n}" for k, n in gd.stop_decision.value_counts().items()),
                })
    return table(pd.DataFrame(rows))


def cost_table(runs: pd.DataFrame, series: list[str]) -> str:
    rows = []
    for s in series:
        for d, gd_all in runs[runs.series == s].groupby("dataset"):
            for v in VARIANT_ORDER:
                gd = gd_all[gd_all.variant == v]
                if gd.empty or "cost_svm_trainings" not in gd or gd.cost_svm_trainings.isna().all():
                    continue
                rows.append({
                    "seria": s, "zbiór": d, "wariant": v, "n": len(gd),
                    "treningi SVM": f(gd.cost_svm_trainings.mean(), 0),
                    "wywołania predict": f(gd.cost_predict_calls.mean(), 0),
                    "wywołania decision_function": f(gd.cost_decision_function_calls.mean(), 0),
                    "T′: wiersze V′ ocenione (faktycznie)": f(gd.cost_tprime_eval_rows_actual.mean(), 0),
                    "T′: nominalnie (bez wcześn. zatrzymania GA)": f(gd.cost_tprime_eval_rows_nominal.mean(), 0),
                    "ΔV′: wiersze ocenione": f(gd.cost_delta_eval_rows.mean(), 0),
                    "kandydaci (suma po doborach)": f(gd.cost_n_candidates.mean(), 0),
                    "V′: wiersze predict": f(gd.cost_vprime_predict_rows.mean(), 0),
                    "Val-GA nominalnie [wiersze]": f(gd.cost_val_ga_nominal_rows.mean(), 0),
                    "obserwacja [wiersze]": f(gd.cost_observation_rows.mean(), 0),
                })
    return table(pd.DataFrame(rows))


def invariants(steps: pd.DataFrame, runs: pd.DataFrame) -> dict:
    keys = ["series", "dataset", "variant", "seed"]
    st = steps.sort_values(keys + ["cycle"])
    prev = st.groupby(keys).shift(1)
    own = (st.variant != "rand_replay")
    v = st.phase == "optimize_V_prime"
    entry_ok = ((prev.phase == "optimize_T_prime") & (prev.decision == "switch_to_V_prime")) | \
               ((prev.phase == "optimize_V_prime") & (prev.next_phase == "optimize_V_prime"))
    t = st[st.phase == "optimize_T_prime"]
    tv = st[v]
    after_v = (prev.phase == "optimize_V_prime") & (prev.next_phase == "optimize_T_prime")
    k_inc = as_bool(st.k_increased)
    exp = as_bool(st.vprime_expanded)
    perfect_k = as_bool(t.tprime_perfect) & as_bool(t.k_increased)
    retry = t[(t.next_phase == "optimize_V_prime") & ~as_bool(t.tprime_perfect) & (t.variant != "rand_replay")]
    dup_retry = int(retry.groupby(keys + ["k_per_class"]).size().gt(1).sum())
    frac_bad = int((np.abs(st.vprime_fraction - st.vprime_rows_after / st.full_validation_rows) > 1e-12).sum())
    # każdy powrót do T′ po fazie V′ jest krokiem T′, a prev_phase zgadza się z poprzednim wierszem
    prev_phase_bad = int(((st.prev_phase.fillna("") != prev.phase.fillna(""))).sum())
    back_to_t_bad = int((after_v & (st.phase != "optimize_T_prime")).sum())
    # przyrost V′ w kroku V′ >= 1 i <= vprime_increment_per_class × klasy; T′ nie zmienia V′
    grew_in_t = int((t.vprime_rows_after != t.vprime_rows_before).sum())
    stop_no_progress = runs[runs.stop_decision == "stop_no_progress_after_vprime_retry"]
    # po porażce próby V′ T′ dostaje zwiększenia K, o ile K mogło rosnąć
    rc = t[as_bool(t.retry_check) & ~as_bool(t.retry_progress) & own.loc[t.index]]
    rp = runs[runs.variant == "rand_replay"].set_index(["series", "dataset", "seed"])
    ev = runs[runs.variant == "evo_switch"].set_index(["series", "dataset", "seed"])
    common = rp.index.intersection(ev.index)
    return {
        "steps": len(st), "runs": len(runs),
        "v_entry_bad": int((v & ~entry_ok).sum()), "back_to_t_bad": back_to_t_bad, "prev_phase_bad": prev_phase_bad,
        "k_and_v_same_step": int((k_inc & exp).sum()),
        "k_change_in_v_step": int((tv.k_after != tv.k_per_class).sum()),
        "v_changed_in_t_step": grew_in_t,
        "v_step_not_one_expansion": int(((tv.vprime_rows_after - tv.vprime_rows_before) != tv.delta_rows).sum()),
        "perfect_increases_k_own": int((perfect_k & (t.variant != "rand_replay")).sum()),
        "perfect_increases_k_replay": int((perfect_k & (t.variant == "rand_replay")).sum()),
        "over_cap": int((st.vprime_rows_after > st.vprime_cap_rows).sum()),
        "frac_bad": frac_bad, "dup_retry_per_k": dup_retry,
        "max_cycles_stops": int((runs.stop_decision == "stop_max_cycles").sum()),
        "replay_pairs": len(common),
        "replay_equal_size": int((rp.loc[common, "vprime_rows"] == ev.loc[common, "vprime_rows"]).sum()),
        "max_vprime_chess": int(runs[runs.dataset == "chessboard"].vprime_rows.max()) if (runs.dataset == "chessboard").any() else None,
        "max_fraction": float(runs.vprime_fraction.max()),
        "retry_fail_checks": len(rc),
        "retry_fail_then_k_increase": int((rc.decision == "increase_k").sum()),
        "retry_fail_then_stop": int((rc.decision == "stop_no_progress_after_vprime_retry").sum()),
        "stop_no_progress_runs": len(stop_no_progress),
    }


def entry_stats(steps: pd.DataFrame) -> pd.DataFrame:
    t = steps[(steps.phase == "optimize_T_prime") & (steps.variant != "rand_replay")]
    sw = t[t.next_phase == "optimize_V_prime"].copy()
    sw["mastered"] = as_bool(sw.tprime_perfect)
    rows = []
    for (d, v), g in sw.groupby(["dataset", "variant"]):
        rows.append({"zbiór": d, "wariant": v, "wejścia do V′": len(g), "z opanowania (BA(V′)=1)": int(g.mastered.sum()),
                     "z zastoju T′ (próba)": int((~g.mastered).sum()),
                     "rozmiary V′ przy opanowaniu": ", ".join(str(x) for x in sorted(set(g[g.mastered].vprime_rows_before.astype(int)))[:8]) or "—"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytest", default="(nie podano)")
    ap.add_argument("--series", nargs="*", default=None)
    args = ap.parse_args()
    runs = pd.read_csv(TABLES_V2 / "per_run.csv")
    steps = pd.read_csv(TABLES_V2 / "per_step.csv", low_memory=False)
    paired = pd.read_csv(TABLES_V2 / "paired.csv")
    series = args.series or sorted(runs.series.unique())
    runs, steps = runs[runs.series.isin(series)], steps[steps.series.isin(series)]
    inv = invariants(steps, runs)
    ent = entry_stats(steps)
    base_ba = runs.groupby("dataset").baseline_full_svm_test_ba.first()

    pm = paired[paired.series.isin(series) & (paired.a == "evo_switch") & (paired.b == "rand_replay")
                & paired.metric.isin(["test_ba", "fullv_ba", "vprime_rows", "steps", "final_k", "mean_delta_difficulty"])].copy()
    for c in ("mean_a", "mean_b", "mean_diff", "sd_diff"):
        pm[c] = pm[c].map(lambda x: f(x))
    paired_tab = table(pm[["series", "dataset", "metric", "n", "mean_a", "mean_b", "mean_diff", "sd_diff", "a_larger", "b_larger"]]
                       .rename(columns={"mean_a": "evo_switch", "mean_b": "rand_replay", "a_larger": "evo większe", "b_larger": "rand większe"}))
    ps_tab = table(pd.DataFrame([{"parametr": k, "wartość": json.dumps(v)} for k, v in PHASE_SWITCH_DEFAULTS.items()]))
    chess = [s for s in series if "chess" in s]
    moons = [s for s in series if "moons" in s]

    def gate(ok: bool) -> str:
        return "spełniony" if ok else "NIESPEŁNIONY"

    cond = [
        ("fazy przełączają się zgodnie z regułą (wejścia do V′ tylko po decyzji przełączenia, powrót do T′ po fazie V′)",
         gate(inv["v_entry_bad"] == 0 and inv["back_to_t_bad"] == 0 and inv["prev_phase_bad"] == 0
              and inv["k_and_v_same_step"] == 0 and inv["k_change_in_v_step"] == 0 and inv["v_changed_in_t_step"] == 0
              and inv["v_step_not_one_expansion"] == 0)),
        ("BA(V′)=1,0 nie kończy całego algorytmu (K nie rośnie, przełączenie do V′; stop tylko gdy V′ nie da się rozszerzyć)",
         gate(inv["perfect_increases_k_own"] == 0)),
        ("V′ nie rośnie mechanicznie do ≈3500", gate((inv["max_vprime_chess"] or 0) < 3500) + f" (maks. \\|V′\\| na Szachownicy = {inv['max_vprime_chess']}; maks. \\|V′\\|/\\|V\\| = {f(inv['max_fraction'], 3)})"),
        ("raportowane jest \\|V′\\|/\\|V\\|", gate(inv["frac_bad"] == 0) + " (`vprime_fraction` w cycles.csv, summary.json, per_run.csv)"),
        ("wariant losowy porównywalny przy tym samym rozmiarze V′", gate(inv["replay_pairs"] > 0 and inv["replay_equal_size"] == inv["replay_pairs"]) + f" ({inv['replay_equal_size']}/{inv['replay_pairs']} par `rand_replay` ma identyczny końcowy \\|V′\\|)"),
        ("warunek stopu nie zapętla algorytmu", gate(inv["max_cycles_stops"] == 0 and inv["over_cap"] == 0 and inv["dup_retry_per_k"] == 0) + f" ({inv['max_cycles_stops']} stopów bezpiecznikiem `max_cycles`, {inv['dup_retry_per_k']} powtórzonych prób na poziom K)"),
    ]
    cond_tab = table(pd.DataFrame(cond, columns=["warunek przed pełnym badaniem", "stan"]))

    md = f"""# followup_phase_switch v2: naprzemienna optymalizacja T′ ↔ V′ — implementacja i smoke

Raport generowany przez `scripts/followup_phase_switch/report_v2.py` z tabel w `reports/followup_phase_switch/v2/tables/`.
**To nie jest pełne badanie.** Wykonano testy jednostkowe/integracyjne i krótkie smoke testy (Moons, Szachownica).
Liczba przebiegów: {inv['runs']}, cykli (kroków automatu): {inv['steps']}. Wyniki v1 (`RAPORT_SMOKE.md`, tabele v1, serie bez prefiksu `v2_`)
i stary tryb `dagasvm_phase` (+ raport `followup_stop_rule`) są nietknięte. Audyt: `AUDYT_v2.md`.

Trudność przyrostu ΔV′ i jakość modelu są **osobnymi właściwościami**; raport nie wnioskuje z jednej o drugiej (trudność ΔV′ jest
sygnałem sterującym fazą V′, a jakość modelu oceniają BA na V′ i — wyłącznie jako obserwacja — pełne V i test).

## 1. Zmienione i nowe pliki

| plik | zmiana |
|---|---|
| `src/ga/phase_switch.py` | tryb `dagasvm_phase_switch`: ustawienia, czyste funkcje decyzyjne, kontroler fazowy, pętla, replay, `CountingModel` (koszt) — poprawiony względem v1 |
| `scripts/run_experiment.py` | `objective_metric_name` = `fitness.tprime_metric` w tym trybie (dispatch trybu istniał w v1) |
| `pyproject.toml` | `[tool.pytest.ini_options] pythonpath` — `./.venv/bin/pytest -q` nie przerywa się już na 2 plikach (brak katalogu głównego w `sys.path`) |
| `tests/test_phase_switch.py` | testy trybu (pkt 14 specyfikacji: `test_req01_*` … `test_req14_*`, koszt, ustawienia) |
| `scripts/followup_phase_switch/{{ps_common,aggregate,make_configs,run_queue,report_v2}}.py` | wersjonowanie v1/v2; v2 zapisuje do osobnych ścieżek |
| `configs/followup_phase_switch/v2/` | szablony i `study_spec.yaml` v2 |
| `artifacts/followup_phase_switch/v2_*` | przebiegi smoke v2 |
| `reports/followup_phase_switch/AUDYT_v2.md`, `v2/` | audyt, tabele, ten raport |

Niezmienione semantycznie: `src/ga/phase_engine.py`, `src/ga/vprime.py`, `src/ga/loop/engine.py`, `src/ga/models/svm.py` (tryb `dagasvm_phase`).

## 2. Maszyna fazowa

Cykl = jeden krok automatu: **jedno** uruchomienie GA dla T′ (przy stałym K, na pełnym skumulowanym V′) **albo jedno** rozszerzenie V′.
W jednym cyklu K rośnie najwyżej raz, V′ jest rozszerzane najwyżej raz, nigdy oba naraz.

```mermaid
stateDiagram-v2
    [*] --> optimize_T_prime
    optimize_T_prime --> optimize_T_prime: BA(V′) < 1, brak zastoju<br/>K + (raz)
    optimize_T_prime --> optimize_V_prime: BA(V′) = 1 (opanowanie)<br/>albo zastój T′ (jedna próba na poziom K)
    optimize_V_prime --> optimize_V_prime: trudność ΔV′ nadal rośnie
    optimize_V_prime --> optimize_T_prime: plateau trudności / limit V′ / brak kandydatów
    optimize_T_prime --> stop: V′ nierozszerzalne i (BA=1 albo zastój)<br/>lub brak postępu po próbie V′
    stop --> [*]
```

Reguła „minimalny rozmiar V′ + obowiązkowe ponowne sprawdzenie” (odrzucona przez promotora) **nie występuje** w tym trybie: parametry
`daga.min_vprime_per_class` i `daga.perfect_recheck_*` nie są czytane (test `test_rejected_minimum_size_plus_recheck_rule_is_not_used_*`).
Długość fazy V′ wynika wyłącznie z trudności ΔV′, a rozmiar V′ ograniczają limity górne — nie ma żadnego dolnego progu rozmiaru.

### Warunki przejść

| przejście | warunek (wyłącznie V′, ΔV′, T′, K, pula kandydatów; nigdy pełne V ani test) |
|---|---|
| **T′ → V′** (opanowanie) | BA(V′) ≥ `perfect_threshold` (1,0) i V′ można rozszerzyć. K bez zmian. **BA(V′)=1 nie kończy algorytmu.** |
| **T′ → V′** (próba po zastoju) | zastój T′: `max_no_improvement_k_increases` (3) kolejnych zwiększeń K bez poprawy najlepszego fitnessu o ≥ `fitness_min_delta`, albo K nie może rosnąć (limit klasy), albo ρ_SV < TSV (bramka `k_growth_requires_sv_signal`); jednorazowo na poziom K; V′ rozszerzalne. Zapamiętywany jest najlepszy fitness sprzed próby. |
| **K → K+** (T′ → T′) | BA(V′) < 1, brak zastoju, ρ_SV ≥ TSV (z bramką), K może rosnąć. Wielkość kroku: dotychczasowy wzór DAGA na ρ_SV. **Najwyżej raz na cykl; nigdy przy BA(V′)=1; nigdy w fazie V′.** |
| **V′ → V′** | trudność ΔV′ (średnia z okna `vprime_difficulty_window`) wzrosła o ≥ `vprime_difficulty_min_delta` |
| **V′ → T′** | `vprime_difficulty_patience` (2) kolejnych przyrostów bez wzrostu trudności **albo** limit \\|V′\\| **albo** brak kandydatów **albo** limit rozszerzeń (globalny / w fazie). T′ optymalizowane jest wtedy na pełnym, skumulowanym V′. |
| **po próbie V′** | T′ nie osiągnęło poziomu odniesienia ⇒ K rośnie (raz na cykl) i T′ jest optymalizowane ponownie; dopiero `max_no_improvement_k_increases` takich zwiększeń bez osiągnięcia poziomu, albo K nie może rosnąć ⇒ **stop**. Poziom osiągnięty ⇒ zwykła praca (licznik zastoju od zera). |
| **stop** | patrz niżej |

Trudność przyrostu domyślnie = odsetek błędów modelu T′ (ustalonego na czas fazy V′) na ΔV′ (`vprime_difficulty_metric: error_rate`).
Zapisywane są też: BA(ΔV′), liczba próbek ΔV′ w każdej klasie, zmiana trudności względem poprzedniego przyrostu i koszt.

### Warunek zakończenia całego algorytmu

| kod | warunek |
|---|---|
| `stop_perfect_on_max_vprime` | BA(V′)=1, a V′ nie da się już rozszerzyć (limit rozmiaru, limit liczby rozszerzeń albo brak kandydatów) |
| `stop_no_progress_after_vprime_retry` | po dodatkowej próbie V′ T′ nie osiągnęło poziomu odniesienia także po zwiększeniach K (albo K nie może rosnąć) |
| `stop_tprime_no_progress`, `stop_k_limit`, `stop_no_k_growth_signal` | zastój T′, a próba V′ przy tym K już była lub V′ nierozszerzalne |
| `stop_max_cycles` | bezpiecznik (`max_cycles`) |

Algorytm nie zapętla się: każda faza V′ dodaje ≥ 1 próbkę na klasę (rozmiar ograniczony), K rośnie monotonicznie do limitu klasy,
próba V′ przypada najwyżej raz na poziom K, a nieudana próba kończy algorytm. Dowód empiryczny: tabela 5, test `test_req09_*`.

### Role parametrów

| parametr | rola w v2 |
|---|---|
| **K** (`k_per_class`) | rozmiar T′ (próbek na klasę). Steruje **wyłącznie** rozmiarem T′; nie wpływa na przyrost V′. |
| **ρ_SV** (udział wektorów podpierających w T′) | (i) sygnał diagnostyczny; (ii) wielkość kroku K (wzór DAGA); (iii) przy `k_growth_requires_sv_signal: true` pomocniczy sygnał braku poprawy: ρ_SV < TSV oznacza, że K nie rośnie, i daje zastój T′ (próba V′). Nie powoduje rozszerzenia V′ sam z siebie. |
| **TSV** | próg ρ_SV (domyślnie 0,25). **Nie może samodzielnie powodować wzrostu V′** — V′ rośnie tylko w fazie V′. |
| **Ts** (`daga.Ts`) | cierpliwość wewnątrz GA dla T′ (wczesne zatrzymanie po `Ts` generacjach bez poprawy). Nie steruje fazami ani rozmiarem V′ (w trybie `dagasvm_phase` liczyło też cykle „stabilnego ρ_SV”). |
| `max_no_improvement_k_increases` | liczba kolejnych zwiększeń K bez poprawy fitnessu = zastój T′ |
| `vprime_increment_per_class` | rozmiar przyrostu ΔV′ (niezależny od K) |

## 3. Kontrola rozmiaru V′

Rozmiar V′ zależy tylko od faz i jawnych limitów; nie rośnie z K ani przez sam sygnał ρ_SV. Limit efektywny:
`min(⌊max_vprime_fraction · |V|⌋, max_vprime_rows, |V|)`, plus globalny `max_vprime_expansions` i `max_vprime_expansions_per_phase`.
Rozszerzanie zatrzymuje: plateau trudności ΔV′, limit rozmiaru, limit rozszerzeń albo brak kandydatów (zapisywane w `vprime_limit_reason`).
Test nigdy nie steruje limitem. Dla Szachownicy porównanie przy ≈1000 próbek: `max_vprime_rows: 1000` (serie `v2_smoke_chess_r1000*`).

Wartości domyślne (sekcja `phase_switch`; `fitness.tprime_metric` i `fitness.vprime_metric` = `balanced_accuracy`):

{ps_tab}

Uzasadnienie (to wartości robocze, **nie strojone**; wrażliwość nie była badana):

| parametr | uzasadnienie |
|---|---|
| `vprime_increment_per_class: 10` | równe `k_start` i V′ startowemu (10/klasę). ΔV′ ma ≥ 20 próbek (2 klasy), więc odsetek błędów ma błąd standardowy rzędu 0,09; mniejszy przyrost zaszumiałby ocenę trudności. Względem \\|V\\| to 0,07–0,4 % na rozszerzenie. |
| `max_vprime_fraction: 0.25` | górny bezpiecznik, nie cel. Jeden ułamek nie daje porównywalnego \\|V′\\| (\\|V\\| od 719 do 27 853), stąd obok `max_vprime_rows`. W smoke nie jest osiągany na Szachownicy ani Moons. |
| `max_vprime_rows: null` | jawny limit tylko tam, gdzie potrzebny (Szachownica: 1000). |
| `max_vprime_expansions: null` | globalny bezpiecznik; domyślnie wyłączony, bo limity rozmiaru i faz wystarczają. |
| `vprime_difficulty_min_delta: 0.01` | wzrost trudności o < 1 p.p. (średnia z okna) traktowany jako brak wzrostu; jedna próbka z 20 to 5 p.p., więc 0,01 jest poniżej rozdzielczości pomiaru. |
| `vprime_difficulty_patience: 2`, `window: 2` | dwa kolejne przyrosty bez wzrostu kończą fazę; okno 2 wygładza fluktuacje Val-GA (stochastyczny). |
| `max_vprime_expansions_per_phase: 10` | bezpiecznik pojedynczej fazy V′. |
| `max_no_improvement_k_increases: 3`, `fitness_min_delta: 0.0005` | wartości ze specyfikacji; 0,0005 = `phase_min_delta` GA, czyli taki sam próg „poprawy” jak wewnątrz GA. |
| `retry_resolution: after_k_increases`, `retry_reference: same_vprime` | patrz `AUDYT_v2.md` §3: T′ dostaje szansę po zwiększeniu K; porównanie na tym samym, rozszerzonym V′ (wariant dosłowny: `literal`). |
| `k_growth_requires_sv_signal: true` | istniejąca polityka K (DAGA). **Do decyzji promotora** (sekcja 9). |
| `warm_start_tprime: true` | ponowna optymalizacja T′ przy tym samym K startuje od poprzedniego najlepszego T′ (bez tego „poprawa względem odniesienia” nie ma sensu). |
| `max_cycles: 300` | bezpiecznik; w smoke maks. liczba cykli = {int(steps.groupby(['series', 'dataset', 'variant', 'seed']).cycle.max().max())}. |

## 4. Fitness i BA

`fitness.tprime_metric` i `fitness.vprime_metric` (domyślnie BA) są konfigurowalne; przygotowane pod F1/ROC AUC (nie uruchamiane w tym etapie).
Pełny inwentarz miejsc użycia BA (T′, wybór V′, przełączanie, stop, raportowanie), ciche podstawienia BA w kodzie i ocena metodologiczna: `AUDYT_v2.md` §2.
Test nie jest nigdzie fitnessem ani warunkiem sterującym (testy `test_req10_*`: śmieciowy obserwator pełnego V i odwrócone etykiety testu nie zmieniają trajektorii).

## 5. Testy

`./.venv/bin/pytest -q` → **{args.pytest}**. Wymagania 1–14 ze specyfikacji mają osobne testy (`test_req01_*` … `test_req14_*`; mapa w `AUDYT_v2.md` §4).
Zdolność testów do wykrywania błędów sprawdzono mutacjami: wstrzyknięcie błędu (T′ widzi tylko ostatnie próbki V′, brak limitu rozmiaru, brak limitu
rozszerzeń, BA=1 zwiększa K, pominięty koszt GA) powoduje niepowodzenie odpowiedniego testu. Przebieg v1 odtwarza się bit w bit na kodzie v2 z flagami
`retry_resolution=first_step` i `vprime_difficulty_metric=one_minus_metric` (45 kroków, 60 kolumn).

## 6. Wyniki smoke

Kolumny: \\|V′\\|, \\|V\\| i \\|V′\\|/\\|V\\| — końcowe; „rozszerzenia V′” — liczba kroków V′; „wejścia do V′” — ile razy faza V′ zaczęła się po opanowaniu V′ (BA=1) /
po zastoju T′ (próba); „BA na V′” — model reprezentatywny (najlepszy na końcowym V′); „BA na ΔV′” — średnia z kroków V′ (model ustalony na czas fazy);
BA test — wyłącznie obserwacja. Pełny SVM (cały zbiór treningowy) ma BA test: {', '.join(f'{d} {f(v)}' for d, v in base_ba.items())}.

### 6.1. Moons

{results_table(runs, moons)}

### 6.2. Szachownica (limit \\|V′\\| = 1000; z bramką ρ_SV = wartość domyślna, `_nogate` = bez bramki)

{results_table(runs, chess)}

Porównanie sparowane evo_switch − rand_replay (te same seedy i identyczny harmonogram faz, K i \\|ΔV′\\|; to smoke z n = 2–3, bez wniosków statystycznych):

{paired_tab}

### 6.3. Skąd pochodzą wejścia do fazy V′ (warianty z własnym automatem)

{table(ent)}

### 6.4. Koszt obliczeń

Wywołania modelu wykonane przez algorytm (obserwacja pełnego V i testu w ostatniej kolumnie, osobno). „Nominalnie” = populacja × (generacje+1) × \\|V′\\|
bez wcześniejszego zatrzymania GA; „faktycznie” = rzeczywiste ewaluacje GA × \\|V′\\|. Ewaluacje GA dla T′ to po jednym `predict` na osobnika (zgodność z
rzeczywistą liczbą wywołań `SVC.predict` sprawdza test `test_cost_counters_equal_the_real_number_of_model_calls`). `svm.probability=false` we wszystkich seriach
(przy `true` dodatkowe przejścia `predict_proba` nie są liczone).

{cost_table(runs, series)}

## 7. Kontrola niezmienników na wszystkich zapisanych cyklach

| sprawdzenie | wynik |
|---|---|
| cykle V′ bez poprzedzającej decyzji przełączenia (lub kontynuacji fazy V′) | {inv['v_entry_bad']} |
| faza V′ zakończona decyzją „do T′”, a następny cykl nie jest T′ | {inv['back_to_t_bad']} |
| `prev_phase` niezgodne z poprzednim cyklem | {inv['prev_phase_bad']} |
| cykle, w których jednocześnie wzrosło K i rozszerzono V′ | {inv['k_and_v_same_step']} |
| zmiana K w cyklu V′ / zmiana V′ w cyklu T′ | {inv['k_change_in_v_step']} / {inv['v_changed_in_t_step']} |
| cykl V′, w którym V′ urosło inaczej niż o jedno ΔV′ | {inv['v_step_not_one_expansion']} |
| BA(V′)=1 i wzrost K w tym samym cyklu (warianty z własnym automatem) | {inv['perfect_increases_k_own']} |
| j.w. w `rand_replay` (K narzucone harmonogramem evo; losowe V′ bywa opanowane tam, gdzie ewolucyjne nie) | {inv['perfect_increases_k_replay']} (informacyjnie) |
| \\|V′\\| powyżej limitu / `vprime_fraction` ≠ \\|V′\\|/\\|V\\| | {inv['over_cap']} / {inv['frac_bad']} |
| więcej niż jedna próba V′ po zastoju przy tym samym K | {inv['dup_retry_per_k']} |
| kontrole po próbie V′ bez osiągnięcia poziomu: dalej zwiększenie K / stop | {inv['retry_fail_then_k_increase']} / {inv['retry_fail_then_stop']} (z {inv['retry_fail_checks']}) |
| przebiegi zakończone bezpiecznikiem `stop_max_cycles` | {inv['max_cycles_stops']} |

## 8. Warunki przed pełnym eksperymentem

{cond_tab}

## 9. Znane ograniczenia i pytania do promotora

Szczegóły: `AUDYT_v2.md` §5.

1. **Próg „opanowania V′”.** Przy doborze ewolucyjnym BA(V′)=1 zdarza się tylko przy małych V′ (tabela 6.3). Dalsze fazy V′ wynikają z zastoju T′. Czy próg ma być < 1?
2. **Bramka ρ_SV.** Z bramką Szachownica kończy przy K ≈ 900 i BA test poniżej pełnego SVM; bez bramki K rośnie do tysięcy (koszt) i jakość zbliża się do pełnego SVM
   (tabela 6.2). To kompromis rozmiar T′ ↔ jakość ↔ czas, niezależny od reguł V′. Czy zachować bramkę, czy zmienić wzór na ΔK (dziś ma wartość bezwzględną |ρ_SV − TSV|)?
3. **„Co najmniej zapamiętany fitness”.** Domyślnie porównanie na tym samym, rozszerzonym V′; wariant dosłowny zaimplementowany (`retry_reference: literal`), niewybrany.
4. **Model reprezentatywny:** `best_on_final_vprime` vs `last_tprime_step`.
5. **Mała liczba seedów** w smoke (Moons 3, Szachownica 2): brak wniosków o różnicy ewolucja ↔ losowanie.
6. **Skład klasowy ΔV′.** `vprime_increment_per_class` określa łączny rozmiar przyrostu (× liczba klas); skład klasowy wybiera strategia (ewolucja i losowanie
   nie wymuszają równowagi). W smoke udział największej klasy w ΔV′: ewolucja średnio 0,62–0,66 (maks. 0,90), losowanie ≈ 0,58; nigdy przyrost z jednej klasy.
   Odsetek błędów (domyślna miara trudności) zależy od składu przyrostu; BA(ΔV′) i `delta_per_class` są zapisywane obok.
7. **Koszt doboru ewolucyjnego przy dużej puli.** Val-GA ocenia całą pulę kandydatów jednym `predict` na rozszerzenie (zbuforowane predykcje), a stare próbki V′ nie są oceniane.
   Na Moons to taniej niż nominalna ocena podzbiorów (96 tys. wobec 265 tys. wierszy), na Szachownicy (pula ≈ 27 tys.) drożej (≈ 690 tys. wobec 315 tys.). Predykcje „leniwe”
   (tylko wiersze faktycznie użyte przez osobniki) obniżyłyby koszt; nie zaimplementowano, bo wymaga zmiany `vprime.py` współdzielonego ze starym trybem.
8. Przy `svm.probability=true` liczniki kosztu nie obejmują `predict_proba`. Metryki inne niż BA wymagają jeszcze doboru progu „opanowania” (AUDYT_v2 §2).
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding="utf-8")
    print(f"zapisano {OUT} ({len(md)} znaków)")


if __name__ == "__main__":
    main()
