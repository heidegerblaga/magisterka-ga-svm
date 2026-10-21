"""Testy rozszerzonej reguły stopu BA(V')=1,0: minimalny rozmiar V', obowiązkowe ponowne sprawdzenie,
zgodność z regułą historyczną przy wartościach domyślnych, brak użycia pełnego V do sterowania,
maks. jeden wzrost K na cykl oraz wspólny budżet puli kandydatów."""
import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.models.svm import SVMConfig
from ga.phase_engine import (
    DECISION_EXPAND,
    DECISION_INCREASE_K,
    STOP_CATEGORY,
    CycleDecision,
    decide_cycle,
    parse_phase_settings,
    perfect_stop_gate,
    run_phase_experiment,
)
from ga.utils.artifacts import ensure_artifact_layout
from ga.utils.logger import get_logger
from ga.vprime import FullValidationObserver, VPrimeConfig, VPrimeSet, budget_pool_ids, expand_vprime


# --------------------------------------------------------------------------------------------
# 1. funkcja decyzyjna: kopia reguły historycznej jako wzorzec (tabela prawdy)
# --------------------------------------------------------------------------------------------
def _legacy_decide(
    *,
    perfect_on_current_val,
    has_growth_signal,
    expansion_possible,
    vprime_mode,
    validation_hardened_for_current_k,
    sv_ratio_stable_since_hardening,
    can_increase_k,
    forced_growth_after_patience=False,
):
    """Dosłowna kopia decide_cycle sprzed zmiany (tylko pary decyzja/powód)."""
    if perfect_on_current_val:
        if expansion_possible and not validation_hardened_for_current_k:
            return DECISION_EXPAND, "T' osiąga próg 100% na V'; K nie rośnie automatycznie — najpierw utrudniam V'"
        return (
            "stop_perfect_on_current_val",
            "T' osiąga próg na V' mimo wcześniejszego utrudnienia V' (lub V' nie da się rozszerzyć); K bez zmian",
        )
    if not has_growth_signal:
        return "stop_no_growth_signal", None
    if vprime_mode == "fixed":
        return (DECISION_INCREASE_K, None) if can_increase_k else ("stop_no_k_growth", None)
    if not validation_hardened_for_current_k:
        return (DECISION_EXPAND, None) if expansion_possible else ("stop_no_validation_candidates", None)
    if sv_ratio_stable_since_hardening:
        return (DECISION_EXPAND, None) if expansion_possible else ("stop_no_validation_candidates", None)
    return (DECISION_INCREASE_K, None) if can_increase_k else ("stop_no_k_growth", None)


def _all_flag_combinations():
    names = [
        "perfect_on_current_val",
        "has_growth_signal",
        "expansion_possible",
        "validation_hardened_for_current_k",
        "sv_ratio_stable_since_hardening",
        "can_increase_k",
        "forced_growth_after_patience",
    ]
    for values in itertools.product([False, True], repeat=len(names)):
        for mode in ("growing", "fixed"):
            yield dict(zip(names, values), vprime_mode=mode)


def test_defaults_reproduce_the_historical_decision_for_every_input():
    n = 0
    for kwargs in _all_flag_combinations():
        new = decide_cycle(**kwargs)  # parametry nowej reguły pominięte = wartości domyślne
        old_decision, old_reason = _legacy_decide(**kwargs)
        assert new.decision == old_decision, kwargs
        if old_reason is not None:
            assert new.reason == old_reason
        assert new.blocked_by == "" and new.gate_pending == ""
        n += 1
    assert n == 256
    # jawnie podane wartości „neutralne” też nie zmieniają decyzji
    for kwargs in _all_flag_combinations():
        a = decide_cycle(**kwargs)
        b = decide_cycle(**kwargs, min_vprime_size_reached=True, perfect_rechecks_done=0, perfect_rechecks_required=0)
        assert (a.decision, a.reason) == (b.decision, b.reason)


def _perfect(**overrides):
    base = dict(
        perfect_on_current_val=True,
        has_growth_signal=True,
        expansion_possible=True,
        vprime_mode="growing",
        validation_hardened_for_current_k=True,  # reguła historyczna zatrzymałaby algorytm
        sv_ratio_stable_since_hardening=False,
        can_increase_k=True,
    )
    base.update(overrides)
    return decide_cycle(**base)


def test_perfect_ba_cannot_stop_when_vprime_is_below_the_minimum_size():
    d = _perfect(min_vprime_size_reached=False)
    assert d.decision == DECISION_EXPAND and not d.stops and not d.increases_k
    assert d.blocked_by == "min_size" and d.gate_pending == "min_size"
    # także wtedy, gdy wymaganych ponownych sprawdzeń jest 0 i V' było już „utwardzone”
    d = _perfect(min_vprime_size_reached=False, perfect_rechecks_required=0)
    assert d.decision == DECISION_EXPAND


def test_mandatory_expansion_after_reaching_minimum_size():
    # minimum osiągnięte, ale 0 z 1 wymaganych ponownych sprawdzeń => rozszerzenie zamiast stopu
    d = _perfect(min_vprime_size_reached=True, perfect_rechecks_done=0, perfect_rechecks_required=1)
    assert d.decision == DECISION_EXPAND and d.blocked_by == "recheck"
    # dwa brakujące warunki naraz
    d = _perfect(min_vprime_size_reached=False, perfect_rechecks_done=0, perfect_rechecks_required=1)
    assert d.decision == DECISION_EXPAND and d.blocked_by == "min_size+recheck"
    # gdy V' nie było jeszcze utwardzone, reguła historyczna i tak rozszerza — wtedy to nie jest „zablokowany stop”
    d = _perfect(min_vprime_size_reached=True, perfect_rechecks_required=1, validation_hardened_for_current_k=False)
    assert d.decision == DECISION_EXPAND and d.blocked_by == "" and d.gate_pending == "recheck"


def test_stop_only_after_the_required_recheck():
    stop = _perfect(min_vprime_size_reached=True, perfect_rechecks_done=1, perfect_rechecks_required=1)
    assert stop.decision == "stop_perfect_on_current_val" and stop.stops
    assert stop.blocked_by == "" and stop.gate_pending == ""
    # dwa wymagane sprawdzenia: po jednym jeszcze nie wolno
    assert _perfect(perfect_rechecks_done=1, perfect_rechecks_required=2).decision == DECISION_EXPAND
    assert _perfect(perfect_rechecks_done=2, perfect_rechecks_required=2).decision == "stop_perfect_on_current_val"


def test_blocked_perfect_with_no_candidates_stops_without_growing_k():
    for hardened in (False, True):
        d = _perfect(
            expansion_possible=False,
            min_vprime_size_reached=False,
            validation_hardened_for_current_k=hardened,
        )
        assert d.decision == "stop_no_validation_candidates" and d.stops and not d.increases_k
        assert d.blocked_by == "min_size"
    d = _perfect(expansion_possible=False, perfect_rechecks_required=1)
    assert d.decision == "stop_no_validation_candidates" and d.blocked_by == "recheck"
    # brak kandydatów, ale warunki spełnione => zwykły stop BA=1,0 (jak dotychczas)
    assert _perfect(expansion_possible=False).decision == "stop_perfect_on_current_val"


def test_perfect_ba_never_increases_k_for_any_gate_combination():
    for kwargs in _all_flag_combinations():
        if not kwargs["perfect_on_current_val"]:
            continue
        for min_ok, done, req in itertools.product([False, True], [0, 1, 2], [0, 1, 2]):
            d = decide_cycle(
                **kwargs, min_vprime_size_reached=min_ok, perfect_rechecks_done=done, perfect_rechecks_required=req
            )
            assert not d.increases_k


def test_fixed_vprime_is_not_subject_to_the_gates():
    # V' stałe nie może zostać rozszerzone, więc warunki nie mają zastosowania (inaczej pętla bez końca)
    assert perfect_stop_gate(
        vprime_mode="fixed", min_vprime_size_reached=False, perfect_rechecks_done=0, perfect_rechecks_required=3
    ) == ""
    d = _perfect(vprime_mode="fixed", expansion_possible=False, min_vprime_size_reached=False, perfect_rechecks_required=1)
    assert d.decision == "stop_perfect_on_current_val"


def test_stop_category_covers_every_stop_decision_code():
    from ga.phase_engine import STOP_DECISIONS

    for code in STOP_DECISIONS | {"stop_k_class_cap", "stop_no_new_vprime_elements"}:
        assert code in STOP_CATEGORY
    assert set(STOP_CATEGORY.values()) == {
        "perfect_ba", "no_k_growth_signal", "k_limit", "no_candidates", "max_cycles",
    }


def test_new_parameters_default_to_the_historical_behaviour_and_are_validated():
    st = parse_phase_settings({"daga": {"k_start": 4}})
    assert st.min_vprime_per_class == 0 and st.perfect_recheck_expansions == 0
    assert st.perfect_recheck_min_add_per_class == 0 and st.vprime.pool_budget_factor is None
    st = parse_phase_settings(
        {"daga": {"min_vprime_per_class": 50, "perfect_recheck_expansions": 1, "perfect_recheck_min_add_per_class": 25}}
    )
    assert (st.min_vprime_per_class, st.perfect_recheck_expansions, st.perfect_recheck_min_add_per_class) == (50, 1, 25)
    with pytest.raises(ValueError):
        parse_phase_settings({"daga": {"min_vprime_per_class": -1}})
    with pytest.raises(ValueError):
        parse_phase_settings({"vprime": {"pool_budget_factor": 0.5}})


# --------------------------------------------------------------------------------------------
# 2. testy integracyjne pętli fazowej
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def easy_data():
    """Łatwy zbiór (mały szum): małe V' dostaje BA = 1,0 już przy K=4 — idealny do testu przedwczesnego stopu."""
    X, y = make_moons(n_samples=3000, noise=0.08, random_state=5)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.4, stratify=y, random_state=5)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=0.7, stratify=y_rest, random_state=6)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _cfg(seed=42, daga=None, **vprime):
    base_daga = {"k_start": 4, "TSV": 0.25, "Ts": 2, "phase_min_delta": 0.0005, "max_cycles": 60}
    base_daga.update(daga or {})
    return {
        "seed": seed,
        "fitness": {"metric": "balanced_accuracy"},
        "ga": {
            "population_size": 6,
            "generations": 4,
            "tournament_k": 3,
            "crossover_rate": 0.8,
            "mutation_rate": 0.05,
            "elitism": 2,
            "early_stopping": {"patience": 2, "min_delta": 0.0001},
        },
        "daga": base_daga,
        "val_ga": {"population_size": 6, "generations": 3, "mutation_rate": 0.1, "elitism": 2},
        "vprime": {"selection": "random", "memory": True, "mode": "growing", **vprime},
        "observation": {"full_validation": True, "test": True},
    }


def _run(tmp_path, data, cfg, tag, observer_factory=None):
    X_tr, y_tr, X_va, y_va, X_te, y_te = data
    layout = ensure_artifact_layout(tmp_path / tag, experiment_name=tag)
    svm = SVMConfig(kernel="rbf", C=10.0, gamma=1.0, probability=False, standardize=False)
    observer = None if observer_factory is None else observer_factory(X_va, y_va)
    out = run_phase_experiment(X_tr, y_tr, X_va, y_va, X_te, y_te, svm, cfg, get_logger(), layout, observer=observer)
    return out, Path(layout["checkpoints"])


def _traj(out):
    return [
        (r["cycle"], r["k_per_class"], r["k_after"], r["decision"], r["val_rows_before"], r["val_rows_after"])
        for r in out["cycle_table_rows"]
    ]


RULE = {"min_vprime_per_class": 25, "perfect_recheck_expansions": 1, "perfect_recheck_min_add_per_class": 25}


def test_default_parameters_equal_explicit_zero_parameters(tmp_path, easy_data):
    implicit, _ = _run(tmp_path, easy_data, _cfg(), "implicit")
    explicit, _ = _run(
        tmp_path,
        easy_data,
        _cfg(daga={"min_vprime_per_class": 0, "perfect_recheck_expansions": 0, "perfect_recheck_min_add_per_class": 0}),
        "explicit",
    )
    assert _traj(implicit) == _traj(explicit)
    assert all(not r["perfect_stop_blocked"] for r in implicit["cycle_table_rows"])
    assert implicit["phase_study"]["stop_rule"]["perfect_stops_blocked_total"] == 0


def test_small_vprime_with_perfect_ba_cannot_stop_before_minimum_size(tmp_path, easy_data):
    y_va = easy_data[3]
    # reguła historyczna: na łatwym zbiorze i małym V' (4 próbki na klasę) algorytm kończy się od razu po utwardzeniu
    legacy, _ = _run(tmp_path, easy_data, _cfg(), "legacy")
    assert legacy["phase_study"]["stop_decision"] == "stop_perfect_on_current_val"
    legacy_size = legacy["cycle_table_rows"][-1]["val_rows_before"]
    assert legacy_size < 2 * RULE["min_vprime_per_class"]  # przedwczesny stop na zbyt małym V'

    new, _ = _run(tmp_path, easy_data, _cfg(daga=RULE), "new")
    rows = new["cycle_table_rows"]
    assert new["phase_study"]["stop_decision"] == "stop_perfect_on_current_val"  # BA=1,0 nadal może zatrzymać...
    last = rows[-1]
    # ...ale dopiero po spełnieniu obu warunków
    assert last["min_vprime_size_reached"] is True or last["min_vprime_size_reached"] == 1
    assert last["vprime_min_class_count"] >= RULE["min_vprime_per_class"]
    assert last["perfect_rechecks_done"] >= RULE["perfect_recheck_expansions"]
    assert last["val_rows_before"] > legacy_size
    # żaden wcześniejszy cykl nie zakończył się stopem, a stop zablokowany został odnotowany
    assert all(r["decision"] in (DECISION_EXPAND, DECISION_INCREASE_K) for r in rows[:-1])
    sr = new["phase_study"]["stop_rule"]
    assert sr["perfect_gate_pending_cycles"] >= 2  # BA=1,0 w cyklu 1 (za małe V') i w cyklu przed ponownym sprawdzeniem
    assert new["phase_study"]["stop_category"] == "perfect_ba"
    assert rows[0]["perfect_on_current_val"] and "min_size" in rows[0]["perfect_gate_pending"]
    assert sr["perfect_stops_blocked_total"] == sum(1 for r in rows if r["perfect_stop_blocked"])
    # tylko ostatni wiersz niesie przyczynę zakończenia
    assert [r["stop_category"] for r in rows[:-1]] == [None] * (len(rows) - 1) and rows[-1]["stop_category"] == "perfect_ba"
    assert int(y_va.size) > rows[-1]["val_rows_before"]


def test_recheck_expansion_happens_after_minimum_and_precedes_the_stop(tmp_path, easy_data):
    out, ck = _run(tmp_path, easy_data, _cfg(daga=RULE), "recheck")
    rows = out["cycle_table_rows"]
    first_ok = next(i for i, r in enumerate(rows) if r["min_vprime_size_reached"])
    stop_i = len(rows) - 1
    assert stop_i > first_ok, "po osiągnięciu minimum musi nastąpić co najmniej jedno rozszerzenie i kolejny cykl"
    # w cyklu, w którym minimum zostało po raz pierwszy osiągnięte, jeszcze nie ma żadnego ponownego sprawdzenia
    assert rows[first_ok]["perfect_rechecks_done"] == 0
    expansions_after_min = [r for r in rows[first_ok:stop_i] if r["decision"] == DECISION_EXPAND]
    assert len(expansions_after_min) >= RULE["perfect_recheck_expansions"]
    # rozszerzenie ponownego sprawdzenia dodaje co najmniej blok `perfect_recheck_min_add_per_class` na klasę
    gate_exp = [r for r in rows[first_ok:stop_i] if r["perfect_gate_pending"] == "recheck"]
    assert gate_exp and all(r["val_rows_added"] >= 2 * RULE["perfect_recheck_min_add_per_class"] for r in gate_exp)
    # V' nie traci elementów (pamięć) i rośnie monotonicznie
    ids = [np.load(ck / f"cycle_{r['cycle']:02d}_vprime_ids.npy") for r in rows]
    for a, b in zip(ids, ids[1:]):
        assert np.array_equal(b[: a.size], a)
    assert out["phase_study"]["stop_rule"]["perfect_rechecks_total"] >= 1


def test_two_required_rechecks_postpone_the_stop_by_two_expansions(tmp_path, easy_data):
    one, _ = _run(tmp_path, easy_data, _cfg(daga=RULE), "r1")
    two, _ = _run(tmp_path, easy_data, _cfg(daga={**RULE, "perfect_recheck_expansions": 2}), "r2")
    assert len(two["cycle_table_rows"]) > len(one["cycle_table_rows"])
    assert two["cycle_table_rows"][-1]["perfect_rechecks_done"] >= 2


def test_blocked_perfect_with_exhausted_candidates_stops_as_no_candidates_and_never_grows_k(tmp_path):
    X, y = make_moons(n_samples=400, noise=0.05, random_state=1)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.4, stratify=y, random_state=1)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=0.7, stratify=y_rest, random_state=2)
    data = (X_tr, y_tr, X_va, y_va, X_te, y_te)
    # minimum większy niż cały zbiór V: wymagane minimum jest ograniczane do najmniejszej klasy w V,
    # a po wyczerpaniu kandydatów algorytm musi się zatrzymać (bez wzrostu K)
    cfg = _cfg(daga={"min_vprime_per_class": 10_000, "perfect_recheck_expansions": 1, "perfect_recheck_min_add_per_class": 10_000})
    out, _ = _run(tmp_path, data, cfg, "exhaust")
    rows = out["cycle_table_rows"]
    assert out["phase_study"]["stop_category"] in {"no_candidates", "perfect_ba", "no_k_growth_signal", "k_limit"}
    assert out["phase_study"]["stop_rule"]["min_vprime_per_class_effective"] == int(min(np.bincount(y_va)))
    assert all(r["k_increases_this_cycle"] <= 1 for r in rows)
    assert all(not (r["perfect_on_current_val"] and r["k_increased_this_cycle"]) for r in rows)
    assert len(rows) < 60  # zakończył się sam, nie bezpiecznikiem max_cycles


def test_perfect_ba_never_grows_k_and_at_most_one_k_increase_per_cycle(tmp_path, easy_data):
    for selection in ("heuristic", "random"):
        out, _ = _run(tmp_path, easy_data, _cfg(daga=RULE, selection=selection), f"k_{selection}")
        rows = out["cycle_table_rows"]
        previous = None
        for r in rows:
            assert r["k_increases_this_cycle"] in (0, 1)
            if previous is not None:
                assert r["k_per_class"] == previous
            previous = r["k_after"]
            if r["perfect_on_current_val"]:
                assert r["k_after"] == r["k_per_class"] and not r["k_increased_this_cycle"]
        assert out["phase_study"]["max_k_increases_in_one_cycle"] <= 1


class _PoisonedObserver(FullValidationObserver):
    def __init__(self, X, y):
        super().__init__(X, y, "balanced_accuracy", None, enabled=True)
        self._rng = np.random.default_rng(321)

    def observe(self, clf):
        self.calls += 1
        value = float(self._rng.uniform(0, 1))
        return {"metric": value, "all": {"accuracy": value}, "score_time": 0.0, "num_support_vectors": 0}


@pytest.mark.parametrize("selection", ["heuristic", "evolutionary", "random"])
def test_full_validation_metric_never_controls_the_extended_rule(tmp_path, easy_data, selection):
    cfg = _cfg(daga=RULE, selection=selection)
    on, ck_on = _run(tmp_path, easy_data, cfg, "obs_on")
    off_cfg = copy.deepcopy(cfg)
    off_cfg["observation"]["full_validation"] = False
    off, ck_off = _run(tmp_path, easy_data, off_cfg, "obs_off")
    poisoned, ck_p = _run(tmp_path, easy_data, cfg, "obs_poison", observer_factory=_PoisonedObserver)
    assert _traj(on) == _traj(off) == _traj(poisoned)
    keys = ("perfect_stop_blocked", "perfect_rechecks_done", "min_vprime_size_reached", "stop_category")
    for a, b, c in zip(on["cycle_table_rows"], off["cycle_table_rows"], poisoned["cycle_table_rows"]):
        assert all(a[k] == b[k] == c[k] for k in keys)
    assert on["phase_study"]["used_full_validation_for_control"] is False
    assert all(c["used_full_validation_for_control"] is False for c in on["cycles"])


def test_fixed_vprime_is_unaffected_by_the_extended_rule(tmp_path, easy_data):
    base = _cfg(selection="random", mode="fixed", fixed_size=120)
    a, _ = _run(tmp_path, easy_data, base, "fix_a")
    b, _ = _run(tmp_path, easy_data, _cfg(daga=RULE, selection="random", mode="fixed", fixed_size=120), "fix_b")
    assert _traj(a) == _traj(b)
    assert all(r["val_rows_added"] == 0 for r in b["cycle_table_rows"])


# --------------------------------------------------------------------------------------------
# 3. wspólny budżet puli kandydatów i uczciwe liczniki kosztu
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pool():
    from sklearn.svm import SVC

    X, y = make_moons(n_samples=1500, noise=0.3, random_state=3)
    clf = SVC(kernel="rbf", C=1.0, gamma=1.0, random_state=0).fit(X[:30], y[:30])
    return X, y, clf


def _expand(pool, strategy, *, factor, seed=0, n_add=10):
    from ga.loop.engine import GAConfig

    X, y, clf = pool
    vset = VPrimeSet(len(y), np.arange(0, 40))
    counter = {"model_predictions": 0, "diagnostic_predictions": 0}
    res = expand_vprime(
        vset,
        memory=True,
        strategy=strategy,
        n_add=n_add,
        per_class_add=n_add // 2,
        n_classes=2,
        X_val=X,
        y_val=y,
        clf=clf,
        metric="balanced_accuracy",
        val_ga_cfg=GAConfig(population_size=8, generations=4, mutation_rate=0.1),
        seed=seed,
        rng=np.random.default_rng(seed),
        counter=counter,
        pool_budget_factor=factor,
    )
    return res, counter


def test_budget_pool_is_independent_of_strategy_and_limits_scored_rows(pool):
    results = {s: _expand(pool, s, factor=5.0) for s in ("random", "heuristic", "evolutionary")}
    for strategy, (res, counter) in results.items():
        assert res.diagnostics["n_scored_pool"] == 50  # 5 × 10 wybieranych
        assert res.diagnostics["n_candidates"] == 1500 - 40
        assert res.selected_ids.size == 10
    # model widzi tylko podpulę: faktyczne predykcje == rozmiar podpuli (ewolucyjny, heurystyka), 0 dla losowego
    assert results["random"][1].get("selection_predict_rows", 0) == 0
    assert results["evolutionary"][1]["selection_predict_rows"] == 50
    assert results["heuristic"][1]["selection_predict_rows"] == 50
    assert results["heuristic"][1]["selection_decision_rows"] == 50
    assert results["evolutionary"][1].get("selection_decision_rows", 0) == 0
    # wszystkie strategie wybierają z TEJ SAMEJ podpuli (zależy od seedu, nie od strategii/modelu)
    X, y, clf = pool
    vset_ids = np.arange(0, 40)
    cand = np.setdiff1d(np.arange(len(y)), vset_ids)
    sub = budget_pool_ids(cand, y[cand], 50, np.random.default_rng(np.random.SeedSequence([0, 303])))
    for strategy, (res, _) in results.items():
        assert np.isin(res.selected_ids, sub).all(), strategy


def test_without_budget_the_full_pool_is_scored_and_nominal_cost_is_reported_separately(pool):
    res, counter = _expand(pool, "evolutionary", factor=None)
    assert res.diagnostics["n_scored_pool"] == res.diagnostics["n_candidates"] == 1460
    # faktyczne predykcje: jedno przejście po puli; nominalne oceny Val-GA to inna (większa) liczba
    assert counter["selection_predict_rows"] == 1460
    assert res.diagnostics["val_ga_nominal_subset_rows_scored"] == (
        res.diagnostics["val_ga_population_evaluations"] * res.diagnostics["selected_rows"]
    )
    assert res.diagnostics["val_ga_nominal_subset_rows_scored"] != counter["selection_predict_rows"]
    assert res.diagnostics["cost_selection_predict_rows"] == 1460


def test_budget_is_a_no_op_when_the_pool_is_smaller_than_the_budget(pool):
    a, _ = _expand(pool, "heuristic", factor=None, n_add=10)
    b, _ = _expand(pool, "heuristic", factor=1000.0, n_add=10)
    assert np.array_equal(a.selected_ids, b.selected_ids)
    with pytest.raises(ValueError):
        VPrimeConfig(selection="random", pool_budget_factor=0.5)


# --------------------------------------------------------------------------------------------
# 4. integracja z CLI: cycles.csv i summary.json zawierają informacje o regule stopu i koszcie
# --------------------------------------------------------------------------------------------
def test_cli_run_writes_stop_rule_and_cost_columns(tmp_path):
    import json
    import subprocess

    import pandas as pd
    import yaml

    X, y = make_moons(n_samples=900, noise=0.1, random_state=42)
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    parts = {"train": frame.iloc[:350], "validation": frame.iloc[350:700], "test": frame.iloc[700:]}
    ddir = tmp_path / "data"
    ddir.mkdir()
    for name, part in parts.items():
        part.to_csv(ddir / f"{name}.csv", index=False)
    cfg = _cfg(
        daga={**RULE, "k_start": 4},
        selection="evolutionary",
        pool_budget_factor=4.0,
    )
    cfg.update(
        {
            "experiment_name": "cli_stop_rule",
            "dataset_name": "moons-test",
            "algorithm_mode": "dagasvm_phase",
            "data": {
                "train_csv": str(ddir / "train.csv"),
                "validation_csv": str(ddir / "validation.csv"),
                "test_csv": str(ddir / "test.csv"),
                "target": "label",
                "stratify": True,
                "header": True,
                "target_candidates": ["label"],
                "max_rows": 0,
            },
            "svm": {"kernel": "rbf", "C": 10.0, "gamma": 1.0, "class_weight": "balanced", "probability": False, "standardize": False},
            "artifacts_root": str(tmp_path / "artifacts"),
        }
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    run_dir = next((tmp_path / "artifacts" / "runs").iterdir())
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    cycles = pd.read_csv(run_dir / "metrics" / "cycles.csv")
    for column in (
        "min_vprime_per_class_required", "min_vprime_size_required", "min_vprime_size_reached", "perfect_rechecks_done",
        "perfect_rechecks_required", "perfect_stop_blocked", "perfect_stop_blocked_reason", "stop_category",
        "vprime_candidates", "vprime_scored_pool", "vprime_increment_rows", "selection_predict_rows",
        "val_ga_nominal_subset_rows_scored",
    ):
        assert column in cycles.columns, column
    ps = summary["phase_study"]
    assert ps["stop_category"] in set(STOP_CATEGORY.values())
    assert ps["stop_rule"]["min_vprime_per_class_required"] == RULE["min_vprime_per_class"]
    assert ps["stop_rule"]["perfect_recheck_expansions_required"] == RULE["perfect_recheck_expansions"]
    assert ps["vprime"]["pool_budget_factor"] == 4.0
    assert cycles["stop_category"].notna().sum() == 1 and cycles["stop_category"].iloc[-1] == ps["stop_category"]
    expansions = cycles[cycles["vprime_scored_pool"].notna()]
    assert (expansions["vprime_scored_pool"] <= expansions["vprime_candidates"]).all()
    assert (expansions["vprime_scored_pool"] <= 4.0 * expansions["vprime_increment_rows"].clip(lower=1) + 1).all()
    # faktyczne wiersze przekazane modelowi != nominalna liczba ocen Val-GA
    assert ps["cost"]["selection_predict_rows"] == int(expansions["selection_predict_rows"].sum())
    assert ps["cost"]["val_ga_nominal_subset_rows_scored"] == int(expansions["val_ga_nominal_subset_rows_scored"].sum())
