"""Testy automatu fazowego T' <-> V' (``algorithm_mode: dagasvm_phase_switch``, ``ga.phase_switch``).

1. czyste funkcje decyzyjne i kontroler (skryptowane wartości fitnessu/trudności, bez SVM),
2. reguła trudności przyrostu ΔV' i kontrola rozmiaru V',
3. integracja z prawdziwym GA/SVM (niezmienniki kroku, limit |V'|, brak sterowania pełnym V/testem),
4. zgodność wsteczna GA (warm start / target_fitness domyślnie wyłączone) i CLI.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.encoding import KPerClassConfig, init_k_per_class
from ga.fitness import FitnessConfig
from ga.loop.engine import GAConfig, run_ga_k_per_class_with_trace
from ga.models.svm import SVMConfig
from ga.phase_switch import (
    PHASE_STOP,
    PHASE_SWITCH_COLUMNS,
    PHASE_T,
    PHASE_V,
    STOP_CATEGORY,
    CountingModel,
    DifficultyTracker,
    PhaseSwitchController,
    PhaseSwitchSettings,
    decide_after_tprime,
    decide_after_vprime,
    expansion_block_reason,
    parse_phase_switch_settings,
    plan_increment_per_class,
    run_phase_switch_experiment,
    vprime_cap_rows,
)
from ga.utils.artifacts import ensure_artifact_layout
from ga.utils.logger import get_logger
from ga.vprime import FullValidationObserver, VPrimeConfig


# ============================================================================================
# 1. funkcja decyzyjna po kroku T'
# ============================================================================================
def _t(**kw):
    base = dict(perfect=False, vprime_expandable=True, no_improvement_count=0, max_no_improvement=3, k_can_grow=True,
                sv_signal=True, k_growth_requires_sv_signal=False, allow_retry=True, retry_used_at_current_k=False)
    base.update(kw)
    return decide_after_tprime(**base)


def test_perfect_vprime_switches_to_vprime_phase_and_does_not_stop():
    d = _t(perfect=True)
    assert d.next_phase == PHASE_V and not d.increase_k and not d.retry and d.stop_code is None


def test_perfect_vprime_never_increases_k_for_any_combination():
    import itertools

    for exp, cnt, grow, sig, req, allow, used in itertools.product(
        [False, True], [0, 3, 5], [False, True], [False, True], [False, True], [False, True], [False, True]
    ):
        d = _t(perfect=True, vprime_expandable=exp, no_improvement_count=cnt, k_can_grow=grow, sv_signal=sig,
               k_growth_requires_sv_signal=req, allow_retry=allow, retry_used_at_current_k=used)
        assert not d.increase_k
        assert d.next_phase == (PHASE_V if exp else PHASE_STOP)


def test_perfect_vprime_at_size_limit_is_the_only_perfect_stop():
    d = _t(perfect=True, vprime_expandable=False)
    assert d.stop_code == "stop_perfect_on_max_vprime"


def test_imperfect_tprime_increases_k_instead_of_switching_to_vprime():
    d = _t(perfect=False, no_improvement_count=2)
    assert d.next_phase == PHASE_T and d.increase_k


def test_stall_triggers_one_vprime_retry_then_stop():
    d = _t(no_improvement_count=3)
    assert d.next_phase == PHASE_V and d.retry
    d2 = _t(no_improvement_count=3, retry_used_at_current_k=True)
    assert d2.next_phase == PHASE_STOP and d2.stop_code == "stop_tprime_no_progress"
    d3 = _t(no_improvement_count=3, allow_retry=False)
    assert d3.stop_code == "stop_tprime_no_progress"
    d4 = _t(no_improvement_count=3, vprime_expandable=False)
    assert d4.stop_code == "stop_tprime_no_progress"


def test_k_limit_and_optional_sv_gate_are_stalls():
    assert _t(k_can_grow=False).retry
    assert _t(k_can_grow=False, retry_used_at_current_k=True).stop_code == "stop_k_limit"
    assert _t(sv_signal=False).increase_k  # ρ_SV domyślnie nie blokuje wzrostu K
    assert _t(sv_signal=False, k_growth_requires_sv_signal=True).retry
    assert _t(sv_signal=False, k_growth_requires_sv_signal=True,
              retry_used_at_current_k=True).stop_code == "stop_no_k_growth_signal"


def test_failed_retry_check_legacy_first_step_stops_with_dedicated_code():
    d = _t(retry_check=True, retry_progress=False, perfect=False, retry_resolution="first_step")
    assert d.stop_code == "stop_no_progress_after_vprime_retry"


def test_failed_retry_check_gives_t_a_chance_after_increasing_k():
    """Specyfikacja: „jeżeli po powrocie do T', NAWET PO ZWIĘKSZENIU K, nie uda się osiągnąć poziomu” -> stop."""
    d = _t(retry_check=True, retry_progress=False, no_improvement_count=0)
    assert d.next_phase == PHASE_T and d.increase_k and d.stop_code is None
    d = _t(retry_check=True, retry_progress=False, no_improvement_count=2)
    assert d.increase_k
    # stop dopiero po `max_no_improvement` zwiększeniach K bez osiągnięcia poziomu, albo gdy K nie może rosnąć
    assert _t(retry_check=True, retry_progress=False, no_improvement_count=3).stop_code == \
        "stop_no_progress_after_vprime_retry"
    assert _t(retry_check=True, retry_progress=False, k_can_grow=False).stop_code == \
        "stop_no_progress_after_vprime_retry"
    assert _t(retry_check=True, retry_progress=False, sv_signal=False,
              k_growth_requires_sv_signal=True).stop_code == "stop_no_progress_after_vprime_retry"
    # osiągnięcie poziomu odniesienia (retry_progress) wraca do zwykłej ścieżki: tu K rośnie
    assert _t(retry_check=True, retry_progress=True).increase_k
    # BA(V')=1 po próbie to „opanowanie V'”, nie porażka próby
    assert _t(retry_check=True, retry_progress=False, perfect=True).next_phase == PHASE_V


def test_every_stop_code_has_a_category():
    codes = {"stop_perfect_on_max_vprime", "stop_no_progress_after_vprime_retry", "stop_tprime_no_progress",
             "stop_k_limit", "stop_no_k_growth_signal", "stop_max_cycles"}
    assert codes <= set(STOP_CATEGORY)


# ============================================================================================
# 2. kontroler: skryptowany przebieg (bez SVM)
# ============================================================================================
def _settings(**kw):
    return PhaseSwitchSettings(vprime=VPrimeConfig(selection="random"), **kw)


def test_controller_full_cycle_t_to_v_to_t():
    ctl = PhaseSwitchController(_settings(vprime_difficulty_patience=2, vprime_difficulty_min_delta=0.01), k_start=5)
    r = ctl.on_tprime(fitness=1.0, k_next=8, sv_signal=True, vprime_expandable=True)
    assert ctl.phase == PHASE_V and ctl.k == 5 and r["decision"].next_phase == PHASE_V
    # trudność rośnie -> faza V' trwa; potem przestaje rosnąć -> powrót do T'
    assert ctl.on_vprime(difficulty=0.1, n_new=10, can_expand_more=True)["next_phase"] == PHASE_V
    assert ctl.on_vprime(difficulty=0.3, n_new=10, can_expand_more=True)["next_phase"] == PHASE_V
    assert ctl.on_vprime(difficulty=0.3, n_new=10, can_expand_more=True)["next_phase"] == PHASE_V  # okno: 0,30 > 0,20
    assert ctl.on_vprime(difficulty=0.3, n_new=10, can_expand_more=True)["next_phase"] == PHASE_V  # 1. bez wzrostu
    res = ctl.on_vprime(difficulty=0.3, n_new=10, can_expand_more=True)                            # 2. bez wzrostu
    assert res["next_phase"] == PHASE_T and res["reason"] == "difficulty_plateau" and ctl.k == 5
    # pierwszy krok T' na nowym V' jest punktem odniesienia (V' się zmieniło)
    r = ctl.on_tprime(fitness=0.9, k_next=8, sv_signal=True, vprime_expandable=True)
    assert r["improved"] is None and r["decision"].increase_k and ctl.k == 8


def test_controller_stall_retry_success_then_second_stall_at_same_k_stops():
    st = _settings(max_no_improvement_k_increases=3, fitness_min_delta=0.001)
    ctl = PhaseSwitchController(st, k_start=4)
    ctl.on_tprime(fitness=0.90, k_next=6, sv_signal=True, vprime_expandable=True)          # odniesienie, K 4->6
    for k_next in (8, 10):
        r = ctl.on_tprime(fitness=0.90, k_next=k_next, sv_signal=True, vprime_expandable=True)
        assert r["decision"].increase_k
    r = ctl.on_tprime(fitness=0.9005, k_next=12, sv_signal=True, vprime_expandable=True)   # 3. bez poprawy
    assert r["decision"].retry and ctl.phase == PHASE_V and ctl.k == 10 and ctl.retry_used_k == {10}
    ctl.on_vprime(difficulty=0.2, n_new=4, can_expand_more=False)                           # limit -> T'
    assert ctl.phase == PHASE_T
    r = ctl.on_tprime(fitness=0.88, k_next=12, sv_signal=True, vprime_expandable=True, retry_reference_same=0.86)
    assert r["retry_check"] and r["retry_progress"] and not r["retry_progress_raw"]  # lepiej na tym samym V'
    assert r["decision"].increase_k and ctl.k == 12
    # K nie może dalej rosnąć -> zastój; druga próba jest dozwolona, bo to nowy poziom K (12), przy K=10 nie byłaby
    r = ctl.on_tprime(fitness=0.88, k_next=ctl.k, sv_signal=True, vprime_expandable=True)
    assert r["decision"].retry and ctl.retry_used_k == {10, 12}


def test_controller_failed_retry_legacy_first_step_stops_immediately():
    ctl = PhaseSwitchController(_settings(max_no_improvement_k_increases=1, retry_resolution="first_step"), k_start=4)
    ctl.on_tprime(fitness=0.9, k_next=6, sv_signal=True, vprime_expandable=True)
    r = ctl.on_tprime(fitness=0.9, k_next=8, sv_signal=True, vprime_expandable=True)
    assert r["decision"].retry
    ctl.on_vprime(difficulty=0.0, n_new=2, can_expand_more=False)
    r = ctl.on_tprime(fitness=0.85, k_next=8, sv_signal=True, vprime_expandable=True, retry_reference_same=0.85)
    assert r["decision"].stop_code == "stop_no_progress_after_vprime_retry"


def _drive_to_retry(ctl, k_start_fitness=0.90):
    """Doprowadza kontroler do próby V' (zastój T'): 1 krok odniesienia + `max_no_improvement` zwiększeń K bez poprawy."""
    n = ctl.st.max_no_improvement_k_increases
    r = ctl.on_tprime(fitness=k_start_fitness, k_next=ctl.k + 2, sv_signal=True, vprime_expandable=True)
    assert r["decision"].increase_k
    for _ in range(n):
        r = ctl.on_tprime(fitness=k_start_fitness, k_next=ctl.k + 2, sv_signal=True, vprime_expandable=True)
    assert r["decision"].retry and ctl.phase == PHASE_V
    ctl.on_vprime(difficulty=0.2, n_new=4, can_expand_more=False)  # krótka faza V' (limit) -> T'
    assert ctl.phase == PHASE_T and ctl.retry is not None
    return r


def test_three_fruitless_k_increases_trigger_exactly_one_vprime_retry_and_then_the_algorithm_stops():
    """Pkt 7 i 8: trzy nieskuteczne zwiększenia K => jedna próba V'; brak poprawy PO ZWIĘKSZENIU K => stop."""
    ctl = PhaseSwitchController(_settings(max_no_improvement_k_increases=3), k_start=4)
    _drive_to_retry(ctl)
    retries, steps_after, k_increases_after, last = 1, 0, 0, None
    while ctl.phase != PHASE_STOP:
        steps_after += 1
        assert steps_after < 20, "automat nie zatrzymał się"
        # nic nie poprawia najlepszego T' względem odniesienia (to samo V', fitness 0.90)
        last = ctl.on_tprime(fitness=0.90, k_next=ctl.k + 2, sv_signal=True, vprime_expandable=True,
                             retry_reference_same=0.90)
        k_increases_after += int(last["decision"].increase_k)
        retries += int(last["decision"].retry)
    assert retries == 1
    assert last["decision"].stop_code == "stop_no_progress_after_vprime_retry"
    assert (steps_after, k_increases_after) == (4, 3)  # 1. krok na tym samym K + 3 kroki po zwiększeniu K


def test_retry_resolved_by_progress_after_a_k_increase_resumes_normal_operation():
    ctl = PhaseSwitchController(_settings(max_no_improvement_k_increases=3, fitness_min_delta=0.001), k_start=4)
    _drive_to_retry(ctl)
    r = ctl.on_tprime(fitness=0.90, k_next=ctl.k + 2, sv_signal=True, vprime_expandable=True, retry_reference_same=0.90)
    assert r["retry_check"] and not r["retry_progress"] and r["retry_pending"] and r["decision"].increase_k
    r = ctl.on_tprime(fitness=0.90, k_next=ctl.k + 2, sv_signal=True, vprime_expandable=True)
    assert r["retry_pending"] and r["retry_fail_count"] == 1 and r["decision"].increase_k
    r = ctl.on_tprime(fitness=0.93, k_next=ctl.k + 2, sv_signal=True, vprime_expandable=True)
    assert r["retry_progress"] and not r["retry_pending"] and r["decision"].increase_k and ctl.retry is None
    assert r["no_improvement_count"] == 0


def test_literal_retry_reference_compares_with_the_remembered_fitness_from_the_old_vprime():
    st = _settings(retry_reference="literal", fitness_min_delta=0.001)
    ctl = PhaseSwitchController(st, k_start=4)
    _drive_to_retry(ctl, k_start_fitness=0.95)
    # na nowym (trudniejszym) V' fitness 0.90 < zapamiętane 0.95 => nie ma postępu, choć T' jest lepsze niż T' sprzed próby (0.85)
    r = ctl.on_tprime(fitness=0.90, k_next=ctl.k + 2, sv_signal=True, vprime_expandable=True, retry_reference_same=0.85)
    assert not r["retry_progress"] and r["retry_progress_raw"] is False
    assert r["retry_reference_raw"] == pytest.approx(0.95) and r["retry_reference_same_vprime"] == pytest.approx(0.85)
    r = ctl.on_tprime(fitness=0.95, k_next=ctl.k + 2, sv_signal=True, vprime_expandable=True)
    assert r["retry_progress"] and not r["retry_pending"]  # „co najmniej” zapamiętany poziom


def test_scripted_controller_always_terminates():
    """Fitness, który nigdy się nie poprawia, i rosnące K do limitu — automat musi się zatrzymać."""
    rng = np.random.default_rng(0)
    for trial in range(50):
        st = _settings(max_no_improvement_k_increases=int(rng.integers(1, 4)),
                       vprime_difficulty_patience=int(rng.integers(1, 3)))
        ctl = PhaseSwitchController(st, k_start=2)
        rows, cap, steps = 10, 60, 0
        while ctl.phase != PHASE_STOP:
            steps += 1
            assert steps < 500
            if ctl.phase == PHASE_T:
                f = float(rng.choice([1.0, 0.9, 0.95]))
                ref = float(rng.uniform(0.8, 1.0)) if ctl.retry is not None else None
                ctl.on_tprime(fitness=f, k_next=min(ctl.k + 2, 30), sv_signal=True, vprime_expandable=rows < cap,
                              retry_reference_same=ref)
            else:
                rows += 2
                ctl.on_vprime(difficulty=float(rng.uniform(0, 0.5)), n_new=2, can_expand_more=rows < cap)


# ============================================================================================
# 3. reguła trudności ΔV' i rozmiar V'
# ============================================================================================
def _track(seq, **kw):
    tr = DifficultyTracker(min_delta=kw.get("min_delta", 0.01), patience=kw.get("patience", 2),
                           window=kw.get("window", 2), min_difficulty=kw.get("min_difficulty", 0.0))
    return [tr.update(d) for d in seq]


def test_difficulty_steadily_rising_never_plateaus():
    ups = _track([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert not any(u.plateau for u in ups) and all(u.no_gain_count == 0 for u in ups)


def test_difficulty_that_stops_rising_ends_the_phase_after_patience():
    ups = _track([0.3, 0.3, 0.3])
    assert [u.no_gain_count for u in ups] == [0, 1, 2] and ups[-1].plateau
    ups = _track([0.4, 0.2, 0.1], patience=2)  # spadek trudności (typowe dla doboru adwersarialnego)
    assert ups[-1].plateau
    assert ups[1].change_vs_prev == pytest.approx(-0.2)


def test_zero_difficulty_increment_counts_as_no_gain():
    ups = _track([0.0, 0.0], patience=2)
    assert ups[0].no_gain_count == 1 and ups[1].plateau


def test_window_smooths_a_single_dip():
    # okno 3: pojedynczy spadek nie zeruje trendu wzrostowego, gdy średnia nadal rośnie
    ups = _track([0.1, 0.3, 0.25, 0.5], window=3, patience=1)
    assert not ups[2].plateau and not ups[3].plateau


def test_decide_after_vprime_reasons():
    kw = dict(expansions_in_phase=1, max_expansions_per_phase=5)
    assert decide_after_vprime(n_new=0, plateau=False, can_expand_more=True, **kw)[1] == "no_new_elements"
    assert decide_after_vprime(n_new=4, plateau=True, can_expand_more=True, **kw)[1] == "difficulty_plateau"
    assert decide_after_vprime(n_new=4, plateau=False, can_expand_more=False, **kw)[1] == "vprime_limit_or_no_candidates"
    assert decide_after_vprime(n_new=4, plateau=False, can_expand_more=True, expansions_in_phase=5,
                               max_expansions_per_phase=5)[1] == "max_expansions_per_phase"
    assert decide_after_vprime(n_new=4, plateau=False, can_expand_more=True, **kw)[0] == PHASE_V


def test_vprime_cap_and_increment_planning():
    assert vprime_cap_rows(27853, 0.25, None) == 6963
    assert vprime_cap_rows(27853, 0.25, 1000) == 1000
    assert vprime_cap_rows(100, 1.0, None) == 100
    kw = dict(n_classes=2, increment_per_class=10)
    assert plan_increment_per_class(vprime_rows=20, cap_rows=1000, n_candidates=500, **kw) == 10
    assert plan_increment_per_class(vprime_rows=995, cap_rows=1000, n_candidates=500, **kw) == 2   # dobicie do limitu
    assert plan_increment_per_class(vprime_rows=999, cap_rows=1000, n_candidates=500, **kw) == 0   # limit V'
    assert plan_increment_per_class(vprime_rows=20, cap_rows=1000, n_candidates=1, **kw) == 0      # brak kandydatów
    assert plan_increment_per_class(vprime_rows=2, cap_rows=1000, n_candidates=500, **kw) == 10    # bardzo małe V'


def test_settings_defaults_and_validation():
    st = parse_phase_switch_settings({"daga": {"k_start": 7}, "vprime": {"selection": "random"}})
    assert st.initial_vprime_per_class == 7 and st.tprime_metric == st.vprime_metric == "balanced_accuracy"
    assert st.max_vprime_fraction == 0.25 and st.vprime_increment_per_class == 10
    st = parse_phase_switch_settings({"fitness": {"tprime_metric": "f1_macro", "vprime_metric": "accuracy"},
                                      "vprime": {"selection": "random"}})
    assert (st.tprime_metric, st.vprime_metric) == ("f1_macro", "accuracy")
    with pytest.raises(ValueError):  # Val-GA po cichu zastąpiłby ROC AUC przez BA
        parse_phase_switch_settings({"fitness": {"vprime_metric": "roc_auc"}})
    with pytest.raises(ValueError):
        parse_phase_switch_settings({"phase_switch": {"max_vprime_fraction": 0.0}})
    with pytest.raises(ValueError):
        parse_phase_switch_settings({"vprime": {"selection": "random", "memory": False}})


# ============================================================================================
# 4. integracja (prawdziwe GA/SVM)
# ============================================================================================
@pytest.fixture(scope="module")
def data():
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=5)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.4, stratify=y, random_state=5)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=0.7, stratify=y_rest, random_state=6)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


SVM = SVMConfig(kernel="rbf", C=10.0, gamma=1.0, probability=False, standardize=False)


def _cfg(seed=42, selection="random", **ps):
    return {
        "seed": seed,
        "fitness": {"metric": "balanced_accuracy"},
        "ga": {"population_size": 6, "generations": 4, "tournament_k": 3, "crossover_rate": 0.8, "mutation_rate": 0.05,
               "elitism": 2},
        "daga": {"k_start": 3, "TSV": 0.25, "Ts": 2, "phase_min_delta": 0.0005},
        "val_ga": {"population_size": 6, "generations": 3, "mutation_rate": 0.1, "elitism": 2},
        "vprime": {"selection": selection, "memory": True, "mode": "growing"},
        "phase_switch": {"vprime_increment_per_class": 5, "max_vprime_fraction": 0.15, "max_cycles": 150, **ps},
        "observation": {"full_validation": True, "test": True},
    }


def _run(tmp_path, data, cfg, tag, observer_factory=None, drop_test=False, ga_runner=None, y_test_override=None):
    X_tr, y_tr, X_va, y_va, X_te, y_te = data
    if drop_test:
        X_te, y_te = None, None
    if y_test_override is not None:
        y_te = y_test_override(y_te)
    layout = ensure_artifact_layout(tmp_path / tag, experiment_name=tag)
    observer = None if observer_factory is None else observer_factory(X_va, y_va)
    kw = {} if ga_runner is None else {"ga_runner": ga_runner}
    return run_phase_switch_experiment(X_tr, y_tr, X_va, y_va, X_te, y_te, SVM, cfg, get_logger(), layout,
                                       observer=observer, **kw)


def _ids(tmp_path, tag):
    return np.load(next((tmp_path / tag).rglob("final_vprime_ids.npy")))


def _increments(tmp_path, tag):
    path = next((tmp_path / tag).rglob("vprime_increments.npz"), None)
    return {} if path is None else dict(np.load(path))


def _recording_runner(log):
    """ga_runner, który zapamiętuje dane widziane przez GA dla T' (V' i K) i deleguje do prawdziwego GA."""
    def runner(**kw):
        log.append({"X_val": kw["X_val"].copy(), "y_val": kw["y_val"].copy(), "k": int(kw["kcfg"].k_per_class)})
        return run_ga_k_per_class_with_trace(**kw)
    return runner


@pytest.fixture(scope="module")
def easy_data():
    """Dwa odległe skupiska: każde T' (K >= 2/klasę) klasyfikuje bezbłędnie — BA(V') = 1,0 od pierwszego kroku."""
    X, y = make_blobs(n_samples=1400, centers=[(-4.0, -4.0), (4.0, 4.0)], cluster_std=0.6, random_state=3)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.3, stratify=y, random_state=3)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=0.7, stratify=y_rest, random_state=4)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _traj(out):
    keys = ("phase", "k_per_class", "k_after", "vprime_rows_before", "vprime_rows_after", "tprime_fitness", "decision")
    return [tuple(r[k] for k in keys) for r in out["cycle_table_rows"]]


@pytest.mark.parametrize("selection", ["random", "evolutionary", "heuristic"])
def test_step_invariants_and_termination(tmp_path, data, selection):
    out = _run(tmp_path, data, _cfg(selection=selection), selection)
    rows = out["cycle_table_rows"]
    cap = rows[0]["vprime_cap_rows"]
    assert out["phase_study"]["stop_decision"] != "stop_max_cycles"
    for i, r in enumerate(rows):
        assert int(r["k_increased"]) + int(r["vprime_expanded"]) <= 1
        assert r["k_after"] - r["k_per_class"] in (0,) or (r["phase"] == PHASE_T and r["k_increased"])
        assert r["vprime_rows_after"] <= cap
        if r["phase"] == PHASE_T:
            assert r["vprime_rows_after"] == r["vprime_rows_before"]
            if r["tprime_perfect"]:
                assert not r["k_increased"]
        else:
            assert r["k_after"] == r["k_per_class"] and r["vprime_rows_after"] > r["vprime_rows_before"]
            assert r["delta_rows"] == r["vprime_rows_after"] - r["vprime_rows_before"]
            assert r["delta_difficulty"] is not None and r["delta_eval_rows"] == r["delta_rows"]
            prev = rows[i - 1]
            # wejście do fazy V' tylko z kroku T' z decyzją przełączenia albo z kontynuacji fazy V'
            assert (prev["phase"] == PHASE_T and prev["decision"] == "switch_to_V_prime") or \
                   (prev["phase"] == PHASE_V and prev["next_phase"] == PHASE_V)
        assert r["vprime_fraction"] == pytest.approx(r["vprime_rows_after"] / r["full_validation_rows"])
    # najwyżej jedna próba V' po zastoju na poziom K
    retry_ks = [r["k_per_class"] for r in rows if r["phase"] == PHASE_T and r["decision"] == "switch_to_V_prime"
                and not r["tprime_perfect"]]
    assert len(retry_ks) == len(set(retry_ks))
    assert [r["stop_category"] for r in rows[:-1]] == [None] * (len(rows) - 1)
    ps = out["phase_study"]
    assert ps["vprime_rows"] == rows[-1]["vprime_rows_after"] and ps["vprime_fraction"] <= 0.15 + 1e-12


def test_first_perfect_on_tiny_vprime_switches_phase_instead_of_stopping(tmp_path, data):
    out = _run(tmp_path, data, _cfg(initial_vprime_per_class=1), "tiny")
    rows = out["cycle_table_rows"]
    assert rows[0]["vprime_rows_before"] == 2
    if rows[0]["tprime_perfect"]:
        assert rows[0]["decision"] == "switch_to_V_prime" and rows[1]["phase"] == PHASE_V
    assert out["phase_study"]["stop_decision"] != "stop_perfect_on_max_vprime" or out["phase_study"]["vprime_cap_reached"]


@pytest.mark.parametrize("fraction,max_rows", [(0.05, None), (0.3, 40)])
def test_vprime_never_exceeds_the_configured_limit(tmp_path, data, fraction, max_rows):
    out = _run(tmp_path, data, _cfg(max_vprime_fraction=fraction, max_vprime_rows=max_rows), f"cap{fraction}")
    n_val = out["phase_study"]["full_validation_rows"]
    cap = vprime_cap_rows(n_val, fraction, max_rows)
    assert max(r["vprime_rows_after"] for r in out["cycle_table_rows"]) <= cap
    assert out["phase_study"]["vprime_cap_rows"] == cap


def test_exhausted_candidates_end_the_vprime_phase(tmp_path):
    X, y = make_moons(n_samples=600, noise=0.25, random_state=1)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.6, stratify=y, random_state=1)
    X_va, X_te, y_va, y_te = X_rest[:24], X_rest[24:], y_rest[:24], y_rest[24:]
    out = _run(tmp_path, (X_tr, y_tr, X_va, y_va, X_te, y_te),
               _cfg(max_vprime_fraction=1.0, vprime_increment_per_class=4), "exhaust")
    rows = out["cycle_table_rows"]
    assert out["phase_study"]["stop_decision"] != "stop_max_cycles"
    assert max(r["vprime_rows_after"] for r in rows) <= 24
    ends = [r["transition_reason"] for r in rows if r["phase"] == PHASE_V and r["next_phase"] == PHASE_T]
    if out["phase_study"]["vprime_rows"] == 24:
        assert "vprime_limit_or_no_candidates" in ends


@pytest.mark.parametrize("selection", ["random", "evolutionary"])
def test_full_validation_and_test_never_control_the_run(tmp_path, data, selection):
    on = _run(tmp_path, data, _cfg(selection=selection), f"on_{selection}")
    off = _run(tmp_path, data, _cfg(selection=selection), f"off_{selection}",
               observer_factory=lambda X, y: FullValidationObserver(X, y, "balanced_accuracy", None, enabled=False),
               drop_test=True)
    assert _traj(on) == _traj(off)
    assert all(r["observed_full_validation_metric"] is None for r in off["cycle_table_rows"])


def test_same_seed_reproducible(tmp_path, data):
    a = _run(tmp_path, data, _cfg(seed=7), "a")
    b = _run(tmp_path, data, _cfg(seed=7), "b")
    assert _traj(a) == _traj(b)


def test_evolutionary_and_random_share_the_start(tmp_path, data):
    """Ten sam seed => to samo V' startowe i ten sam pierwszy krok T' (różnią się dopiero doborem ΔV')."""
    e = _run(tmp_path, data, _cfg(selection="evolutionary"), "pe")["cycle_table_rows"][0]
    r = _run(tmp_path, data, _cfg(selection="random"), "pr")["cycle_table_rows"][0]
    assert (e["vprime_rows_before"], e["tprime_fitness"], e["decision"]) == \
           (r["vprime_rows_before"], r["tprime_fitness"], r["decision"])


def test_random_replay_reproduces_the_reference_schedule(tmp_path, data):
    """rand_replay: te same fazy, K i |V'| w każdym kroku co przebieg referencyjny; inny jest tylko dobór ΔV'."""
    import pandas as pd

    ref = _run(tmp_path, data, _cfg(selection="evolutionary"), "ref")
    csv = tmp_path / "ref_cycles.csv"
    pd.DataFrame(ref["cycle_table_rows"])[PHASE_SWITCH_COLUMNS].to_csv(csv, index=False)
    rep = _run(tmp_path, data, _cfg(selection="random", schedule_replay=str(csv)), "replay")
    a, b = ref["cycle_table_rows"], rep["cycle_table_rows"]
    key = ("phase", "k_per_class", "k_after", "vprime_rows_before", "vprime_rows_after")
    assert [tuple(r[k] for k in key) for r in a] == [tuple(r[k] for k in key) for r in b]
    assert rep["phase_study"]["stop_decision"] == "stop_replay_end"
    assert rep["phase_study"]["reference_stop_decision"] == ref["phase_study"]["stop_decision"]
    assert rep["phase_study"]["vprime_rows"] == ref["phase_study"]["vprime_rows"]
    ref_ids = np.load(next((tmp_path / "ref").rglob("final_vprime_ids.npy")))
    rep_ids = np.load(next((tmp_path / "replay").rglob("final_vprime_ids.npy")))
    if ref["phase_study"]["vprime_steps"]:
        assert set(ref_ids.tolist()) != set(rep_ids.tolist())


# ============================================================================================
# 5. GA: warm start i target_fitness
# ============================================================================================
def test_ga_new_arguments_default_to_historical_behaviour(data):
    X_tr, y_tr, X_va, y_va, _, _ = data
    kw = dict(X_pool=X_tr, y_pool=y_tr, X_val=X_va[:60], y_val=y_va[:60], kcfg=KPerClassConfig(k_per_class=5),
              svm_cfg=SVM, fcfg=FitnessConfig(metric="balanced_accuracy", class_imbalance_penalty=0.0),
              gcfg=GAConfig(population_size=6, generations=3, elitism=2), seed=3)
    h1, m1, _, _ = run_ga_k_per_class_with_trace(**kw)
    h2, m2, _, _ = run_ga_k_per_class_with_trace(**kw, seed_masks=None, target_fitness=None)
    assert np.array_equal(h1, h2) and np.array_equal(m1, m2)


def test_ga_warm_start_keeps_seed_and_target_stops_early(data):
    X_tr, y_tr, X_va, y_va, _, _ = data
    kw = dict(X_pool=X_tr, y_pool=y_tr, X_val=X_va[:60], y_val=y_va[:60], kcfg=KPerClassConfig(k_per_class=5),
              svm_cfg=SVM, fcfg=FitnessConfig(metric="balanced_accuracy", class_imbalance_penalty=0.0),
              gcfg=GAConfig(population_size=6, generations=5, elitism=2), seed=3)
    h, best, _, _ = run_ga_k_per_class_with_trace(**kw)
    seeded = init_k_per_class(y_tr, KPerClassConfig(k_per_class=5, random_state=99))
    h2, _, _, _ = run_ga_k_per_class_with_trace(**{**kw, "seed": 11}, seed_masks=[best])
    assert h2[0] >= max(h) - 1e-12  # osobnik z warm startu jest w populacji startowej
    h3, _, trace3, _ = run_ga_k_per_class_with_trace(**kw, seed_masks=[seeded], target_fitness=0.0)
    assert len(h3) == 1 and trace3 == []  # cel osiągnięty już w populacji startowej


# ============================================================================================
# 6. CLI
# ============================================================================================
def test_cli_phase_switch_writes_columns_and_vprime_fraction(tmp_path):
    import pandas as pd
    import yaml

    X, y = make_moons(n_samples=900, noise=0.2, random_state=42)
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    parts = {"train": frame.iloc[:350], "validation": frame.iloc[350:700], "test": frame.iloc[700:]}
    ddir = tmp_path / "data"
    ddir.mkdir()
    for name, part in parts.items():
        part.to_csv(ddir / f"{name}.csv", index=False)
    cfg = _cfg(selection="evolutionary", max_vprime_fraction=0.2)
    cfg.update({
        "experiment_name": "cli_phase_switch", "dataset_name": "moons-test", "algorithm_mode": "dagasvm_phase_switch",
        "data": {"train_csv": str(ddir / "train.csv"), "validation_csv": str(ddir / "validation.csv"),
                 "test_csv": str(ddir / "test.csv"), "target": "label", "stratify": True, "header": True,
                 "target_candidates": ["label"], "max_rows": 0},
        "svm": {"kernel": "rbf", "C": 10.0, "gamma": 1.0, "class_weight": "balanced", "probability": False,
                "standardize": False},
        "artifacts_root": str(tmp_path / "artifacts"),
    })
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, str(repo / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
                          capture_output=True, text=True, cwd=str(repo))
    assert proc.returncode == 0, proc.stderr[-2000:]
    run_dir = next((tmp_path / "artifacts" / "runs").iterdir())
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    cycles = pd.read_csv(run_dir / "metrics" / "cycles.csv")
    assert list(cycles.columns) == PHASE_SWITCH_COLUMNS
    ps = summary["phase_study"]
    assert ps["algorithm_mode"] == "dagasvm_phase_switch"
    assert ps["full_validation_rows"] == 350 and ps["vprime_rows"] <= 70
    assert ps["vprime_fraction"] == pytest.approx(ps["vprime_rows"] / 350)
    assert cycles["stop_category"].notna().sum() == 1
    assert summary["report_row"] is not None
    assert (run_dir / "metrics" / "phase_switch_study.json").exists()


# ============================================================================================
# 7. Wymagania z pkt. 14 specyfikacji (jawnie nazwane: test_reqNN_*)
# ============================================================================================
def _phase_rows(out, phase):
    return [r for r in out["cycle_table_rows"] if r["phase"] == phase]


def _easy_cfg(**ps):
    base = dict(initial_vprime_per_class=2, max_vprime_fraction=0.2, vprime_increment_per_class=5)
    base.update(ps)
    return _cfg(selection="random", **base)


def test_req01_perfect_vprime_switches_to_optimize_vprime_and_does_not_end_the_algorithm(tmp_path, easy_data):
    out = _run(tmp_path, easy_data, _easy_cfg(), "r1")
    rows, t_rows = out["cycle_table_rows"], _phase_rows(out, PHASE_T)
    assert rows[0]["tprime_perfect"] and rows[0]["decision"] == "switch_to_V_prime" and rows[0]["next_phase"] == PHASE_V
    assert rows[1]["phase"] == PHASE_V and not rows[0]["k_increased"]
    assert len(t_rows) > 1, "BA(V')=1 zakończyło algorytm po pierwszym kroku"
    # każde T' opanowało V' => każde (poza ostatnim) przełącza do V'; K nigdy nie rośnie przy BA(V')=1
    assert all(r["tprime_perfect"] for r in t_rows)
    assert all(r["decision"] == "switch_to_V_prime" for r in t_rows[:-1])
    assert all(r["k_per_class"] == r["k_after"] == 3 for r in rows)
    # jedyny stop przy BA(V')=1: V' nie da się już rozszerzyć
    assert out["phase_study"]["stop_decision"] == "stop_perfect_on_max_vprime"
    assert rows[-1]["vprime_limit_reason"] is not None


def test_req02_after_vprime_phase_the_algorithm_returns_to_optimize_tprime(tmp_path, easy_data, data):
    for tag, d, cfg in (("r2a", easy_data, _easy_cfg()), ("r2b", data, _cfg(selection="evolutionary"))):
        rows = _run(tmp_path, d, cfg, tag)["cycle_table_rows"]
        assert any(r["phase"] == PHASE_V for r in rows)
        for i, r in enumerate(rows):
            assert r["prev_phase"] == (rows[i - 1]["phase"] if i else None)
            if i + 1 == len(rows) or r["phase"] != PHASE_V:
                continue
            if r["next_phase"] == PHASE_T:
                assert rows[i + 1]["phase"] == PHASE_T and rows[i + 1]["prev_phase"] == PHASE_V
            else:
                assert rows[i + 1]["phase"] == PHASE_V
        assert {r["transition_reason"] for r in rows if r["phase"] == PHASE_V and r["next_phase"] == PHASE_T} <= \
            {"difficulty_plateau", "vprime_limit_or_no_candidates", "max_expansions_per_phase", "no_new_elements"}


@pytest.mark.parametrize("selection", ["evolutionary", "random", "heuristic"])
def test_req03_new_samples_are_selected_and_scored_separately_from_old_ones(tmp_path, data, monkeypatch, selection):
    import ga.vprime as gv

    pool_sizes = []
    orig = gv.run_val_ga

    def spy(X_val, y_val, clf, vcfg, gcfg, **kw):
        pool_sizes.append((int(X_val.shape[0]), float(vcfg.subset_fraction)))
        return orig(X_val, y_val, clf, vcfg, gcfg, **kw)

    monkeypatch.setattr(gv, "run_val_ga", spy)
    out = _run(tmp_path, data, _cfg(selection=selection), f"r3{selection}")
    n_val = data[2].shape[0]
    v_rows = _phase_rows(out, PHASE_V)
    assert v_rows
    ids = _ids(tmp_path, f"r3{selection}")
    inc = _increments(tmp_path, f"r3{selection}")
    seen = set(ids[: v_rows[0]["vprime_rows_before"]].tolist())
    for r in v_rows:
        delta = inc[f"step_{r['cycle']:03d}"]
        assert not (set(delta.tolist()) & seen), "ΔV' zawiera próbki, które już były w V'"
        # pula kandydatów = V \ V' (stare próbki nie są kandydatami)
        assert r["n_candidates"] == n_val - r["vprime_rows_before"]
        assert r["delta_eval_rows"] == r["delta_rows"] == len(delta)
        seen |= set(delta.tolist())
        # liczba wierszy przekazanych modelowi: pula kandydatów (tylko ewolucja) + 2×ΔV' (diagnostyka i ocena przyrostu);
        # ŻADNEGO przejścia po starym V'
        expected = (r["n_candidates"] if selection != "random" else 0) + 2 * r["delta_rows"]
        assert r["predict_rows"] == expected, (selection, r["predict_rows"], expected)
    if selection == "evolutionary":
        assert len(pool_sizes) == len(v_rows)
        for (n_pool, frac), r in zip(pool_sizes, v_rows):
            assert n_pool == r["n_candidates"]                 # Val-GA widzi wyłącznie V \ V'
            assert round(frac * n_pool) == r["delta_rows"]     # osobnik = podzbiór ΔV'; fitness liczony na ΔV'
    # metryka ΔV' = metryka modelu ustalonego policzona bezpośrednio na ΔV'
    assert all(r["delta_metric"] is not None for r in v_rows)


def test_req03b_delta_metric_equals_the_model_metric_computed_only_on_delta(tmp_path, data, monkeypatch):
    import ga.phase_switch as ps_mod

    captured = []
    orig = ps_mod.expand_vprime

    def spy(vset, **kw):
        res = orig(vset, **kw)
        captured.append((kw["clf"], res.selected_ids.copy()))
        return res

    monkeypatch.setattr(ps_mod, "expand_vprime", spy)
    out = _run(tmp_path, data, _cfg(selection="evolutionary"), "r3b")
    X_va, y_va = data[2], data[3]
    for (clf, delta), r in zip(captured, _phase_rows(out, PHASE_V)):
        pred = clf.predict(X_va[delta])
        assert r["delta_metric"] == pytest.approx(balanced_accuracy_score(y_va[delta], pred))
        assert r["delta_error_rate"] == pytest.approx(float((pred != y_va[delta]).mean()))
        assert sum(r["delta_per_class"].values()) == r["delta_rows"]


def test_req04_tprime_is_optimized_on_the_full_cumulative_vprime(tmp_path, data):
    log = []
    out = _run(tmp_path, data, _cfg(selection="evolutionary"), "r4", ga_runner=_recording_runner(log))
    t_rows = _phase_rows(out, PHASE_T)
    X_va, y_va, ids = data[2], data[3], _ids(tmp_path, "r4")
    assert len(log) == len(t_rows)
    for seen, r in zip(log, t_rows):
        n = r["vprime_rows_before"]
        assert seen["X_val"].shape[0] == n
        assert np.array_equal(seen["X_val"], X_va[ids[:n]]) and np.array_equal(seen["y_val"], y_va[ids[:n]])
        assert seen["k"] == r["k_per_class"]
    sizes = [r["vprime_rows_before"] for r in t_rows]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]


def test_req05_k_grows_at_most_once_per_cycle(tmp_path, data):
    log = []
    out = _run(tmp_path, data, _cfg(selection="evolutionary"), "r5", ga_runner=_recording_runner(log))
    rows = out["cycle_table_rows"]
    assert len(log) == len(_phase_rows(out, PHASE_T))  # jedno GA (jedno K) na cykl T'
    for i, r in enumerate(rows):
        assert (r["k_after"] > r["k_per_class"]) == bool(r["k_increased"])
        if r["k_increased"]:
            assert r["phase"] == PHASE_T and not r["vprime_expanded"] and not r["tprime_perfect"]
        if i + 1 < len(rows):
            assert rows[i + 1]["k_per_class"] == r["k_after"]
        assert r["k_increases_total"] == sum(1 for x in rows[: i + 1] if x["k_increased"])
    ctl = PhaseSwitchController(_settings(), k_start=3)
    r = ctl.on_tprime(fitness=0.8, k_next=9, sv_signal=True, vprime_expandable=True)
    assert r["k_before"] == 3 and r["k_after"] == 9 and ctl.k_increases == 1  # jedno przejście 3 -> 9, nie kilka


def test_req06_vprime_expands_at_most_once_per_cycle(tmp_path, data):
    out = _run(tmp_path, data, _cfg(selection="evolutionary"), "r6")
    rows = out["cycle_table_rows"]
    prev_total = 0
    for r in rows:
        grew = r["vprime_rows_after"] - r["vprime_rows_before"]
        if r["phase"] == PHASE_T:
            assert grew == 0 and not r["vprime_expanded"]
        else:
            assert grew == r["delta_rows"] and 0 < grew <= 5 * 2  # przyrost <= vprime_increment_per_class × klasy
            assert r["vprime_expanded"] and not r["k_increased"]
        assert r["vprime_expansions_total"] - prev_total in (0, 1)
        prev_total = r["vprime_expansions_total"]
    assert out["phase_study"]["vprime_expansions"] == prev_total == len(_phase_rows(out, PHASE_V))


def test_req08_integration_failed_retry_stops_only_after_k_was_increased(tmp_path, data):
    """fitness_min_delta=1,0 => żadne zwiększenie K nigdy nie „poprawia” fitnessu: wymusza zastój, próbę V' i jej porażkę."""
    out = _run(tmp_path, data, _cfg(selection="evolutionary", fitness_min_delta=1.0, max_no_improvement_k_increases=2,
                                    k_growth_requires_sv_signal=False, initial_vprime_per_class=12,
                                    max_vprime_fraction=0.3), "r8")
    rows = out["cycle_table_rows"]
    ps = out["phase_study"]
    assert ps["stop_decision"] != "stop_max_cycles"
    retries = [i for i, r in enumerate(rows) if r["phase"] == PHASE_V and r["retry_phase"]]
    assert retries, "oczekiwano próby V' po zastoju"
    if ps["stop_decision"] == "stop_no_progress_after_vprime_retry":
        j = max(i for i in retries)  # ostatnia próba V'
        after = rows[j + 1:]
        assert after[-1]["retry_check"] and not after[-1]["retry_progress"]
        assert sum(1 for r in after if r["k_increased"]) >= 1   # „nawet po zwiększeniu K”
        assert after[-1]["retry_fail_count"] >= 2 or not after[-1]["k_increased"]
    # na jeden poziom K najwyżej jedna próba V' po zastoju
    retry_k = [rows[i - 1]["k_per_class"] for i in retries if rows[i - 1]["phase"] == PHASE_T]
    assert len(retry_k) == len(set(retry_k))


def test_req09_algorithm_never_alternates_forever(tmp_path, easy_data):
    # T' zawsze opanowuje V' (najgorszy scenariusz dla pętli T' <-> V'), a fazy V' są długie
    out = _run(tmp_path, easy_data, _easy_cfg(vprime_difficulty_patience=50, max_vprime_expansions_per_phase=3,
                                              max_cycles=500), "r9a")
    assert out["phase_study"]["stop_decision"] == "stop_perfect_on_max_vprime"
    assert len(out["cycle_table_rows"]) < 500
    # globalny limit rozszerzeń
    out = _run(tmp_path, easy_data, _easy_cfg(max_vprime_expansions=3, max_vprime_fraction=0.9), "r9b")
    ps = out["phase_study"]
    assert ps["vprime_expansions"] == 3 and ps["stop_decision"] == "stop_perfect_on_max_vprime"
    assert out["cycle_table_rows"][-1]["vprime_limit_reason"] == "expansion_limit"
    assert ps["vprime_cap_rows"] > ps["vprime_rows"]  # zatrzymał limit rozszerzeń, nie rozmiar
    # bezpiecznik liczby cykli
    out = _run(tmp_path, easy_data, _easy_cfg(max_cycles=4, max_vprime_fraction=0.9), "r9c")
    assert out["phase_study"]["stop_decision"] == "stop_max_cycles" and len(out["cycle_table_rows"]) == 4


def _garbage_observer(X, y):
    rng = np.random.default_rng(123)
    return FullValidationObserver(
        X, y, "balanced_accuracy",
        lambda clf, X_, y_, metric: {"metric": float(rng.choice([0.0, 1.0, rng.random()])), "all": {"accuracy": 0.5}})


@pytest.mark.parametrize("selection", ["random", "evolutionary", "heuristic"])
def test_req10_neither_full_validation_nor_test_influences_decisions(tmp_path, data, selection):
    base = _run(tmp_path, data, _cfg(selection=selection), f"r10base{selection}")
    # obserwator pełnego V zwraca losowe śmieci (0 / 1 / losowo), a etykiety testu są odwrócone
    noisy = _run(tmp_path, data, _cfg(selection=selection), f"r10noisy{selection}", observer_factory=_garbage_observer,
                 y_test_override=lambda y: 1 - y)
    assert _traj(base) == _traj(noisy)
    assert [r["decision"] for r in base["cycle_table_rows"]] == [r["decision"] for r in noisy["cycle_table_rows"]]
    assert [r["observed_full_validation_metric"] for r in base["cycle_table_rows"]] != \
           [r["observed_full_validation_metric"] for r in noisy["cycle_table_rows"]]  # obserwacje faktycznie się różnią
    ps = base["phase_study"]
    assert ps["used_full_validation_for_control"] is False and ps["used_test_for_control"] is False


def test_req11_vprime_fraction_is_vprime_over_the_full_validation_set(tmp_path, data):
    out = _run(tmp_path, data, _cfg(selection="evolutionary"), "r11")
    n_val = data[2].shape[0]
    ids = _ids(tmp_path, "r11")
    ps = out["phase_study"]
    assert ps["full_validation_rows"] == n_val and ps["vprime_rows"] == ids.size
    assert ps["vprime_fraction"] == pytest.approx(ids.size / n_val)
    for r in out["cycle_table_rows"]:
        assert r["full_validation_rows"] == n_val
        assert r["vprime_fraction"] == pytest.approx(r["vprime_rows_after"] / n_val)
        assert 0.0 < r["vprime_fraction"] <= 1.0


def test_req12_random_and_evolutionary_share_start_schedule_limit_and_stop_rule(tmp_path, data):
    import pandas as pd

    ref = _run(tmp_path, data, _cfg(selection="evolutionary"), "r12ref")
    csv = tmp_path / "r12ref.csv"
    pd.DataFrame(ref["cycle_table_rows"])[PHASE_SWITCH_COLUMNS].to_csv(csv, index=False)
    rep = _run(tmp_path, data, _cfg(selection="random", schedule_replay=str(csv)), "r12rep")
    a, b = ref["cycle_table_rows"], rep["cycle_table_rows"]
    n0 = a[0]["vprime_rows_before"]
    assert np.array_equal(_ids(tmp_path, "r12ref")[:n0], _ids(tmp_path, "r12rep")[:n0])  # to samo V' startowe
    for key in ("phase", "k_per_class", "k_after", "vprime_rows_before", "vprime_rows_after", "vprime_cap_rows",
                "full_validation_rows", "delta_rows", "expansions_in_phase", "vprime_phase_id"):
        assert [r[key] for r in a] == [r[key] for r in b], key
    assert rep["phase_study"]["vprime_rows"] == ref["phase_study"]["vprime_rows"]
    assert rep["phase_study"]["reference_stop_decision"] == ref["phase_study"]["stop_decision"]
    # ten sam podział, seed i parametry automatu — różni się tylko strategia doboru ΔV'
    assert rep["phase_study"]["vprime"]["selection"] == "random" and ref["phase_study"]["vprime"]["selection"] == "evolutionary"
    assert {k: v for k, v in rep["phase_study"]["settings"].items()} == {k: v for k, v in ref["phase_study"]["settings"].items()}


def test_req13_vprime_growth_stops_before_the_whole_validation_set(tmp_path, easy_data):
    out = _run(tmp_path, easy_data, _easy_cfg(max_vprime_fraction=0.1, vprime_difficulty_patience=100,
                                              max_vprime_expansions_per_phase=1000), "r13")
    ps = out["phase_study"]
    assert ps["vprime_rows"] == ps["vprime_cap_rows"] == vprime_cap_rows(ps["full_validation_rows"], 0.1, None)
    assert ps["vprime_rows"] < ps["full_validation_rows"] and ps["vprime_fraction"] <= 0.1
    assert out["cycle_table_rows"][-1]["vprime_limit_reason"] == "size_limit" and ps["vprime_cap_reached"]
    # rozmiar nie zależy od K ani od ρ_SV: K stałe, a V' i tak rośnie do limitu
    assert {r["k_per_class"] for r in out["cycle_table_rows"]} == {3}


def test_req14_old_mode_dagasvm_phase_still_runs_through_the_cli(tmp_path):
    import pandas as pd
    import yaml

    X, y = make_moons(n_samples=600, noise=0.2, random_state=42)
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    ddir = tmp_path / "data"
    ddir.mkdir()
    for name, part in {"train": frame.iloc[:250], "validation": frame.iloc[250:450], "test": frame.iloc[450:]}.items():
        part.to_csv(ddir / f"{name}.csv", index=False)
    cfg = {
        "seed": 42, "experiment_name": "cli_old_phase", "dataset_name": "moons-test", "algorithm_mode": "dagasvm_phase",
        "data": {"train_csv": str(ddir / "train.csv"), "validation_csv": str(ddir / "validation.csv"),
                 "test_csv": str(ddir / "test.csv"), "target": "label", "stratify": True, "header": True,
                 "target_candidates": ["label"], "max_rows": 0},
        "svm": {"kernel": "rbf", "C": 10.0, "gamma": 1.0, "class_weight": "balanced", "probability": False,
                "standardize": False},
        "fitness": {"metric": "balanced_accuracy", "class_imbalance_penalty": 0.0},
        "ga": {"population_size": 6, "generations": 3, "tournament_k": 3, "crossover_rate": 0.8, "mutation_rate": 0.05,
               "elitism": 2},
        "daga": {"k_start": 3, "TSV": 0.25, "Ts": 2, "phase_min_delta": 0.0005, "max_cycles": 40},
        "val_ga": {"population_size": 6, "generations": 3, "mutation_rate": 0.1, "elitism": 2},
        "vprime": {"selection": "random", "memory": True, "mode": "growing"},
        "observation": {"full_validation": True, "test": True},
        "artifacts_root": str(tmp_path / "artifacts"),
    }
    cfg_path = tmp_path / "old.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, str(repo / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
                          capture_output=True, text=True, cwd=str(repo))
    assert proc.returncode == 0, proc.stderr[-2000:]
    run_dir = next((tmp_path / "artifacts" / "runs").iterdir())
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    cycles = pd.read_csv(run_dir / "metrics" / "cycles.csv")
    assert summary["algorithm_mode"] == "dagasvm_phase"
    # semantyka starego trybu: decyzje cyklu, brak kolumn nowego automatu
    assert {"decision", "perfect_on_current_val", "stable_cycles_at_current_k", "perfect_gate_pending"} <= set(cycles.columns)
    assert "phase" not in cycles.columns and "prev_phase" not in cycles.columns
    assert set(cycles["decision"].dropna()) <= {"expand_validation_only", "increase_k", "stop_perfect_on_current_val",
                                                "stop_no_validation_candidates", "stop_no_k_growth",
                                                "stop_no_growth_signal", "stop_max_cycles"}
    assert summary["phase_study"]["stop_category"] is not None and "stop_rule" in summary["phase_study"]


# ============================================================================================
# 8. Koszt: liczniki wywołań, nominalny vs faktyczny, nowe ustawienia
# ============================================================================================
def test_counting_model_counts_calls_and_rows_and_delegates():
    X = np.random.default_rng(0).normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = SVC(kernel="rbf", probability=False).fit(X, y)
    m = CountingModel(clf)
    assert np.array_equal(m.predict(X[:10]), clf.predict(X[:10]))
    m.predict(X[:5])
    m.decision_function(X[:7])
    assert m.snapshot() == {"predict_calls": 2, "predict_rows": 15, "decision_function_calls": 1,
                            "decision_function_rows": 7}
    assert hasattr(m, "n_support_") and not hasattr(m, "predict_proba")  # SVC(probability=False) nie ma predict_proba


@pytest.mark.parametrize("selection", ["random", "evolutionary", "heuristic"])
def test_cost_counters_equal_the_real_number_of_model_calls(tmp_path, data, monkeypatch, selection):
    calls = {"predict": 0, "decision_function": 0}
    orig_predict, orig_decision = SVC.predict, SVC.decision_function

    def predict(self, X):
        calls["predict"] += 1
        return orig_predict(self, X)

    def decision_function(self, X):
        calls["decision_function"] += 1
        return orig_decision(self, X)

    monkeypatch.setattr(SVC, "predict", predict)
    monkeypatch.setattr(SVC, "decision_function", decision_function)
    out = _run(tmp_path, data, _cfg(selection=selection), f"cost{selection}")
    rows = out["cycle_table_rows"]
    n_t = len(_phase_rows(out, PHASE_T))
    # algorytm + obserwacja (po jednym `predict` na pełnym V i na teście w każdym kroku T')
    assert calls["predict"] == sum(r["predict_calls"] for r in rows) + 2 * n_t
    assert calls["decision_function"] == sum(r["decision_function_calls"] for r in rows)
    ps_cost = out["phase_study"]["cost"]
    assert ps_cost["total"]["predict_calls"] == sum(r["predict_calls"] for r in rows)
    assert ps_cost["tprime_phase"]["predict_calls"] + ps_cost["vprime_phase"]["predict_calls"] == ps_cost["total"]["predict_calls"]
    # nominalny >= faktyczny koszt oceny T' (wcześniejsze zatrzymanie GA); treningi = ewaluacje GA + 1 trening końcowy/krok T'
    for r in _phase_rows(out, PHASE_T):
        assert r["tprime_eval_rows_actual"] <= r["tprime_eval_rows_nominal"]
        assert r["tprime_eval_rows_actual"] == r["ga_svm_evaluations"] * r["vprime_rows_before"]
        assert r["svm_trainings"] == r["ga_svm_evaluations"] + 1
    assert all(r["svm_trainings"] == 0 for r in _phase_rows(out, PHASE_V))
    if selection == "heuristic":
        assert all(r["decision_function_rows"] >= r["n_candidates"] for r in _phase_rows(out, PHASE_V))
    assert ps_cost["observation_rows"] > 0


def test_expansion_block_reason_and_expansion_limit():
    kw = dict(vprime_rows=20, cap_rows=100, n_candidates=500, n_classes=2)
    assert expansion_block_reason(**kw) is None
    assert expansion_block_reason(**{**kw, "vprime_rows": 99}) == "size_limit"
    assert expansion_block_reason(**{**kw, "n_candidates": 1}) == "no_candidates"
    assert expansion_block_reason(**kw, expansions_left=0) == "expansion_limit"
    assert plan_increment_per_class(**kw, increment_per_class=10, expansions_left=0) == 0
    assert plan_increment_per_class(**kw, increment_per_class=10, expansions_left=2) == 10


def test_new_settings_defaults_and_validation():
    st = parse_phase_switch_settings({"vprime": {"selection": "random"}})
    assert st.vprime_difficulty_metric == "error_rate" and st.max_vprime_expansions is None
    assert st.retry_resolution == "after_k_increases" and st.retry_reference == "same_vprime"
    assert st.max_no_improvement_k_increases == 3 and st.allow_one_vprime_retry_after_t_stall
    assert st.fitness_min_delta == 0.0005 and st.vprime.score_candidate_pool is False
    for bad in ({"vprime_difficulty_metric": "auc"}, {"retry_resolution": "never"}, {"retry_reference": "x"},
                {"max_vprime_expansions": 0}):
        with pytest.raises(ValueError):
            parse_phase_switch_settings({"phase_switch": bad, "vprime": {"selection": "random"}})


@pytest.mark.parametrize("metric", ["error_rate", "one_minus_metric"])
def test_difficulty_metric_option_controls_the_phase_end_signal(tmp_path, data, metric):
    out = _run(tmp_path, data, _cfg(selection="evolutionary", vprime_difficulty_metric=metric), f"diff{metric}")
    v_rows = _phase_rows(out, PHASE_V)
    assert v_rows
    for r in v_rows:
        expected = r["delta_error_rate"] if metric == "error_rate" else 1.0 - r["delta_metric"]
        assert r["delta_difficulty"] == pytest.approx(expected)


def test_multiclass_with_binary_f1_selection_metric_is_rejected(tmp_path):
    from sklearn.datasets import make_blobs as mb

    X, y = mb(n_samples=300, centers=3, random_state=0)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.4, stratify=y, random_state=0)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=0.7, stratify=y_rest, random_state=1)
    cfg = _cfg(selection="random")
    cfg["fitness"] = {"tprime_metric": "balanced_accuracy", "vprime_metric": "f1"}
    with pytest.raises(ValueError, match="binarne"):
        _run(tmp_path, (X_tr, y_tr, X_va, y_va, X_te, y_te), cfg, "mc")


def test_multiclass_pr_auc_is_rejected_instead_of_silently_becoming_ba(tmp_path):
    X, y = make_blobs(n_samples=300, centers=3, random_state=0)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.4, stratify=y, random_state=0)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=0.7, stratify=y_rest, random_state=1)
    cfg = _cfg(selection="random")
    cfg["fitness"] = {"tprime_metric": "pr_auc", "vprime_metric": "balanced_accuracy"}
    svm_prob = SVMConfig(kernel="rbf", C=10.0, gamma=1.0, probability=True, standardize=False)
    layout = ensure_artifact_layout(tmp_path / "pr", experiment_name="pr")
    with pytest.raises(ValueError, match="pr_auc"):
        run_phase_switch_experiment(X_tr, y_tr, X_va, y_va, X_te, y_te, svm_prob, cfg, get_logger(), layout)


def test_cli_objective_metric_name_follows_tprime_metric(tmp_path):
    import pandas as pd
    import yaml

    X, y = make_moons(n_samples=500, noise=0.2, random_state=1)
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    ddir = tmp_path / "data"
    ddir.mkdir()
    for name, part in {"train": frame.iloc[:200], "validation": frame.iloc[200:380], "test": frame.iloc[380:]}.items():
        part.to_csv(ddir / f"{name}.csv", index=False)
    cfg = _cfg(selection="random", max_vprime_fraction=0.2)
    cfg.update({
        "experiment_name": "cli_metric", "dataset_name": "moons-test", "algorithm_mode": "dagasvm_phase_switch",
        "data": {"train_csv": str(ddir / "train.csv"), "validation_csv": str(ddir / "validation.csv"),
                 "test_csv": str(ddir / "test.csv"), "target": "label", "stratify": True, "header": True,
                 "target_candidates": ["label"], "max_rows": 0},
        "svm": {"kernel": "rbf", "C": 10.0, "gamma": 1.0, "class_weight": "balanced", "probability": False,
                "standardize": False},
        "fitness": {"tprime_metric": "f1_macro", "vprime_metric": "balanced_accuracy"},
        "artifacts_root": str(tmp_path / "artifacts"),
    })
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, str(repo / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
                          capture_output=True, text=True, cwd=str(repo))
    assert proc.returncode == 0, proc.stderr[-2000:]
    run_dir = next((tmp_path / "artifacts" / "runs").iterdir())
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["objective_metric_name"] == "f1_macro"
    cycles = pd.read_csv(run_dir / "metrics" / "cycles.csv")
    assert set(cycles["tprime_metric"].dropna()) == {"f1_macro"}


def test_rejected_minimum_size_plus_recheck_rule_is_not_used_in_phase_switch_mode(tmp_path, data):
    """Promotor odrzucił regułę „minimalny rozmiar V' + obowiązkowe ponowne sprawdzenie”: parametry starego trybu
    (daga.min_vprime_per_class, perfect_recheck_*) nie mogą w tym trybie niczego zmieniać."""
    base = _run(tmp_path, data, _cfg(selection="random"), "min0")
    cfg = _cfg(selection="random")
    cfg["daga"].update(min_vprime_per_class=50, perfect_recheck_expansions=3, perfect_recheck_min_add_per_class=20)
    gated = _run(tmp_path, data, cfg, "min1")
    assert _traj(base) == _traj(gated)
    assert base["cycle_table_rows"][0]["vprime_rows_before"] == 6  # V' startowe = 2·k_start, bez dobijania do minimum
    first_v = next(r for r in base["cycle_table_rows"] if r["phase"] == PHASE_V)
    assert first_v["vprime_rows_before"] < 50  # faza V' wynika z BA(V')=1 / zastoju, nie z progu rozmiaru
