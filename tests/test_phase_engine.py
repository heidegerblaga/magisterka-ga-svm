"""Testy fazowego DAGASVM: rozdział ról V/V', polityka K, pamięć V' i powtarzalność seedów."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.models.svm import SVMConfig
from ga.phase_engine import DECISION_EXPAND, DECISION_INCREASE_K, run_phase_experiment
from ga.utils.artifacts import ensure_artifact_layout
from ga.utils.logger import get_logger
from ga.vprime import FullValidationObserver


@pytest.fixture(scope="module")
def data():
    X, y = make_moons(n_samples=2400, noise=0.28, random_state=5)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.4, stratify=y, random_state=5)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=0.7, stratify=y_rest, random_state=6)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _cfg(seed=42, **vprime):
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
        "daga": {"k_start": 4, "TSV": 0.25, "Ts": 2, "phase_min_delta": 0.0005, "max_cycles": 40},
        "val_ga": {"population_size": 6, "generations": 3, "mutation_rate": 0.1, "elitism": 2},
        "vprime": {"selection": "heuristic", "memory": True, "mode": "growing", **vprime},
        "observation": {"full_validation": True, "test": True},
    }


def _run(tmp_path, data, cfg, tag, observer_factory=None):
    X_tr, y_tr, X_va, y_va, X_te, y_te = data
    layout = ensure_artifact_layout(tmp_path / tag, experiment_name=tag)
    svm = SVMConfig(kernel="rbf", C=1.0, gamma=1.0, probability=False, standardize=False)
    observer = None if observer_factory is None else observer_factory(X_va, y_va)
    out = run_phase_experiment(
        X_tr, y_tr, X_va, y_va, X_te, y_te, svm, cfg, get_logger(), layout, observer=observer
    )
    return out, Path(layout["checkpoints"])


def _trajectory(out):
    return [
        (r["cycle"], r["k_per_class"], r["k_after"], r["decision"], r["val_rows_before"], r["val_rows_after"])
        for r in out["cycle_table_rows"]
    ]


def _ids(checkpoints, out):
    return [np.load(checkpoints / f"cycle_{r['cycle']:02d}_vprime_ids.npy") for r in out["cycle_table_rows"]]


class _PoisonedObserver(FullValidationObserver):
    """Obserwator zwracający losowe 'metryki' — gdyby algorytm je czytał, przebieg by się zmienił."""

    def __init__(self, X, y):
        super().__init__(X, y, "balanced_accuracy", None, enabled=True)
        self._rng = np.random.default_rng(123)

    def observe(self, clf):
        self.calls += 1
        value = float(self._rng.uniform(0, 1))
        return {"metric": value, "all": {"accuracy": value}, "score_time": 0.0, "num_support_vectors": 0}


@pytest.mark.parametrize("selection", ["heuristic", "evolutionary", "random"])
def test_full_validation_never_controls_the_run(tmp_path, data, selection):
    cfg = _cfg(selection=selection)
    on, ck_on = _run(tmp_path, data, cfg, "obs_on")
    off_cfg = copy.deepcopy(cfg)
    off_cfg["observation"]["full_validation"] = False
    off, ck_off = _run(tmp_path, data, off_cfg, "obs_off")
    poisoned, ck_poison = _run(tmp_path, data, cfg, "obs_poison", observer_factory=_PoisonedObserver)

    assert _trajectory(on) == _trajectory(off) == _trajectory(poisoned)
    for a, b, c in zip(_ids(ck_on, on), _ids(ck_off, off), _ids(ck_poison, poisoned)):
        assert np.array_equal(a, b) and np.array_equal(a, c)
    assert off["cycle_table_rows"][0]["train_ga_best_full_validation_metric"] is None
    assert on["cycle_table_rows"][0]["train_ga_best_full_validation_metric"] is not None
    assert on["phase_study"]["used_full_validation_for_control"] is False
    assert all(c["used_full_validation_for_control"] is False for c in on["cycles"])


def test_k_increases_at_most_once_per_cycle_and_never_when_perfect(tmp_path, data):
    out, _ = _run(tmp_path, data, _cfg(selection="heuristic"), "kpolicy")
    rows = out["cycle_table_rows"]
    assert len(rows) >= 3
    previous_k_after = None
    for row in rows:
        assert row["k_increases_this_cycle"] in (0, 1)
        if previous_k_after is not None:
            assert row["k_per_class"] == previous_k_after  # K jest stałe w obrębie cyklu T'
        previous_k_after = row["k_after"]
        if row["decision"] == DECISION_INCREASE_K:
            assert row["k_after"] > row["k_per_class"] and row["k_increases_this_cycle"] == 1
            assert not row["perfect_on_current_val"]
        else:
            assert row["k_after"] == row["k_per_class"] and row["k_increases_this_cycle"] == 0
    assert out["phase_study"]["max_k_increases_in_one_cycle"] <= 1
    assert all(r["decision_reason"] for r in rows)


def test_vprime_is_expanded_before_first_k_increase(tmp_path, data):
    out, _ = _run(tmp_path, data, _cfg(selection="heuristic"), "order")
    decisions = [r["decision"] for r in out["cycle_table_rows"]]
    if DECISION_INCREASE_K in decisions:
        assert DECISION_EXPAND in decisions[: decisions.index(DECISION_INCREASE_K)]


@pytest.mark.parametrize("selection", ["heuristic", "evolutionary", "random"])
def test_memory_option_controls_carry_over_of_vprime_ids(tmp_path, data, selection):
    X_va = data[2]
    on, ck_on = _run(tmp_path, data, _cfg(selection=selection, memory=True), "mem_on")
    off, ck_off = _run(tmp_path, data, _cfg(selection=selection, memory=False), "mem_off")
    for out, ck, memory in ((on, ck_on, True), (off, ck_off, False)):
        ids_per_cycle = _ids(ck, out)
        expansions = 0
        for prev, curr, row in zip(ids_per_cycle, ids_per_cycle[1:], out["cycle_table_rows"][1:]):
            assert np.unique(curr).size == curr.size  # brak duplikatów
            assert curr.min() >= 0 and curr.max() < len(X_va)
        for i, row in enumerate(out["cycle_table_rows"]):
            after = ids_per_cycle[i]
            before = ids_per_cycle[i - 1] if i else None
            if row["decision"] == DECISION_EXPAND and before is not None:
                expansions += 1
                if memory:
                    assert np.array_equal(after[: before.size], before)  # pamięć: stare ID zachowane
                    assert after.size >= before.size
                else:
                    assert np.intersect1d(after, before).size == 0  # bez pamięci: stare ID odrzucone
            elif before is not None:
                assert np.array_equal(after, before)  # inne decyzje nie zmieniają V'
        assert expansions > 0 or len(out["cycle_table_rows"]) <= 2


def test_fixed_random_vprime_has_constant_size_and_no_expansion(tmp_path, data):
    out, ck = _run(
        tmp_path, data, _cfg(selection="random", mode="fixed", fixed_size=120), "fixed"
    )
    rows = out["cycle_table_rows"]
    assert {r["val_rows_before"] for r in rows} == {120} == {r["val_rows_after"] for r in rows}
    assert all(r["decision"] != DECISION_EXPAND for r in rows)
    ids = _ids(ck, out)
    assert all(np.array_equal(ids[0], i) for i in ids)
    y_va = data[3]
    assert abs(float((y_va[ids[0]] == 1).mean()) - float((y_va == 1).mean())) < 0.02  # stratyfikacja


def test_same_seed_is_reproducible_and_different_seed_differs(tmp_path, data):
    a, ck_a = _run(tmp_path, data, _cfg(seed=7, selection="evolutionary"), "seed_a")
    b, ck_b = _run(tmp_path, data, _cfg(seed=7, selection="evolutionary"), "seed_b")
    c, ck_c = _run(tmp_path, data, _cfg(seed=8, selection="evolutionary"), "seed_c")
    assert _trajectory(a) == _trajectory(b)
    for x, y in zip(_ids(ck_a, a), _ids(ck_b, b)):
        assert np.array_equal(x, y)
    assert [r["train_ga_best_val_metric"] for r in a["cycle_table_rows"]] == [
        r["train_ga_best_val_metric"] for r in b["cycle_table_rows"]
    ]
    # inne ziarno => inna początkowa V' => inne identyfikatory
    assert not np.array_equal(_ids(ck_a, a)[0], _ids(ck_c, c)[0])
