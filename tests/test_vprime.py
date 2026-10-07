import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.loop.engine import GAConfig, ValEncodingConfig, run_val_ga
from ga.models.svm import SVMConfig, evaluate_fitted_svm, fit_svm_model
from ga.phase_engine import decide_cycle
from ga.vprime import (
    VPrimeConfig,
    VPrimeSet,
    class_balanced_ids,
    expand_vprime,
    stratified_random_ids,
)


@pytest.fixture(scope="module")
def pool():
    X, y = make_moons(n_samples=1200, noise=0.3, random_state=3)
    clf = SVC(kernel="rbf", C=1.0, gamma=1.0, random_state=0).fit(X[:30], y[:30])
    return X, y, clf


def _expand(vset, pool, *, memory, strategy, n_add=10, seed=0):
    X, y, clf = pool
    return expand_vprime(
        vset,
        memory=memory,
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
        counter={"model_predictions": 0, "diagnostic_predictions": 0},
    )


def test_vprime_set_rejects_duplicates_and_out_of_range():
    vset = VPrimeSet(10, [1, 2, 3])
    with pytest.raises(ValueError):
        vset.extend([3, 4])  # 3 już jest w V'
    with pytest.raises(ValueError):
        vset.replace([1, 1])
    with pytest.raises(ValueError):
        vset.replace([11])
    assert vset.ids.tolist() == [1, 2, 3]


@pytest.mark.parametrize("strategy", ["random", "heuristic", "evolutionary"])
def test_memory_on_keeps_previous_ids_and_adds_unique_new_ones(pool, strategy):
    X, y, _ = pool
    vset = VPrimeSet(len(y), class_balanced_ids(y, 5, np.random.default_rng(0)))
    history = [vset.ids]
    for step in range(4):
        res = _expand(vset, pool, memory=True, strategy=strategy, seed=step)
        history.append(vset.ids)
        assert res.n_dropped == 0
        assert res.n_new == res.size_after - res.size_before > 0
        assert np.unique(vset.ids).size == vset.size  # brak duplikatów
    for older, newer in zip(history, history[1:]):
        # identyfikatory zachowane i w tej samej kolejności (prefiks)
        assert np.array_equal(newer[: older.size], older)
        assert newer.size > older.size
    # identyfikatory wskazują te same próbki co w pełnym V
    assert np.all(X[history[-1]] == X[np.asarray(history[-1])])


@pytest.mark.parametrize("strategy", ["random", "heuristic", "evolutionary"])
def test_memory_off_discards_previous_ids_with_same_size_schedule(pool, strategy):
    _, y, _ = pool
    start = class_balanced_ids(y, 5, np.random.default_rng(0))
    mem_on, mem_off = VPrimeSet(len(y), start), VPrimeSet(len(y), start)
    for step in range(3):
        before = mem_off.ids
        _expand(mem_on, pool, memory=True, strategy=strategy, seed=step)
        res = _expand(mem_off, pool, memory=False, strategy=strategy, seed=step)
        assert np.intersect1d(mem_off.ids, before).size == 0  # stare elementy odrzucone
        assert res.n_dropped == before.size
        assert np.unique(mem_off.ids).size == mem_off.size
        if strategy != "heuristic":  # heurystyka jest zbalansowana klasowo (zaokrąglenia ±1)
            assert mem_off.size == mem_on.size


def test_stratified_random_ids_preserve_class_proportions_and_are_reproducible():
    y = np.array([0] * 700 + [1] * 300)
    a = stratified_random_ids(y, 100, np.random.default_rng(1))
    b = stratified_random_ids(y, 100, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert a.size == 100 and np.unique(a).size == 100
    assert int((y[a] == 1).sum()) == 30


def test_heuristic_prefers_model_errors(pool):
    X, y, clf = pool
    vset = VPrimeSet(len(y), class_balanced_ids(y, 3, np.random.default_rng(0)))
    candidates = vset.candidates()
    errors_available = int((clf.predict(X[candidates]) != y[candidates]).sum())
    assert errors_available > 20
    res = _expand(vset, pool, memory=True, strategy="heuristic", n_add=20)
    assert res.diagnostics["selected_error_rate"] == pytest.approx(1.0)


def test_fast_metric_path_matches_full_path(pool):
    X, y, _ = pool
    clf = fit_svm_model(X[:100], y[:100], SVMConfig(C=1.0, gamma=1.0, probability=False, standardize=False))
    for metric in ("balanced_accuracy", "accuracy"):
        full = evaluate_fitted_svm(clf, X[100:], y[100:], metric=metric)
        fast = evaluate_fitted_svm(clf, X[100:], y[100:], metric=metric, all_metrics=False)
        assert fast["metric"] == pytest.approx(full["metric"], abs=0, rel=0)
        assert fast["num_support_vectors"] == full["num_support_vectors"]


def test_val_ga_with_cached_predictions_is_identical(pool):
    X, y, clf = pool
    gcfg = GAConfig(population_size=10, generations=5, mutation_rate=0.1)
    vcfg = ValEncodingConfig(subset_fraction=0.05)
    h1, m1 = run_val_ga(X, y, clf, vcfg, gcfg, seed=11, verbose=False)
    h2, m2 = run_val_ga(X, y, clf, vcfg, gcfg, seed=11, preds=clf.predict(X), verbose=False)
    assert np.array_equal(h1, h2) and np.array_equal(m1, m2)
    # fitness najlepszego osobnika = 1 − BA ustalonego modelu na wybranym podzbiorze
    assert 1 - balanced_accuracy_score(y[m1], clf.predict(X[m1])) == pytest.approx(h1.max(), abs=1e-12)


def test_vprime_config_validates_fixed_mode():
    with pytest.raises(ValueError):
        VPrimeConfig(selection="evolutionary", mode="fixed", fixed_size=100)
    with pytest.raises(ValueError):
        VPrimeConfig(selection="random", mode="fixed", fixed_size=None)
    with pytest.raises(ValueError):
        VPrimeConfig(selection="magic")


# --- polityka decyzji K / V' -------------------------------------------------------------


def _decide(**overrides):
    base = dict(
        perfect_on_current_val=False,
        has_growth_signal=True,
        expansion_possible=True,
        vprime_mode="growing",
        validation_hardened_for_current_k=False,
        sv_ratio_stable_since_hardening=False,
        can_increase_k=True,
    )
    base.update(overrides)
    return decide_cycle(**base)


def test_perfect_on_vprime_never_increases_k():
    for hardened in (False, True):
        for possible in (False, True):
            d = _decide(
                perfect_on_current_val=True,
                validation_hardened_for_current_k=hardened,
                expansion_possible=possible,
            )
            assert not d.increases_k
    assert _decide(perfect_on_current_val=True).expands_vprime  # najpierw utrudniam V'
    assert _decide(perfect_on_current_val=True, validation_hardened_for_current_k=True).decision == (
        "stop_perfect_on_current_val"
    )


def test_vprime_is_hardened_before_k_grows():
    assert _decide().expands_vprime
    assert _decide(validation_hardened_for_current_k=True).increases_k
    assert _decide(validation_hardened_for_current_k=True, sv_ratio_stable_since_hardening=True).expands_vprime


def test_fixed_vprime_grows_k_directly_and_stops_without_signal():
    d = _decide(vprime_mode="fixed", expansion_possible=False)
    assert d.increases_k
    assert _decide(vprime_mode="fixed", expansion_possible=False, has_growth_signal=False).stops
