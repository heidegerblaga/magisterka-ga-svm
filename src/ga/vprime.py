"""Zarządzanie podzbiorem walidacyjnym V' w fazowym DAGASVM.

Terminologia (zgodnie z resztą projektu):
  - V  — pełny zbiór walidacyjny (``validation.csv``). Pełni dwie role:
         (a) PULA KANDYDATÓW, z której wybierane są nowe elementy V';
         (b) źródło metryk OBSERWACYJNYCH (nigdy nie steruje decyzjami algorytmu).
  - V' — aktualny podzbiór walidacyjny; to na nim liczony jest fitness T'.
  - T' — aktualny podzbiór treningowy (k_per_class próbek na klasę).
  - K  — ``k_per_class``: liczba próbek na klasę w T'.

Elementy V' są identyfikowane przez indeks wiersza w pełnym V (``sample id``).
Moduł dba o to, żeby V' było zbiorem BEZ duplikatów, a identyfikatory przetrwały
przejście między fazami (opcja ``memory``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ga.loop.engine import GAConfig, ValEncodingConfig, run_val_ga

SELECTION_STRATEGIES = ("heuristic", "evolutionary", "random")
# liczniki kosztu doboru V' (wiersze faktycznie przekazane do modelu); nominalne wartości Val-GA są osobno w diagnostyce
_COST_KEYS = ("selection_predict_rows", "selection_decision_rows", "diagnostic_scored_rows")
VPRIME_MODES = ("growing", "fixed")


@dataclass
class VPrimeConfig:
    """Konfiguracja doboru V'.

    selection: sposób wyboru NOWYCH elementów V'
      - ``heuristic``    — błędy aktualnego modelu, potem najmniejszy margines
                           (wymaga sklasyfikowania całej puli kandydatów),
      - ``evolutionary`` — adwersarialny Val-GA (``run_val_ga``) nad maskami kandydatów,
      - ``random``       — losowo, bez informacji o modelu.
    memory: czy elementy wybrane w poprzednich fazach pozostają w V'.
      - True  — V'_{t+1} = V'_t ∪ Δ_t (Δ_t z puli V \\ V'_t),
      - False — V'_{t+1} = świeży wybór o rozmiarze |V'_t| + |Δ_t| z puli V \\ V'_t
                (poprzednie elementy są odrzucane; harmonogram rozmiaru jak przy memory=True).
    mode: ``growing`` (V' rozszerzane w kolejnych cyklach) albo ``fixed`` (V' wylosowane
          raz na starcie o rozmiarze ``fixed_size`` i potem nie zmieniane — wariant bazowy).
    pool_budget_factor: opcjonalny, WSPÓLNY dla wszystkich strategii limit budżetu doboru. Przy każdym
          rozszerzeniu strategia widzi tylko losową (stratyfikowaną klasowo) podpulę kandydatów o rozmiarze
          co najwyżej ``pool_budget_factor × (liczba wybieranych elementów)``; podpula nie zależy od
          strategii ani od modelu. ``None`` (domyślnie) = brak limitu = zachowanie historyczne.
    """

    selection: str = "heuristic"
    memory: bool = True
    mode: str = "growing"
    fixed_size: int | None = None
    score_candidate_pool: bool = True  # diagnostyka: błędność całej puli (nie steruje algorytmem)
    pool_budget_factor: float | None = None

    def __post_init__(self) -> None:
        if self.pool_budget_factor is not None and float(self.pool_budget_factor) < 1.0:
            raise ValueError("vprime.pool_budget_factor musi wynosić co najmniej 1 (pula >= liczba wybieranych elementów).")
        if self.selection not in SELECTION_STRATEGIES:
            raise ValueError(
                f"Nieznana strategia doboru V': {self.selection!r}; oczekiwano jednej z {SELECTION_STRATEGIES}."
            )
        if self.mode not in VPRIME_MODES:
            raise ValueError(f"Nieznany tryb V': {self.mode!r}; oczekiwano jednej z {VPRIME_MODES}.")
        if self.mode == "fixed":
            if self.selection != "random":
                raise ValueError("Tryb 'fixed' ma sens tylko dla selection='random' (wariant bazowy).")
            if not self.fixed_size or int(self.fixed_size) <= 0:
                raise ValueError("Tryb 'fixed' wymaga dodatniego vprime.fixed_size.")


class VPrimeSet:
    """Uporządkowany zbiór identyfikatorów próbek V' (indeksy wierszy pełnego V), bez duplikatów."""

    def __init__(self, n_total: int, ids=()):
        self.n_total = int(n_total)
        self._ids = np.empty(0, dtype=np.int64)
        if len(ids):
            self.replace(ids)

    # --- dostęp ---
    @property
    def ids(self) -> np.ndarray:
        return self._ids.copy()

    @property
    def size(self) -> int:
        return int(self._ids.size)

    def mask(self) -> np.ndarray:
        mask = np.zeros(self.n_total, dtype=bool)
        mask[self._ids] = True
        return mask

    def candidates(self) -> np.ndarray:
        """Identyfikatory V, które NIE należą do V' (pula do dalszego wyboru)."""
        return np.flatnonzero(~self.mask())

    # --- walidacja ---
    def _check(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).ravel()
        if ids.size and (ids.min() < 0 or ids.max() >= self.n_total):
            raise ValueError("Identyfikator próbki V' poza zakresem pełnego V.")
        if np.unique(ids).size != ids.size:
            raise ValueError("Duplikaty identyfikatorów próbek w V'.")
        return ids

    # --- modyfikacje ---
    def extend(self, new_ids) -> int:
        """Dodaje nowe elementy z zachowaniem dotychczasowych (memory=True). Zwraca liczbę dodanych."""
        new_ids = self._check(new_ids)
        if np.intersect1d(new_ids, self._ids).size:
            raise ValueError("Nowe elementy V' nie mogą powtarzać elementów już wybranych.")
        self._ids = np.concatenate([self._ids, new_ids])
        return int(new_ids.size)

    def replace(self, new_ids) -> None:
        """Zastępuje V' nowym zbiorem (memory=False albo inicjalizacja)."""
        self._ids = self._check(new_ids).copy()


def class_balanced_ids(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Losowe, zbalansowane klasowo ``per_class`` identyfikatorów na klasę (start V')."""
    y = np.asarray(y)
    picked: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        take = min(int(per_class), int(cls_idx.size))
        if take > 0:
            picked.append(rng.choice(cls_idx, size=take, replace=False))
    if not picked:
        return np.empty(0, dtype=np.int64)
    # kolejność rosnąca = identyczna z maską bool używaną w poprzedniej implementacji
    return np.sort(np.concatenate(picked)).astype(np.int64)


def stratified_random_ids(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Losowe ``n`` identyfikatorów z zachowaniem proporcji klas pełnego V (największe reszty)."""
    y = np.asarray(y)
    n = int(min(n, y.size))
    classes, counts = np.unique(y, return_counts=True)
    exact = counts / counts.sum() * n
    base = np.floor(exact).astype(int)
    remainder = n - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:remainder]] += 1
    picked = []
    for cls, take in zip(classes, base):
        cls_idx = np.flatnonzero(y == cls)
        picked.append(rng.choice(cls_idx, size=min(int(take), cls_idx.size), replace=False))
    return np.sort(np.concatenate(picked)).astype(np.int64)


def budget_pool_ids(candidate_ids: np.ndarray, y_cand: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Losowa podpula kandydatów o rozmiarze ``size`` z zachowaniem proporcji klas (budżet doboru V').

    Wynik zależy wyłącznie od etykiet i generatora ``rng`` — nigdy od modelu ani od strategii doboru.
    """
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if size >= candidate_ids.size:
        return candidate_ids
    local = stratified_random_ids(np.asarray(y_cand), int(size), rng)
    return candidate_ids[local].astype(np.int64)


def _decision_margin_abs(clf, X: np.ndarray) -> np.ndarray:
    try:
        margin = clf.decision_function(X)
    except Exception:
        return np.ones(X.shape[0], dtype=float)
    margin = np.asarray(margin)
    if margin.ndim == 1:
        return np.abs(margin)
    return np.min(np.abs(margin), axis=1)


@dataclass
class SelectionResult:
    ids: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _select_heuristic(X_cand, y_cand, candidate_ids, clf, per_class, counter) -> SelectionResult:
    """Błędy modelu, a następnie najmniejszy margines — osobno w każdej klasie."""
    selected_local: list[np.ndarray] = []
    pred = clf.predict(X_cand)
    counter["model_predictions"] += int(len(candidate_ids))
    counter["selection_predict_rows"] = counter.get("selection_predict_rows", 0) + int(len(candidate_ids))
    incorrect = pred != y_cand
    margins = _decision_margin_abs(clf, X_cand)
    # drugie, osobne przejście modelu po całej puli (decision_function); fallback bez wywołania modelu
    # liczymy tylko wtedy, gdy model faktycznie udostępnia decision_function
    if hasattr(clf, "decision_function"):
        counter["selection_decision_rows"] = counter.get("selection_decision_rows", 0) + int(len(candidate_ids))
    for cls in np.unique(y_cand):
        cls_local_idx = np.flatnonzero(y_cand == cls)
        if cls_local_idx.size == 0:
            continue
        cls_incorrect = (~incorrect[cls_local_idx]).astype(int)
        order = np.lexsort((margins[cls_local_idx], cls_incorrect))
        take = min(int(per_class), int(cls_local_idx.size))
        selected_local.append(cls_local_idx[order[:take]])
    local = np.concatenate(selected_local) if selected_local else np.empty(0, dtype=int)
    return SelectionResult(
        ids=candidate_ids[local].astype(np.int64),
        diagnostics={
            "candidate_pool_error_rate": float(incorrect.mean()) if incorrect.size else None,
            "candidate_pool_errors_available": int(incorrect.sum()),
        },
    )


def _select_evolutionary(
    X_cand, y_cand, candidate_ids, clf, n_select, metric, val_ga_cfg, seed, counter
) -> SelectionResult:
    """Adwersarialny Val-GA: maski nad kandydatami, fitness = 1 − metryka ustalonego modelu."""
    n_candidates = int(len(candidate_ids))
    n_sel = min(int(n_select), n_candidates)
    vcfg = ValEncodingConfig(subset_fraction=float(n_sel) / float(n_candidates))
    # Model jest ustalony na czas Val-GA, więc predykcje liczymy raz dla puli (wynik identyczny).
    pool_pred = clf.predict(X_cand)
    counter["model_predictions"] += n_candidates
    counter["selection_predict_rows"] = counter.get("selection_predict_rows", 0) + n_candidates
    history_adv, best_mask = run_val_ga(
        X_cand,
        y_cand,
        clf,
        vcfg,
        val_ga_cfg,
        metric=metric,
        seed=seed,
        preds=pool_pred,
        verbose=False,
    )
    evaluations = int(len(history_adv)) * int(val_ga_cfg.population_size)
    incorrect = pool_pred != y_cand
    return SelectionResult(
        ids=candidate_ids[best_mask].astype(np.int64),
        diagnostics={
            "val_ga_history": [float(v) for v in history_adv],
            "val_ga_population_evaluations": evaluations,
            # nominalna liczba klasyfikacji wierszy, gdyby fitness nie korzystał z cache predykcji
            "val_ga_nominal_subset_rows_scored": int(evaluations * int(best_mask.sum())),
            "subset_fraction": float(vcfg.subset_fraction),
            "candidate_pool_error_rate": float(incorrect.mean()) if incorrect.size else None,
            "candidate_pool_errors_available": int(incorrect.sum()),
        },
    )


def select_new_ids(
    strategy: str,
    *,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidate_ids: np.ndarray,
    clf,
    n_select: int,
    per_class: int,
    metric: str,
    val_ga_cfg: GAConfig | None,
    seed: int,
    rng: np.random.Generator,
    counter: dict,
) -> SelectionResult:
    """Wybiera nowe identyfikatory z puli ``candidate_ids`` zadaną strategią."""
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    empty = SelectionResult(ids=np.empty(0, dtype=np.int64), diagnostics={})
    if candidate_ids.size == 0 or n_select <= 0:
        return empty

    if strategy == "random":
        take = min(int(n_select), int(candidate_ids.size))
        return SelectionResult(ids=np.sort(rng.choice(candidate_ids, size=take, replace=False)).astype(np.int64))

    X_cand = X_val[candidate_ids]
    y_cand = y_val[candidate_ids]
    if strategy == "heuristic":
        return _select_heuristic(X_cand, y_cand, candidate_ids, clf, per_class, counter)
    if strategy == "evolutionary":
        if val_ga_cfg is None:
            raise ValueError("Strategia 'evolutionary' wymaga konfiguracji val_ga.")
        return _select_evolutionary(
            X_cand, y_cand, candidate_ids, clf, n_select, metric, val_ga_cfg, seed, counter
        )
    raise ValueError(f"Nieznana strategia doboru V': {strategy!r}")


@dataclass
class ExpansionResult:
    selected_ids: np.ndarray
    size_before: int
    size_after: int
    n_new: int          # elementy V' po operacji, których nie było w V' przed operacją
    n_dropped: int      # elementy usunięte z V' (zawsze 0 przy memory=True)
    diagnostics: dict


def expand_vprime(
    vset: VPrimeSet,
    *,
    memory: bool,
    strategy: str,
    n_add: int,
    per_class_add: int,
    n_classes: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    clf,
    metric: str,
    val_ga_cfg: GAConfig | None,
    seed: int,
    rng: np.random.Generator,
    counter: dict,
    score_candidate_pool: bool = True,
    pool_budget_factor: float | None = None,
) -> ExpansionResult:
    """Rozszerza (memory=True) albo regeneruje (memory=False) V' i pilnuje spójności identyfikatorów.

    ``pool_budget_factor`` (opcjonalnie): wspólny dla wszystkich strategii limit budżetu — strategia wybiera
    wyłącznie z losowej, stratyfikowanej klasowo podpuli o rozmiarze <= czynnik × liczba wybieranych elementów.
    Podpula jest losowana generatorem zależnym tylko od ``seed`` (nie od strategii, modelu ani ``rng``).
    """
    before_ids = vset.ids
    size_before = vset.size
    candidates = vset.candidates()
    cost_before = {k: int(counter.get(k, 0)) for k in _COST_KEYS}

    if memory:
        n_select = int(n_add)
        per_class = int(per_class_add)
    else:
        # ten sam harmonogram rozmiaru co przy memory=True, ale bez przenoszenia elementów
        n_select = min(size_before + int(n_add), int(candidates.size))
        per_class = int(np.ceil((size_before + int(n_add)) / max(1, n_classes)))

    pool_ids = candidates
    if pool_budget_factor is not None and n_select > 0 and candidates.size:
        cap = int(np.ceil(float(pool_budget_factor) * int(n_select)))
        if cap < candidates.size:
            pool_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 303]))
            pool_ids = budget_pool_ids(candidates, y_val[candidates], cap, pool_rng)

    result = select_new_ids(
        strategy,
        X_val=X_val,
        y_val=y_val,
        candidate_ids=pool_ids,
        clf=clf,
        n_select=n_select,
        per_class=per_class,
        metric=metric,
        val_ga_cfg=val_ga_cfg,
        seed=seed,
        rng=rng,
        counter=counter,
    )
    selected = result.ids
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "selection_strategy": strategy,
            "memory": bool(memory),
            "n_candidates": int(candidates.size),
            "n_scored_pool": int(pool_ids.size),
            "n_select_target": int(n_select),
            "pool_budget_factor": None if pool_budget_factor is None else float(pool_budget_factor),
        }
    )

    if selected.size:
        if memory:
            vset.extend(selected)
        else:
            vset.replace(selected)

    after_ids = vset.ids
    # --- inwarianty identyfikatorów próbek ---
    assert np.unique(after_ids).size == after_ids.size, "duplikaty w V'"
    if memory:
        assert np.array_equal(after_ids[:size_before], before_ids), "memory=True: zgubiono wcześniejsze elementy V'"
    else:
        if selected.size:
            assert np.intersect1d(after_ids, before_ids).size == 0, "memory=False: stare elementy nie powinny wrócić"

    n_new = int(np.setdiff1d(after_ids, before_ids).size)
    n_dropped = int(np.setdiff1d(before_ids, after_ids).size)

    # --- diagnostyka trudności wybranych elementów (obserwacja, nie steruje algorytmem) ---
    if selected.size:
        sel_pred = clf.predict(X_val[selected])
        sel_err = sel_pred != y_val[selected]
        diagnostics["selected_rows"] = int(selected.size)
        diagnostics["selected_errors"] = int(sel_err.sum())
        diagnostics["selected_error_rate"] = float(sel_err.mean())
        diagnostics["selected_mean_abs_margin"] = float(_decision_margin_abs(clf, X_val[selected]).mean())
    else:
        diagnostics.update(
            {"selected_rows": 0, "selected_errors": 0, "selected_error_rate": None, "selected_mean_abs_margin": None}
        )
    if selected.size:
        # diagnostyka przyrostu: predict + decision_function na wybranych wierszach
        counter["diagnostic_scored_rows"] = counter.get("diagnostic_scored_rows", 0) + 2 * int(selected.size)
    if score_candidate_pool and diagnostics.get("candidate_pool_error_rate") is None and pool_ids.size:
        pool_pred = clf.predict(X_val[pool_ids])
        diagnostics["candidate_pool_error_rate"] = float((pool_pred != y_val[pool_ids]).mean())
        diagnostics["candidate_pool_errors_available"] = int((pool_pred != y_val[pool_ids]).sum())
        counter["diagnostic_predictions"] = counter.get("diagnostic_predictions", 0) + int(pool_ids.size)
        counter["diagnostic_scored_rows"] = counter.get("diagnostic_scored_rows", 0) + int(pool_ids.size)

    # --- koszt tego rozszerzenia (faktyczne wiersze przekazane do modelu vs nominalne wartości Val-GA) ---
    diagnostics["cost_selection_predict_rows"] = int(counter.get("selection_predict_rows", 0)) - cost_before["selection_predict_rows"]
    diagnostics["cost_selection_decision_rows"] = int(counter.get("selection_decision_rows", 0)) - cost_before["selection_decision_rows"]
    diagnostics["cost_diagnostic_scored_rows"] = int(counter.get("diagnostic_scored_rows", 0)) - cost_before["diagnostic_scored_rows"]

    return ExpansionResult(
        selected_ids=selected,
        size_before=size_before,
        size_after=vset.size,
        n_new=n_new,
        n_dropped=n_dropped,
        diagnostics=diagnostics,
    )


class FullValidationObserver:
    """Zewnętrzna OBSERWACJA jakości na pełnym V.

    Wynik ``observe`` jest tylko zapisywany do logów i podsumowań — pętla algorytmu nie
    może go odczytywać do podejmowania decyzji. ``enabled=False`` wyłącza obserwację
    (używane m.in. w teście, że wyłączenie nie zmienia przebiegu algorytmu).
    """

    def __init__(self, X_full, y_full, metric: str, evaluate, enabled: bool = True):
        self._X = X_full
        self._y = y_full
        self._metric = metric
        self._evaluate = evaluate
        self.enabled = bool(enabled)
        self.calls = 0
        self.rows_scored = 0

    def observe(self, clf) -> dict | None:
        if not self.enabled:
            return None
        self.calls += 1
        self.rows_scored += int(len(self._y))
        return self._evaluate(clf, self._X, self._y, metric=self._metric)
