# src/ga/loop/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from ga.encoding import EncodingConfig, init_individual, decode
from ga.models.svm import SVMConfig
from ga.fitness import FitnessConfig, evaluate_individual
from dataclasses import dataclass, field



from dataclasses import dataclass
from typing import Tuple
import numpy as np

from ga.encoding import (
    KPerClassConfig, init_k_per_class,
    mutate_k_per_class, repair_to_k_per_class
)
from ga.fitness import FitnessConfig, evaluate_individual_fixed_train
from ga.models.svm import SVMConfig

# --- Konfiguracje ---

@dataclass
class Constraints:
    min_class_fraction: float = 0.0  # na razie nieużywane (hook pod przyszłe ograniczenia)

@dataclass
class EarlyStopping:
    patience: int = 0           # 0 = off
    min_delta: float = 0.0


@dataclass
class GAConfig:
    population_size: int = 60
    generations: int = 50
    tournament_k: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    elitism: int = 2
    constraints: Constraints = field(default_factory=Constraints)
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)

# --- Selekcja / krzyżowanie / mutacja ---

def _tournament_select(scores: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Zwraca indeks zwycięzcy turnieju rozmiaru k."""
    idx = rng.choice(len(scores), size=k, replace=False)
    best_local = idx[np.argmax(scores[idx])]
    return int(best_local)

def _crossover_masks(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Prosty uniform crossover na maskach bool: dla każdej pozycji wybierz gen z p1 lub p2."""
    assert p1.shape == p2.shape
    mask = rng.random(p1.shape[0]) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1.copy(), c2.copy()

def _mutate_mask(m: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Flip kilku pozycji (train<->val) z prawdopodobieństwem `rate`."""
    m = m.copy()
    flips = rng.random(m.shape[0]) < rate
    if np.any(flips):
        m[flips] = ~m[flips]
    return m

# --- Ewaluacja populacji ---

def _evaluate_population(pop: np.ndarray, X: np.ndarray, y: np.ndarray,
                         svm_cfg: SVMConfig, fcfg: FitnessConfig) -> np.ndarray:
    scores = np.empty(pop.shape[0], dtype=float)
    for i, mask in enumerate(pop):
        scores[i] = evaluate_individual(mask, X, y, svm_cfg, fcfg)
    return scores

# --- Główna pętla GA ---

def run_ga(
    X: np.ndarray,
    y: np.ndarray,
    enc_cfg: EncodingConfig,
    svm_cfg: SVMConfig,
    fcfg: FitnessConfig,
    gcfg: GAConfig,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zwraca:
      - history: np.ndarray[generacje] z najlepszym wynikiem w generacji
      - best_mask: np.ndarray[bool] – najlepszy znaleziony podział (True=val, False=train)
    """
    rng = np.random.default_rng(seed)

    # Inicjalizacja populacji (maski bool: True -> walidacja)
    pop = np.stack([init_individual(len(X), enc_cfg) for _ in range(gcfg.population_size)], axis=0)
    scores = _evaluate_population(pop, X, y, svm_cfg, fcfg)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_mask = pop[best_idx].copy()
    history = [best_score]
    no_improve = 0

    for _gen in range(gcfg.generations):
        new_pop = []

        # --- Elityzm ---
        elite_idx = np.argsort(scores)[-gcfg.elitism:]
        for ei in elite_idx:
            new_pop.append(pop[int(ei)].copy())

        # --- Reprodukcja ---
        while len(new_pop) < gcfg.population_size:
            # selekcja rodziców
            p1 = pop[_tournament_select(scores, gcfg.tournament_k, rng)]
            p2 = pop[_tournament_select(scores, gcfg.tournament_k, rng)]

            # krzyżowanie
            if rng.random() < gcfg.crossover_rate:
                c1, c2 = _crossover_masks(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # mutacja
            c1 = _mutate_mask(c1, gcfg.mutation_rate, rng)
            c2 = _mutate_mask(c2, gcfg.mutation_rate, rng)

            new_pop.extend([c1, c2])

        # przytnij do rozmiaru populacji
        pop = np.stack(new_pop[:gcfg.population_size], axis=0)
        scores = _evaluate_population(pop, X, y, svm_cfg, fcfg)

        # logika best / early-stopping
        gen_best = float(scores.max())
        history.append(gen_best)

        if gen_best > best_score + gcfg.early_stopping.min_delta:
            best_score = gen_best
            best_mask = pop[int(np.argmax(scores))].copy()
            no_improve = 0
        else:
            no_improve += 1
            if gcfg.early_stopping.patience and no_improve >= gcfg.early_stopping.patience:
                break

    return np.array(history, dtype=float), best_mask








def _uniform_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    assert p1.shape == p2.shape
    mask = rng.random(p1.shape[0]) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1.copy(), c2.copy()

def _evaluate_population_k(
    pop: np.ndarray, X_pool: np.ndarray, y_pool: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    svm_cfg: SVMConfig, fcfg: FitnessConfig
) -> np.ndarray:
    scores = np.empty(pop.shape[0], dtype=float)
    for i, train_mask in enumerate(pop):
        scores[i] = evaluate_individual_fixed_train(train_mask, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg)
    return scores

def run_ga_k_per_class(
    X_pool: np.ndarray, y_pool: np.ndarray,    # to jest pula, z której wybieramy k/klasę -> TRAIN
    X_val:  np.ndarray, y_val:  np.ndarray,    # STAŁA walidacja
    kcfg: KPerClassConfig,
    svm_cfg: SVMConfig,
    fcfg: FitnessConfig,
    gcfg: GAConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zwraca: (history, best_train_mask)
      best_train_mask: True => próbka z puli trafia do TRAIN (dokładnie k na klasę)
    """
    rng = np.random.default_rng(seed)

    # inicjalizacja populacji
    pop = np.stack([init_k_per_class(y_pool, kcfg) for _ in range(gcfg.population_size)], axis=0)
    scores = _evaluate_population_k(pop, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_mask = pop[best_idx].copy()
    history = [best_score]
    no_improve = 0

    for _gen in range(gcfg.generations):
        new_pop = []

        # elityzm
        elite_idx = np.argsort(scores)[-gcfg.elitism:]
        for ei in elite_idx:
            new_pop.append(pop[int(ei)].copy())

        # reprodukcja
        while len(new_pop) < gcfg.population_size:
            p1 = pop[_tournament_select(scores, gcfg.tournament_k, rng)]
            p2 = pop[_tournament_select(scores, gcfg.tournament_k, rng)]

            # krzyżowanie + naprawa k-per-class
            if rng.random() < gcfg.crossover_rate:
                c1, c2 = _uniform_crossover(p1, p2, rng)
                c1 = repair_to_k_per_class(c1, y_pool, kcfg.k_per_class, rng)
                c2 = repair_to_k_per_class(c2, y_pool, kcfg.k_per_class, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # mutacja
            c1 = mutate_k_per_class(c1, y_pool, gcfg.mutation_rate, rng, kcfg.k_per_class)
            c2 = mutate_k_per_class(c2, y_pool, gcfg.mutation_rate, rng, kcfg.k_per_class)

            new_pop.extend([c1, c2])

        pop = np.stack(new_pop[:gcfg.population_size], axis=0)
        scores = _evaluate_population_k(pop, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg)

        gen_best = float(scores.max())
        history.append(gen_best)
        # log postępu co generację
        print(f"[GEN {_gen + 1:03d}/{gcfg.generations}] best={gen_best:.4f}, global_best={best_score:.4f}")

        if gen_best > best_score + gcfg.early_stopping.min_delta:
            best_score = gen_best
            best_mask = pop[int(np.argmax(scores))].copy()
            no_improve = 0
        else:
            no_improve += 1
            if gcfg.early_stopping.patience and no_improve >= gcfg.early_stopping.patience:
                break

    return np.array(history, dtype=float), best_mask

from dataclasses import dataclass
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score


@dataclass
class ValEncodingConfig:
    subset_fraction: float = 0.5  # jaki % walidacji bierzemy do adwersarialnego podzbioru


def _init_val_mask(n: int, vcfg: ValEncodingConfig, rng: np.random.Generator) -> np.ndarray:
    """Losowy osobnik dla Val-GA: maska bool długości n, dokładnie n_sel True."""
    n_sel = max(1, int(round(vcfg.subset_fraction * n)))
    mask = np.zeros(n, dtype=bool)
    idx = rng.choice(n, size=n_sel, replace=False)
    mask[idx] = True
    return mask


def _repair_val_mask(mask: np.ndarray, n_sel: int, rng: np.random.Generator) -> np.ndarray:
    """Naprawia maskę po krzyżowaniu/mutacji, aby mieć dokładnie n_sel True."""
    m = mask.copy()
    idx_true = np.flatnonzero(m)
    if len(idx_true) > n_sel:
        drop = rng.choice(idx_true, size=len(idx_true) - n_sel, replace=False)
        m[drop] = False
    elif len(idx_true) < n_sel:
        idx_false = np.flatnonzero(~m)
        add = rng.choice(idx_false, size=n_sel - len(idx_true), replace=False)
        m[add] = True
    return m


def _metric_from_preds(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "f1":
        return f1_score(y_true, y_pred, average="binary", zero_division=0)
    elif metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    else:
        return balanced_accuracy_score(y_true, y_pred)


def _evaluate_population_val(
    pop: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    clf,
    metric: str,
) -> np.ndarray:
    """
    Ewaluacja populacji Val-GA:
      - dla każdej maski wybieramy podzbiór walidacji
      - liczmy METRYKĘ ustalonego SVM na tym podzbiorze
      - fitness = 1 - metric  (adwersarialnie: im gorzej, tym lepiej)
    """
    scores = np.empty(pop.shape[0], dtype=float)
    for i, mask in enumerate(pop):
        if not np.any(mask):
            scores[i] = 0.0
            continue
        X_sub = X_val[mask]
        y_sub = y_val[mask]
        pred = clf.predict(X_sub)
        m = _metric_from_preds(y_sub, pred, metric)
        scores[i] = 1.0 - m  # im gorsza metryka, tym większy fitness
    return scores


def run_val_ga(
    X_val: np.ndarray,
    y_val: np.ndarray,
    clf,
    vcfg: ValEncodingConfig,
    gcfg: GAConfig,
    metric: str = "balanced_accuracy",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drugi GA (Val-GA): szuka podzbioru walidacji, który MAKSYMALNIE PSUJE wynik ustalonego SVM.

    Zwraca:
      - history_adv: najlepszy (najbardziej szkodliwy) fitness w każdej generacji
      - best_val_mask: maska bool nad X_val/y_val (True => wchodzi do adwersarialnej walidacji)
    """
    rng = np.random.default_rng(seed)
    n = X_val.shape[0]
    n_sel = max(1, int(round(vcfg.subset_fraction * n)))

    # inicjalna populacja masek walidacji
    pop = np.stack([_init_val_mask(n, vcfg, rng) for _ in range(gcfg.population_size)], axis=0)
    scores = _evaluate_population_val(pop, X_val, y_val, clf, metric)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_mask = pop[best_idx].copy()
    history = [best_score]
    no_improve = 0

    for gen in range(gcfg.generations):
        new_pop = []

        # elityzm
        elite_idx = np.argsort(scores)[-gcfg.elitism:]
        for ei in elite_idx:
            new_pop.append(pop[int(ei)].copy())

        # reprodukcja
        while len(new_pop) < gcfg.population_size:
            p1 = pop[_tournament_select(scores, gcfg.tournament_k, rng)]
            p2 = pop[_tournament_select(scores, gcfg.tournament_k, rng)]

            # krzyżowanie
            if rng.random() < gcfg.crossover_rate:
                c1, c2 = _uniform_crossover(p1, p2, rng)
                c1 = _repair_val_mask(c1, n_sel, rng)
                c2 = _repair_val_mask(c2, n_sel, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # mutacja + naprawa
            c1 = _mutate_mask(c1, gcfg.mutation_rate, rng)
            c1 = _repair_val_mask(c1, n_sel, rng)

            c2 = _mutate_mask(c2, gcfg.mutation_rate, rng)
            c2 = _repair_val_mask(c2, n_sel, rng)

            new_pop.extend([c1, c2])

        pop = np.stack(new_pop[:gcfg.population_size], axis=0)
        scores = _evaluate_population_val(pop, X_val, y_val, clf, metric)

        gen_best = float(scores.max())
        history.append(gen_best)

        # log z pokolenia (opcjonalnie zostawione)
        # statystyki populacji (raport stabilności)
        gen_mean = float(scores.mean())
        gen_worst = float(scores.min())
        gen_var = float(scores.var())

        print(
            f"[VAL-GA GEN {gen + 1:03d}/{gcfg.generations}] "
            f"best={gen_best:.4f}, mean={gen_mean:.4f}, worst={gen_worst:.4f}, var={gen_var:.6f}, "
            f"global_best={best_score:.4f}"
        )

        if gen_best > best_score + gcfg.early_stopping.min_delta:
            best_score = gen_best
            best_mask = pop[int(np.argmax(scores))].copy()
            no_improve = 0
        else:
            no_improve += 1
            if gcfg.early_stopping.patience and no_improve >= gcfg.early_stopping.patience:
                break

    return np.array(history, dtype=float), best_mask
