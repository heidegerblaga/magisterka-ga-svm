# src/ga/loop/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from ga.encoding import EncodingConfig, init_individual, decode
from ga.models.svm import SVMConfig
from ga.fitness import FitnessConfig, evaluate_individual
from dataclasses import dataclass, field

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
