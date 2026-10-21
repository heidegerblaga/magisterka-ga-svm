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


def build_ga_config(section: dict) -> GAConfig:
    """Buduje GAConfig z sekcji YAML (``ga`` / ``val_ga``); brakujące pola dostają wartości domyślne."""
    gcfg = GAConfig(
        population_size=section.get("population_size", 60),
        generations=section.get("generations", 20),
        tournament_k=section.get("tournament_k", 3),
        crossover_rate=section.get("crossover_rate", 0.8),
        mutation_rate=section.get("mutation_rate", 0.05),
        elitism=section.get("elitism", 2),
        early_stopping=section.get("early_stopping", {"patience": 0, "min_delta": 0.0}),
    )
    if isinstance(gcfg.early_stopping, dict):
        gcfg.early_stopping = EarlyStopping(**gcfg.early_stopping)
    return gcfg

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
    svm_cfg: SVMConfig, fcfg: FitnessConfig,
    *,
    return_meta: bool = False,
) -> tuple[np.ndarray, list[dict] | None]:
    scores = np.empty(pop.shape[0], dtype=float)
    metas: list[dict] | None = [] if return_meta else None
    for i, train_mask in enumerate(pop):
        if return_meta:
            score, meta = evaluate_individual_fixed_train(
                train_mask, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg, return_meta=True
            )
            scores[i] = float(score)
            metas.append(meta)
        else:
            scores[i] = evaluate_individual_fixed_train(train_mask, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg)
    return scores, metas

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

    # Każdy osobnik musi mieć inny seed startowy, inaczej populacja początkowa
    # składa się z identycznych masek i GA traci realną różnorodność.
    pop = np.stack(
        [
            init_k_per_class(
                y_pool,
                KPerClassConfig(k_per_class=kcfg.k_per_class, random_state=seed + i),
            )
            for i in range(gcfg.population_size)
        ],
        axis=0,
    )
    scores, _ = _evaluate_population_k(pop, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg)

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
        scores, _ = _evaluate_population_k(pop, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg)

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


def run_ga_k_per_class_with_trace(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    kcfg: KPerClassConfig,
    svm_cfg: SVMConfig,
    fcfg: FitnessConfig,
    gcfg: GAConfig,
    seed: int = 42,
    seed_masks: list[np.ndarray] | None = None,
    target_fitness: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Wariant GASVM: zwraca też trace per generacja + najlepsze maski w generacjach.

    ``seed_masks`` (opcjonalnie): maski (dokładnie k/klasę) wstawiane na początek populacji startowej
    (warm start T'); ``target_fitness`` (opcjonalnie): GA kończy się, gdy najlepszy fitness osiągnie tę wartość.
    Przy wartościach domyślnych (None) przebieg jest identyczny z historycznym.
    """
    rng = np.random.default_rng(seed)

    pop = np.stack(
        [
            init_k_per_class(
                y_pool,
                KPerClassConfig(k_per_class=kcfg.k_per_class, random_state=seed + i),
            )
            for i in range(gcfg.population_size)
        ],
        axis=0,
    )
    for i, mask in enumerate((seed_masks or [])[: gcfg.population_size]):
        pop[i] = np.asarray(mask, dtype=bool)
    scores, metas = _evaluate_population_k(
        pop, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg, return_meta=True
    )

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_mask = pop[best_idx].copy()

    history = [best_score]
    trace: list[dict] = []
    best_masks_per_generation: list[np.ndarray] = []
    no_improve = 0

    def sv_ratio(meta: dict) -> float | None:
        training_rows = int(meta.get("training_rows", 0) or 0)
        sv_count = int(meta.get("num_support_vectors", 0) or 0)
        return None if training_rows <= 0 else float(sv_count / training_rows)

    reached = target_fitness is not None and best_score >= target_fitness
    for gen in range(0 if reached else gcfg.generations):
        new_pop = []
        elite_idx = np.argsort(scores)[-gcfg.elitism:]
        for ei in elite_idx:
            new_pop.append(pop[int(ei)].copy())

        while len(new_pop) < gcfg.population_size:
            p1 = pop[_tournament_select(scores, gcfg.tournament_k, rng)]
            p2 = pop[_tournament_select(scores, gcfg.tournament_k, rng)]

            if rng.random() < gcfg.crossover_rate:
                c1, c2 = _uniform_crossover(p1, p2, rng)
                c1 = repair_to_k_per_class(c1, y_pool, kcfg.k_per_class, rng)
                c2 = repair_to_k_per_class(c2, y_pool, kcfg.k_per_class, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate_k_per_class(c1, y_pool, gcfg.mutation_rate, rng, kcfg.k_per_class)
            c2 = mutate_k_per_class(c2, y_pool, gcfg.mutation_rate, rng, kcfg.k_per_class)

            new_pop.extend([c1, c2])

        pop = np.stack(new_pop[: gcfg.population_size], axis=0)
        scores, metas = _evaluate_population_k(
            pop, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg, return_meta=True
        )

        gen_best_idx = int(np.argmax(scores))
        gen_best = float(scores[gen_best_idx])
        gen_mean = float(scores.mean())
        history.append(gen_best)
        best_masks_per_generation.append(pop[gen_best_idx].copy())

        gen_meta = (metas or [])[gen_best_idx] if metas else {}
        sv_counts = [int(m.get("num_support_vectors", 0)) for m in (metas or [])]
        trace.append(
            {
                "generation": gen + 1,
                "k_per_class": int(kcfg.k_per_class),
                "best_fitness": float(gen_best),
                "mean_fitness": float(gen_mean),
                "global_best_fitness": float(best_score),
                "best_num_support_vectors": int(gen_meta.get("num_support_vectors", 0) or 0),
                "mean_num_support_vectors": float(np.mean(sv_counts)) if sv_counts else None,
                "best_sv_ratio": sv_ratio(gen_meta),
            }
        )

        if gen_best > best_score + gcfg.early_stopping.min_delta:
            best_score = gen_best
            best_mask = pop[gen_best_idx].copy()
            no_improve = 0
            if target_fitness is not None and best_score >= target_fitness:
                break
        else:
            no_improve += 1
            if gcfg.early_stopping.patience and no_improve >= gcfg.early_stopping.patience:
                break

    return (
        np.array(history, dtype=float),
        best_mask,
        trace,
        np.stack(best_masks_per_generation, axis=0)
        if best_masks_per_generation
        else np.zeros((0, X_pool.shape[0]), dtype=bool),
    )


@dataclass
class DagaKAdaptationConfig:
    """Uproszczona implementacja adaptacji K wg schematu z DAGASVM/MASVM.

    Nazewnictwo jak w pseudokodzie:
      - TSV: próg na ρ_SV = s / t' (udział SV w wybranym TRAIN)
      - Ts: minimalna liczba generacji bez zwiększania K
      - cs: licznik stabilnego rozmiaru chromosomu (wewnętrzny stan)
      - ε: współczynnik selekcji LGA (local-global adaptation)
    """

    k_start: int = 10
    TSV: float = 0.25
    Ts: int = 3
    epsilon: float = 0.5
    k_max: int | None = None


def _lga_parent_indices(
    scores: np.ndarray,
    tournament_k: int,
    rng: np.random.Generator,
    *,
    mode: str,
    epsilon: float,
) -> tuple[int, int]:
    """Dobiera parę rodziców wg LGA (local/global)."""
    n = int(scores.shape[0])
    if n < 2:
        return 0, 0

    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    split = int(round(epsilon * n))
    split = min(max(split, 1), n - 1)
    order = np.argsort(scores)
    fit_idx = order[-split:]
    weak_idx = order[: n - split]

    def tournament(indices: np.ndarray) -> int:
        if indices.size == 1:
            return int(indices[0])
        k = min(int(tournament_k), int(indices.size))
        cand = rng.choice(indices, size=k, replace=False)
        return int(cand[np.argmax(scores[cand])])

    if mode == "global":
        return tournament(fit_idx), tournament(weak_idx)

    # local: eksploruj obie połówki niezależnie
    # (pierwszy rodzic z tej samej części co drugi)
    if rng.random() < epsilon:
        return tournament(fit_idx), tournament(fit_idx)
    return tournament(weak_idx), tournament(weak_idx)


def run_daga_k_per_class(
    *,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    kcfg: KPerClassConfig,
    daca: DagaKAdaptationConfig,
    svm_cfg: SVMConfig,
    fcfg: FitnessConfig,
    gcfg: GAConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """GA z adaptacją K (DAGASVM): start od małego K i adaptacyjne zwiększanie.

    Zwraca:
      - history_best: najlepszy fitness w kolejnych generacjach
      - best_mask: najlepsza maska TRAIN (True => wchodzi do TRAIN)
      - trace: lista słowników z metrykami per generacja (K, ρ_SV, itp.)
      - best_masks_per_generation: tablica bool (G x |pool|) z najlepszym osobnikiem w generacji
    """
    rng = np.random.default_rng(seed)

    # K może się zmieniać w trakcie, ale chromosomy są maskami nad X_pool.
    current_k_float = float(max(1, int(daca.k_start or kcfg.k_per_class)))
    k_cap = None
    if daca.k_max is not None:
        k_cap = int(daca.k_max)
    else:
        # twardy cap: najmniejsza klasa w puli
        _, counts = np.unique(np.asarray(y_pool), return_counts=True)
        k_cap = int(counts.min())

    lga_mode = "local"
    cs = 0  # stable chromosome size counter

    def k_int() -> int:
        value = int(np.ceil(current_k_float))
        value = max(1, value)
        return min(value, k_cap)

    # inicjalizacja populacji dla startowego K
    pop = np.stack(
        [
            init_k_per_class(
                y_pool,
                KPerClassConfig(k_per_class=k_int(), random_state=seed + i),
            )
            for i in range(gcfg.population_size)
        ],
        axis=0,
    )
    scores, metas = _evaluate_population_k(
        pop, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg, return_meta=True
    )

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_mask = pop[best_idx].copy()
    best_meta = (metas or [])[best_idx] if metas else {}

    eta_B = best_score
    eta_I = best_score  # best fitness at last K increase
    eta_P = best_score  # previous best fitness (for ΔR)

    history = [best_score]
    trace: list[dict] = []
    best_masks_per_generation: list[np.ndarray] = []

    def sv_ratio_from_meta(meta: dict) -> float | None:
        training_rows = int(meta.get("training_rows", 0) or 0)
        sv_count = int(meta.get("num_support_vectors", 0) or 0)
        return None if training_rows <= 0 else float(sv_count / training_rows)

    # generacje
    for gen in range(gcfg.generations):
        # elityzm
        new_pop = []
        elite_idx = np.argsort(scores)[-gcfg.elitism:]
        for ei in elite_idx:
            new_pop.append(pop[int(ei)].copy())

        # reprodukcja (LGA zamiast zwykłej selekcji)
        while len(new_pop) < gcfg.population_size:
            p1_idx, p2_idx = _lga_parent_indices(
                scores,
                gcfg.tournament_k,
                rng,
                mode=lga_mode,
                epsilon=daca.epsilon,
            )
            p1 = pop[p1_idx]
            p2 = pop[p2_idx]

            if rng.random() < gcfg.crossover_rate:
                c1, c2 = _uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # naprawa do aktualnego K
            kk = k_int()
            c1 = repair_to_k_per_class(c1, y_pool, kk, rng)
            c2 = repair_to_k_per_class(c2, y_pool, kk, rng)

            # mutacja + naprawa
            c1 = mutate_k_per_class(c1, y_pool, gcfg.mutation_rate, rng, kk)
            c2 = mutate_k_per_class(c2, y_pool, gcfg.mutation_rate, rng, kk)

            new_pop.extend([c1, c2])

        pop = np.stack(new_pop[: gcfg.population_size], axis=0)
        scores, metas = _evaluate_population_k(
            pop, X_pool, y_pool, X_val, y_val, svm_cfg, fcfg, return_meta=True
        )

        gen_best_idx = int(np.argmax(scores))
        gen_best = float(scores[gen_best_idx])
        gen_mean = float(scores.mean())
        history.append(gen_best)
        best_masks_per_generation.append(pop[gen_best_idx].copy())

        # aktualizacja globalnego best
        if gen_best > best_score + gcfg.early_stopping.min_delta:
            best_score = gen_best
            best_mask = pop[gen_best_idx].copy()
            best_meta = (metas or [])[gen_best_idx] if metas else {}
            eta_B = best_score

        # trace
        sv_counts = [int(m.get("num_support_vectors", 0)) for m in (metas or [])]
        mean_sv = float(np.mean(sv_counts)) if sv_counts else None
        best_sv = int(best_meta.get("num_support_vectors", 0) or 0)
        best_sv_ratio = sv_ratio_from_meta(best_meta)
        trace.append(
            {
                "generation": gen + 1,
                "k_per_class": int(k_int()),
                "lga_mode": lga_mode,
                "cs": int(cs),
                "best_fitness": float(gen_best),
                "mean_fitness": float(gen_mean),
                "global_best_fitness": float(best_score),
                "best_num_support_vectors": int(best_sv),
                "mean_num_support_vectors": mean_sv,
                "best_sv_ratio": best_sv_ratio,
            }
        )

        # --- Adaptacja K wg pseudokodu ---
        delta_T = float(eta_B - eta_I)
        delta_R = float(eta_B - eta_P)
        best_sv_ratio_now = best_sv_ratio

        should_skip_k_increase = (
            (cs < int(daca.Ts))
            or (delta_T == 0.0)
            or (delta_R >= 0.5 * delta_T if delta_T != 0.0 else True)
            or (best_sv_ratio_now is not None and best_sv_ratio_now < float(daca.TSV))
        )

        if should_skip_k_increase:
            cs += 1
        else:
            if lga_mode == "local":
                lga_mode = "global"
            else:
                # δK ← 1 + |ρSV(pB) − TSV| / (1 − TSV)
                rho = float(best_sv_ratio_now if best_sv_ratio_now is not None else 1.0)
                tsv = float(np.clip(daca.TSV, 0.0, 0.999999))
                delta_K = 1.0 + abs(rho - tsv) / (1.0 - tsv)
                prev_k = k_int()
                current_k_float = float(current_k_float * delta_K)
                if k_int() <= prev_k and prev_k < k_cap:
                    current_k_float = float(prev_k + 1)

                # po zwiększeniu K: ηI ← ηB; cs ← 0; LGA -> local
                eta_I = eta_B
                cs = 0
                lga_mode = "local"

                # dopasuj populację do nowego K (bez pełnej regeneracji)
                kk = k_int()
                for i in range(pop.shape[0]):
                    pop[i] = repair_to_k_per_class(pop[i], y_pool, kk, rng)

        eta_P = eta_B

    return (
        np.array(history, dtype=float),
        best_mask,
        trace,
        np.stack(best_masks_per_generation, axis=0) if best_masks_per_generation else np.zeros((0, X_pool.shape[0]), dtype=bool),
    )

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
    preds: np.ndarray | None = None,
    counter: dict | None = None,
) -> np.ndarray:
    """
    Ewaluacja populacji Val-GA:
      - dla każdej maski wybieramy podzbiór walidacji
      - liczmy METRYKĘ ustalonego SVM na tym podzbiorze
      - fitness = 1 - metric  (adwersarialnie: im gorzej, tym lepiej)

    `preds` (opcjonalnie): predykcje ustalonego modelu dla CAŁEJ puli kandydatów,
    policzone raz. Model jest stały w trakcie Val-GA, więc `preds[mask]` jest
    identyczne z `clf.predict(X_val[mask])` — wynik się nie zmienia, znika tylko
    wielokrotne wołanie modelu. `counter["model_predictions"]` zlicza liczbę
    predykcji modelu (koszt obliczeniowy doboru V').
    """
    scores = np.empty(pop.shape[0], dtype=float)
    for i, mask in enumerate(pop):
        if not np.any(mask):
            scores[i] = 0.0
            continue
        y_sub = y_val[mask]
        if preds is not None:
            pred = preds[mask]
        else:
            pred = clf.predict(X_val[mask])
            if counter is not None:
                counter["model_predictions"] = counter.get("model_predictions", 0) + int(mask.sum())
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
    preds: np.ndarray | None = None,
    counter: dict | None = None,
    verbose: bool = True,
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
    scores = _evaluate_population_val(pop, X_val, y_val, clf, metric, preds, counter)

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
        scores = _evaluate_population_val(pop, X_val, y_val, clf, metric, preds, counter)

        gen_best = float(scores.max())
        history.append(gen_best)

        # log z pokolenia (opcjonalnie zostawione)
        # statystyki populacji (raport stabilności)
        gen_mean = float(scores.mean())
        gen_worst = float(scores.min())
        gen_var = float(scores.var())

        if verbose:
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
