# src/ga/diversity.py
from dataclasses import dataclass
import numpy as np


@dataclass
class DiversityConfig:
    # jak mocno różnorodność wpływa na selekcję (0 = ignorujemy)
    weight: float = 0.0
    # minimalna pożądana odległość NN w klasie (do mutacji)
    min_pairwise_dist: float = 0.0
    # liczyć diversity osobno w klasach czy w całości
    per_class: bool = True


def precompute_dist_matrix(X: np.ndarray) -> np.ndarray:
    """
    Pełna macierz odległości euklidesowych między próbkami z puli TRAIN (X_pool).
    Zwraca macierz (n_samples, n_samples).
    """
    X = np.asarray(X, dtype=float)
    # (x_i - x_j)^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    # broadcast: n x n
    d2 = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    # mogą się pojawić małe wartości ujemne przez numerykę
    d2 = np.maximum(d2, 0.0)
    D = np.sqrt(d2)
    return D


def _mean_nn_distance(indices: np.ndarray, dist_matrix: np.ndarray) -> float:
    """
    Średnia odległość do najbliższego sąsiada wewnątrz zbioru indices.
    """
    if indices.size <= 1:
        return 0.0

    sub = dist_matrix[np.ix_(indices, indices)].copy()
    # wykluczamy "odległość do siebie"
    np.fill_diagonal(sub, np.inf)
    nn = sub.min(axis=1)  # (n_in_set,)
    return float(nn.mean())


def diversity_score(
    mask: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: DiversityConfig,
) -> float:
    """
    Liczy "różnorodność" osobnika (maski TRAIN) na podstawie średniej
    odległości do najbliższego sąsiada w k (k-per-class).

    Wyższa wartość = bardziej różnorodny zbiór.
    """
    mask = np.asarray(mask, dtype=bool)
    train_idx = np.flatnonzero(mask)
    if train_idx.size <= 1:
        return 0.0

    if not cfg.per_class:
        return _mean_nn_distance(train_idx, dist_matrix)

    # liczymy oddzielnie dla każdej klasy i uśredniamy
    ys = np.asarray(y_pool)
    classes = np.unique(ys[train_idx])
    if classes.size == 0:
        return 0.0

    vals = []
    for c in classes:
        idx_c = np.flatnonzero(mask & (ys == c))
        if idx_c.size <= 1:
            continue
        vals.append(_mean_nn_distance(idx_c, dist_matrix))

    if not vals:
        return 0.0
    return float(np.mean(vals))


def mutate_mask_diverse(
    mask: np.ndarray,
    y_pool: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: DiversityConfig,
    rng: np.random.Generator,
    p_mut: float = 0.01,
) -> np.ndarray:
    """
    Mutacja k-per-class z naciskiem na różnorodność:

    - w każdej klasie szukamy punktów, które mają bardzo bliskiego sąsiada (< min_pairwise_dist),
    - te punkty z pewnym prawdopodobieństwem wyrzucamy z k,
    - zamiast nich dokładamy punkty z tej samej klasy, które są jak najdalej od obecnych w k.
    """
    new_mask = mask.copy()
    ys = np.asarray(y_pool)
    classes = np.unique(ys)

    for c in classes:
        cls_idx = np.flatnonzero(ys == c)
        sel_idx = cls_idx[new_mask[cls_idx]]
        if sel_idx.size <= 1:
            continue

        # odległości w obrębie wybranych próbek danej klasy
        sub = dist_matrix[np.ix_(sel_idx, sel_idx)].copy()
        np.fill_diagonal(sub, np.inf)
        nn = sub.min(axis=1)

        # "za blisko" względem progu
        close_mask = nn < cfg.min_pairwise_dist
        to_consider = sel_idx[close_mask]
        if to_consider.size == 0:
            continue

        # kandydaci do dodania (z tej samej klasy, ale spoza k)
        cand_idx = cls_idx[~new_mask[cls_idx]]
        if cand_idx.size == 0:
            continue

        for i in to_consider:
            if rng.random() > p_mut:
                continue

            # wyrzucamy i z k
            new_mask[i] = False

            # liczymy odległość każdego kandydata do najbliższego aktualnie wybranego w tej klasie
            sel_after = cls_idx[new_mask[cls_idx]]
            if sel_after.size == 0:
                # jeśli opróżniliśmy klasę, to po prostu losujemy kandydata
                j = int(rng.choice(cand_idx))
                new_mask[j] = True
                continue

            D = dist_matrix[np.ix_(cand_idx, sel_after)]
            nn_cand = D.min(axis=1)
            # bierzemy kandydata jak najdalej od istniejących
            j = int(cand_idx[np.argmax(nn_cand)])
            new_mask[j] = True

    return new_mask
