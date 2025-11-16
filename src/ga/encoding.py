import numpy as np
from dataclasses import dataclass

@dataclass
class EncodingConfig:
    val_fraction: float = 0.2  # fraction of total samples assigned to validation
    random_state: int = 42

def init_individual(n_samples: int, cfg: EncodingConfig) -> np.ndarray:
    """
    Chromosom: wektor bool (True=val, False=train) o zadanym udziale walidacji.
    """
    rng = np.random.default_rng(cfg.random_state)
    n_val = int(round(cfg.val_fraction * n_samples))
    mask = np.array([True]*n_val + [False]*(n_samples - n_val), dtype=bool)
    rng.shuffle(mask)
    return mask

def decode(mask: np.ndarray, X, y):
    X_train = X[~mask]; y_train = y[~mask]
    X_val = X[mask];  y_val  = y[mask]
    return X_train, y_train, X_val, y_val



# --- K-PER-CLASS ENCODING (train=True, val=False) ---

from collections import defaultdict
from dataclasses import dataclass
import numpy as np

@dataclass
class KPerClassConfig:
    k_per_class: int
    random_state: int = 42

def init_k_per_class(y: np.ndarray, cfg: KPerClassConfig) -> np.ndarray:
    """
    Zwraca maskę bool długości len(y): True => próbka trafia do TRAIN.
    W każdej klasie dokładnie k_per_class elementów ma True.
    """
    rng = np.random.default_rng(cfg.random_state)
    y = np.asarray(y)
    mask = np.zeros(len(y), dtype=bool)
    classes = np.unique(y)
    for c in classes:
        idx = np.flatnonzero(y == c)
        if len(idx) < cfg.k_per_class:
            raise ValueError(f"Klasa {c} ma {len(idx)} próbek < k={cfg.k_per_class}")
        chosen = rng.choice(idx, size=cfg.k_per_class, replace=False)
        mask[chosen] = True
    return mask

def mutate_k_per_class(mask: np.ndarray, y: np.ndarray, rate: float, rng: np.random.Generator, k: int) -> np.ndarray:
    """
    Mutacja: w każdej klasie 'wymień' ~rate * k zaznaczonych próbek na inne z tej samej klasy,
    zachowując dokładnie k per klasa.
    """
    m = mask.copy()
    y = np.asarray(y)
    classes = np.unique(y)
    for c in classes:
        cls_idx = np.flatnonzero(y == c)
        true_idx = cls_idx[m[cls_idx]]
        false_idx = cls_idx[~m[cls_idx]]

        if len(true_idx) != k or len(false_idx) == 0:
            continue

        k_flip = max(1, int(round(rate * k))) if rate > 0 else 0
        k_flip = min(k_flip, len(true_idx), len(false_idx))
        if k_flip == 0:
            continue

        drop = rng.choice(true_idx, size=k_flip, replace=False)
        add  = rng.choice(false_idx, size=k_flip, replace=False)
        m[drop] = False
        m[add]  = True
    return m

def repair_to_k_per_class(mask: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    'Napraw' maskę po krzyżowaniu tak, żeby w każdej klasie było dokładnie k True.
    """
    m = mask.copy()
    y = np.asarray(y)
    classes = np.unique(y)
    for c in classes:
        cls_idx = np.flatnonzero(y == c)
        true_idx = cls_idx[m[cls_idx]]
        if len(true_idx) > k:
            drop = rng.choice(true_idx, size=len(true_idx)-k, replace=False)
            m[drop] = False
        elif len(true_idx) < k:
            false_idx = cls_idx[~m[cls_idx]]
            if len(false_idx) < (k - len(true_idx)):
                raise ValueError(f"Za mało próbek klasy {c} do naprawy k={k}")
            add = rng.choice(false_idx, size=k-len(true_idx), replace=False)
            m[add] = True
    return m
