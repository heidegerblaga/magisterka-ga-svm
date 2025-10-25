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
