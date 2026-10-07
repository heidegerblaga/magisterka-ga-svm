import random, numpy as np

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def limit_dataset(X_train, X_val, y_train, y_val, max_rows=1000, seed: int = 42):
    if max_rows is None or max_rows <= 0:
        return X_train, y_train, X_val, y_val

    n_train = min(len(X_train), max_rows // 2)
    n_val = min(len(X_val), max_rows - n_train)

    rng = np.random.default_rng(seed)
    idx_train = rng.choice(len(X_train), n_train, replace=False)
    idx_val = rng.choice(len(X_val), n_val, replace=False)

    X_train = X_train.iloc[idx_train] if hasattr(X_train, "iloc") else X_train[idx_train]
    y_train = y_train.iloc[idx_train] if hasattr(y_train, "iloc") else y_train[idx_train]
    X_val = X_val.iloc[idx_val] if hasattr(X_val, "iloc") else X_val[idx_val]
    y_val = y_val.iloc[idx_val] if hasattr(y_val, "iloc") else y_val[idx_val]

    return X_train, y_train, X_val, y_val
