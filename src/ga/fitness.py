import numpy as np
from dataclasses import dataclass
from ga.models.svm import SVMConfig, train_eval_svm
from ga.encoding import decode

@dataclass
class FitnessConfig:
    metric: str = "roc_auc"
    # kary (0..1): jak mocno penalizować złe rozmiary/balans
    val_size_penalty: float = 0.0
    class_imbalance_penalty: float = 0.5
    sv_ratio_penalty: float = 0.5  # zostawimy na później, gdy będziemy wyciągać %SV z modelu

def _class_balance_penalty(y_train, y_val) -> float:
    # prosta kara: im bardziej nierówny udział klas, tym większa kara (0..1)
    def imbalance(y):
        vals, cnt = np.unique(y, return_counts=True)
        p = cnt / cnt.sum()
        return 1.0 - p.min()  # 0 idealnie, →1 gdy bardzo źle
    return 0.5 * imbalance(y_train) + 0.5 * imbalance(y_val)

def evaluate_individual(mask, X, y, svm_cfg: SVMConfig, fcfg: FitnessConfig) -> float:
    X_tr, y_tr, X_va, y_va = decode(mask, X, y)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return 0.0  # nie da się trenować sensownie
    base = train_eval_svm(X_tr, y_tr, X_va, y_va, svm_cfg, metric=fcfg.metric)
    # proste kary (skalowanie 0..1)
    penalty = 0.0
    penalty += fcfg.class_imbalance_penalty * _class_balance_penalty(y_tr, y_va)
    # val_size_penalty można dodać: penalizuj zbyt małe/duże walidacje względem cfg.val_fraction
    return max(0.0, base * (1.0 - penalty))


from dataclasses import dataclass
import numpy as np
from ga.models.svm import SVMConfig, train_eval_svm


def evaluate_individual_fixed_train(
    train_mask: np.ndarray,
    X_pool: np.ndarray, y_pool: np.ndarray,
    X_val: np.ndarray,  y_val: np.ndarray,
    svm_cfg: SVMConfig, fcfg: FitnessConfig
) -> float:
    """
    Ocena osobnika: TRAIN = X_pool[train_mask], VAL = stałe (X_val, y_val).
    """
    X_tr = X_pool[train_mask]
    y_tr = y_pool[train_mask]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0

    base = train_eval_svm(X_tr, y_tr, X_val, y_val, svm_cfg, metric=fcfg.metric)

    if fcfg.class_imbalance_penalty > 0.0:
        def imb(y):
            y = np.asarray(y)
            # bezpieczne liczenie częstości
            classes, counts = np.unique(y, return_counts=True)
            p = counts / counts.sum()
            return 1.0 - p.min()
        penalty = fcfg.class_imbalance_penalty * imb(y_tr)
        base = max(0.0, base * (1.0 - penalty))

    return float(base)

