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

    # --- NOWE: stabilność / spójność na wielu walidacjach ---
    # mnożnik kary za rozpiętość wyników na wielu walidacjach (0 = wyłączone)
    consistency_weight: float = 0.0
    # na ile "mini-walidacji" dzielimy zbiór walidacyjny przy ocenie osobnika
    consistency_n_splits: int = 1
    # sposób liczenia rozrzutu pomiędzy walidacjami: "range" | "std" | "var"
    consistency_mode: str = "range"


def _class_balance_penalty(y_train, y_val) -> float:
    """
    Prosta kara: im bardziej nierówny udział klas, tym większa kara (0..1).
    """
    def imbalance(y):
        vals, cnt = np.unique(y, return_counts=True)
        p = cnt / cnt.sum()
        return 1.0 - p.min()  # 0 idealnie, →1 gdy bardzo źle

    return 0.5 * imbalance(y_train) + 0.5 * imbalance(y_val)


def _consistency_penalty(scores: np.ndarray, mode: str = "range") -> float:
    """
    Liczy miarę rozrzutu (niestabilności) wyników na wielu walidacjach.
    scores – np. [score_full_val, score_fold_1, score_fold_2, ...]
    Zwraca wartość >= 0, typowo w przedziale ~[0, 1].
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size < 2:
        return 0.0

    if mode == "std":
        return float(np.std(scores))
    if mode == "var":
        return float(np.var(scores))

    # domyślnie: range = max - min
    return float(scores.max() - scores.min())


def _maybe_apply_consistency(
    base_score: float,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    svm_cfg: SVMConfig,
    fcfg: FitnessConfig,
) -> float:
    """
    Na podstawie fcfg.consistency_* ewaluuje ten sam TRAIN na kilku "mini-walidacjach"
    (podział X_val/y_val na consistency_n_splits części) i dokłada karę za rozrzut.
    """
    if fcfg.consistency_weight <= 0.0 or fcfg.consistency_n_splits <= 1:
        return base_score

    n_val = X_val.shape[0]
    if n_val < 2:
        return base_score

    n_splits = int(max(1, min(fcfg.consistency_n_splits, n_val)))
    if n_splits <= 1:
        return base_score

    idx = np.arange(n_val)
    scores = [float(base_score)]

    for k in range(n_splits):
        fold_mask = (idx % n_splits) == k
        if fold_mask.sum() < 2:
            continue

        y_val_fold = y_val[fold_mask]
        # jeśli w foldzie jest tylko jedna klasa, to pomijamy – nic nam to nie mówi o stabilności
        if len(np.unique(y_val_fold)) < 2:
            continue

        X_val_fold = X_val[fold_mask]
        s = train_eval_svm(
            X_tr, y_tr,
            X_val_fold, y_val_fold,
            svm_cfg,
            metric=fcfg.metric,
        )
        scores.append(float(s))

    if len(scores) < 2:
        return base_score

    spread = _consistency_penalty(np.array(scores), mode=fcfg.consistency_mode)
    # im większa rozpiętość (spread), tym mniejszy finalny fitness
    penalized = base_score * (1.0 - fcfg.consistency_weight * spread)
    return float(max(0.0, penalized))


def evaluate_individual(mask, X, y, svm_cfg: SVMConfig, fcfg: FitnessConfig) -> float:
    """
    Pierwszy GA: maska decyduje, co idzie do TRAIN, a co do VAL (dynamiczna walidacja).
    """
    X_tr, y_tr, X_va, y_va = decode(mask, X, y)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return 0.0  # nie da się trenować sensownie

    base = train_eval_svm(X_tr, y_tr, X_va, y_va, svm_cfg, metric=fcfg.metric)

    # kara za nierównowagę klas
    penalty = 0.0
    if fcfg.class_imbalance_penalty > 0.0:
        penalty += fcfg.class_imbalance_penalty * _class_balance_penalty(y_tr, y_va)

    base = max(0.0, base * (1.0 - penalty))

    # NOWE: kara za niespójność wyników na wielu walidacjach (podział X_va/y_va na foldy)
    base = _maybe_apply_consistency(base, X_tr, y_tr, X_va, y_va, svm_cfg, fcfg)

    return float(base)


def evaluate_individual_fixed_train(
    train_mask: np.ndarray,
    X_pool: np.ndarray, y_pool: np.ndarray,
    X_val: np.ndarray,  y_val: np.ndarray,
    svm_cfg: SVMConfig, fcfg: FitnessConfig
) -> float:
    """
    K-per-class / TRAIN-GA z *ustaloną* walidacją:
      TRAIN = X_pool[train_mask], VAL = stałe (X_val, y_val).

    Tu właśnie "wiele walidacji" realizujemy jako podział X_val/y_val na kilka foldów
    przy liczeniu kary za niespójność.
    """
    X_tr = X_pool[train_mask]
    y_tr = y_pool[train_mask]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0

    base = train_eval_svm(X_tr, y_tr, X_val, y_val, svm_cfg, metric=fcfg.metric)

    # kara za nierównowagę klas w TRAIN
    if fcfg.class_imbalance_penalty > 0.0:
        def imb(y):
            y = np.asarray(y)
            # bezpieczne liczenie częstości
            classes, counts = np.unique(y, return_counts=True)
            p = counts / counts.sum()
            return 1.0 - p.min()

        penalty = fcfg.class_imbalance_penalty * imb(y_tr)
        base = max(0.0, base * (1.0 - penalty))

    # NOWE: kara/bonus za spójność wyników na wielu walidacjach (foldy w ramach X_val)
    base = _maybe_apply_consistency(base, X_tr, y_tr, X_val, y_val, svm_cfg, fcfg)

    return float(base)
