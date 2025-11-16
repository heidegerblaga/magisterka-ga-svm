import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# dołącz src/ do sys.path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.utils.random import set_global_seed
from ga.encoding import KPerClassConfig
from ga.fitness import FitnessConfig
from ga.loop.engine import GAConfig, EarlyStopping, run_ga_k_per_class
from ga.models.svm import SVMConfig


def main():
    set_global_seed(42)

    # 1) SYNTHETYCZNE DANE 2D
    # make_moons: dwie "półksiężycowe" klasy, fajne do wizualizacji granicy
    X, y = make_moons(n_samples=600, noise=0.25, random_state=42)

    # Podział:
    #  - pierwsze 300 próbek -> pula do TRAIN (GA wybiera k/klasę)
    #  - ostatnie 300 próbek -> STAŁA walidacja
    X_pool, y_pool = X[:300], y[:300]
    X_val, y_val = X[300:], y[300:]

    # 2) KONFIGURACJE
    k_per_class = 40  # ile próbek na każdą klasę ma wybrać GA do TRAIN
    kcfg = KPerClassConfig(k_per_class=k_per_class, random_state=42)

    svm_cfg = SVMConfig(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
    )

    fcfg = FitnessConfig(
        metric="balanced_accuracy",
        class_imbalance_penalty=0.0,
    )

    gcfg = GAConfig(
        population_size=40,
        generations=20,
        tournament_k=3,
        crossover_rate=0.8,
        mutation_rate=0.05,
        elitism=2,
        early_stopping=EarlyStopping(patience=5, min_delta=1e-3),
    )

    # 3) ODPAŁ GA (k na klasę, walidacja stała)
    history, best_train_mask = run_ga_k_per_class(
        X_pool=X_pool,
        y_pool=y_pool,
        X_val=X_val,
        y_val=y_val,
        kcfg=kcfg,
        svm_cfg=svm_cfg,
        fcfg=fcfg,
        gcfg=gcfg,
        seed=42,
    )

    print(f"GA finished. Best fitness (gen): {history[-1]:.4f}")
    print(f"Liczność TRAIN per klasa (powinno być {k_per_class}):")
    vals, cnts = np.unique(y_pool[best_train_mask], return_counts=True)
    print(dict(zip(vals, cnts)))

    # 4) TRENING SVM NA NAJLEPSZYM PODZIALE (do wizualizacji)
    X_tr_best = X_pool[best_train_mask]
    y_tr_best = y_pool[best_train_mask]

    clf_vis = SVC(
        kernel=svm_cfg.kernel,
        C=svm_cfg.C,
        gamma=svm_cfg.gamma,
        class_weight=svm_cfg.class_weight,
        probability=svm_cfg.probability,
        random_state=0,
    ).fit(X_tr_best, y_tr_best)

    # 5) GRID DO GRANICY DECYZYJNEJ
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    Z = clf_vis.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 6) RYSOWANIE
    plt.figure(figsize=(8, 6))

    # tło: granica decyzyjna SVM
    cs = plt.contourf(xx, yy, Z, levels=20, alpha=0.3)
    plt.colorbar(cs, label="decision function")

    # pula, która NIE weszła do TRAIN (pool \ k) – przygaszona
    mask_not_selected = ~best_train_mask
    plt.scatter(
        X_pool[mask_not_selected, 0],
        X_pool[mask_not_selected, 1],
        s=20,
        alpha=0.2,
        label="pool (nie wybrane do k)",
        marker="o",
        edgecolors="none",
    )

    # próbki wybrane do TRAIN (k na klasę) – wyróżnione
    plt.scatter(
        X_tr_best[:, 0],
        X_tr_best[:, 1],
        s=40,
        c=y_tr_best,
        cmap="coolwarm",
        label=f"TRAIN (k={k_per_class} na klasę)",
        marker="o",
        edgecolors="k",
        linewidths=0.5,
    )

    # stała walidacja – jako X
    plt.scatter(
        X_val[:, 0],
        X_val[:, 1],
        s=40,
        c=y_val,
        cmap="coolwarm",
        label="VAL (stała walidacja)",
        marker="x",
        linewidths=1.5,
    )

    plt.legend()
    plt.title("2D demo: GA wybiera k próbek na klasę (TRAIN), SVM granica na stałej walidacji")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
