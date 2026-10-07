from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# ensure src on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from ga.models.svm import SVMConfig, fit_svm_model, evaluate_fitted_svm
from ga.reporting import load_datasets_from_config


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _savefig(fig, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")


def _plot_timeseries(outdir: Path, trace: list[dict], metric_name: str) -> None:
    import matplotlib.pyplot as plt

    gens = [int(row["generation"]) for row in trace]
    k_values = [int(row.get("k_per_class") or 0) for row in trace]
    best_sv = [row.get("best_num_support_vectors") for row in trace]
    mean_sv = [row.get("mean_num_support_vectors") for row in trace]
    best_fit = [row.get("best_fitness") for row in trace]
    mean_fit = [row.get("mean_fitness") for row in trace]

    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.plot(gens, k_values, linewidth=2)
    ax.set_title("K per generation")
    ax.set_xlabel("generation")
    ax.set_ylabel("k_per_class")
    ax.grid(True, alpha=0.25)
    _savefig(fig, outdir / "k_per_generation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.plot(gens, best_sv, label="best SV")
    ax.plot(gens, mean_sv, label="mean SV", alpha=0.8)
    ax.set_title("Support vectors per generation")
    ax.set_xlabel("generation")
    ax.set_ylabel("#SV")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _savefig(fig, outdir / "support_vectors_per_generation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.plot(gens, best_fit, label=f"best ({metric_name})")
    ax.plot(gens, mean_fit, label=f"mean ({metric_name})", alpha=0.8)
    ax.set_title("Fitness per generation")
    ax.set_xlabel("generation")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _savefig(fig, outdir / "fitness_per_generation.png")
    plt.close(fig)


def _compute_split_metrics_for_masks(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    masks: np.ndarray,
    svm_cfg: SVMConfig,
    metric: str,
    *,
    every: int,
) -> list[dict]:
    rows: list[dict] = []
    for gen_idx in range(masks.shape[0]):
        if every > 1 and ((gen_idx + 1) % every != 0) and gen_idx != masks.shape[0] - 1:
            continue

        mask = masks[gen_idx].astype(bool, copy=False)
        X_tr = X_pool[mask]
        y_tr = y_pool[mask]
        if X_tr.shape[0] < 2:
            continue

        clf, _ = fit_svm_model(X_tr, y_tr, svm_cfg, return_metadata=True)
        train_eval = evaluate_fitted_svm(clf, X_tr, y_tr, metric=metric)
        val_eval = evaluate_fitted_svm(clf, X_val, y_val, metric=metric)
        test_eval = None if X_test is None else evaluate_fitted_svm(clf, X_test, y_test, metric=metric)

        rows.append(
            {
                "generation": int(gen_idx + 1),
                "k_per_class_est": int(X_tr.shape[0] // max(1, int(np.unique(y_tr).size))),
                "train_metric": float(train_eval["metric"]),
                "val_metric": float(val_eval["metric"]),
                "test_metric": None if test_eval is None else float(test_eval["metric"]),
                "train_accuracy": float(train_eval["all"]["accuracy"]),
                "val_accuracy": float(val_eval["all"]["accuracy"]),
                "test_accuracy": None if test_eval is None else float(test_eval["all"]["accuracy"]),
                "num_support_vectors": int(train_eval["num_support_vectors"]),
            }
        )
    return rows


def _plot_split_metrics(outdir: Path, rows: list[dict], metric_name: str) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    gens = [int(r["generation"]) for r in rows]
    train_m = [r["train_metric"] for r in rows]
    val_m = [r["val_metric"] for r in rows]
    test_m = [r["test_metric"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.plot(gens, train_m, label="train")
    ax.plot(gens, val_m, label="validation")
    if any(v is not None for v in test_m):
        ax.plot(gens, [np.nan if v is None else v for v in test_m], label="test")
    ax.set_title(f"{metric_name} over generations (best individual)")
    ax.set_xlabel("generation")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _savefig(fig, outdir / "metric_train_val_test.png")
    plt.close(fig)

    train_acc = [r["train_accuracy"] for r in rows]
    val_acc = [r["val_accuracy"] for r in rows]
    test_acc = [r["test_accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.plot(gens, train_acc, label="train T'")
    ax.plot(gens, val_acc, label="validation")
    if any(v is not None for v in test_acc):
        ax.plot(gens, [np.nan if v is None else v for v in test_acc], label="test")
    ax.set_title("Accuracy over generations (best individual)")
    ax.set_xlabel("generation")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _savefig(fig, outdir / "accuracy_train_val_test.png")
    plt.close(fig)


def _plot_best_individual_frames(
    outdir: Path,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    masks: np.ndarray,
    svm_cfg: SVMConfig,
    *,
    every: int,
    mesh_step: float | None,
) -> None:
    import matplotlib.pyplot as plt

    if X_pool.ndim != 2 or X_pool.shape[1] != 2:
        return

    frames_dir = outdir / "best_individual_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    x_min, x_max = float(X_pool[:, 0].min()), float(X_pool[:, 0].max())
    y_min, y_max = float(X_pool[:, 1].min()), float(X_pool[:, 1].max())
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    step = float(mesh_step) if mesh_step is not None else max((x_max - x_min), (y_max - y_min)) / 200.0
    step = max(step, 1e-6)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()]

    unique_labels = np.unique(np.asarray(y_pool))
    palette = plt.cm.get_cmap("tab10", max(2, int(unique_labels.size)))

    for gen_idx in range(masks.shape[0]):
        if every > 1 and ((gen_idx + 1) % every != 0) and gen_idx != masks.shape[0] - 1:
            continue

        mask = masks[gen_idx].astype(bool, copy=False)
        X_tr = X_pool[mask]
        y_tr = y_pool[mask]
        if X_tr.shape[0] < 2:
            continue

        clf = fit_svm_model(X_tr, y_tr, svm_cfg, return_metadata=False)
        Z = clf.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
        ax.contourf(xx, yy, Z, alpha=0.25, levels=len(unique_labels), cmap=palette)
        ax.scatter(X_pool[:, 0], X_pool[:, 1], c="#999999", s=6, alpha=0.35, linewidths=0)

        for li, label in enumerate(unique_labels):
            idx = np.asarray(y_tr) == label
            ax.scatter(
                X_tr[idx, 0],
                X_tr[idx, 1],
                s=18,
                alpha=0.9,
                color=palette(li),
                edgecolors="black",
                linewidths=0.2,
                label=f"class {label}",
            )

        ax.set_title(f"Best individual | gen {gen_idx + 1} | |T'|={int(X_tr.shape[0])}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.15)

        _savefig(fig, frames_dir / f"best_individual_gen_{gen_idx + 1:03d}.png")
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot per-generation traces for a single run.")
    parser.add_argument("--summary", required=True, type=Path, help="Path to summary.json")
    parser.add_argument("--every", type=int, default=1, help="Plot/evaluate every N generations (default: 1)")
    parser.add_argument("--mesh-step", type=float, default=None, help="Decision-boundary mesh step (2D only).")
    args = parser.parse_args(argv)

    summary_path: Path = args.summary
    summary = _read_json(summary_path)
    cfg_path = summary_path.parent / "config_snapshot.json"
    cfg = _read_json(cfg_path) if cfg_path.exists() else {}

    metric = (cfg.get("fitness") or {}).get("metric", "balanced_accuracy")
    svm_section = cfg.get("svm") or {}
    svm_cfg = SVMConfig(
        kernel=svm_section.get("kernel", "rbf"),
        C=svm_section.get("C", 1.0),
        gamma=svm_section.get("gamma", "scale"),
        class_weight=svm_section.get("class_weight", "balanced"),
        probability=svm_section.get("probability", True),
        standardize=svm_section.get("standardize", True),
    )

    # Maski są zapisane względem puli treningowej użytej w runie (z limitem max_rows).
    # Do wizualizacji przebiegu musimy wczytać tę samą pulę.
    datasets_pool = load_datasets_from_config(cfg, apply_limit=True)
    X_pool = datasets_pool["train"]["X"]
    y_pool = datasets_pool["train"]["y"]

    # Ewaluacja metryk jest bardziej informacyjna na pełnych VAL/TEST, więc domyślnie
    # wczytujemy pełne zbiory do oceny (train nadal pochodzi z puli z runu).
    datasets_eval = load_datasets_from_config(cfg, apply_limit=False)
    X_val = datasets_eval["validation"]["X"]
    y_val = datasets_eval["validation"]["y"]
    X_test = datasets_eval["test"]["X"]
    y_test = datasets_eval["test"]["y"]

    def to_numpy(value):
        if value is None:
            return None
        return value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)

    X_pool = to_numpy(X_pool)
    y_pool = to_numpy(y_pool)
    X_val = to_numpy(X_val)
    y_val = to_numpy(y_val)
    X_test = to_numpy(X_test)
    y_test = to_numpy(y_test)

    # run_root w tym projekcie to rodzic folderu `metrics/`
    run_root = summary_path.parent.parent
    outdir = run_root / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    cycles = summary.get("cycles") or []
    if not cycles:
        return 0

    # zakładamy 1 cykl (GASVM/DAGASVM) z wpisanym trace i zapisanymi maskami
    cycle = cycles[0]
    train_ga = cycle.get("train_ga") or {}
    history_path = Path(train_ga.get("history_path") or "")
    best_masks_path = Path(train_ga.get("best_masks_per_generation_path") or "")

    def resolve_path(path: Path) -> Path:
        if path.is_absolute():
            return path
        local = (summary_path.parent / path).resolve()
        if local.exists():
            return local
        return (REPO_ROOT / path).resolve()

    history_path = resolve_path(history_path)
    best_masks_path = resolve_path(best_masks_path)

    history_payload = _read_json(history_path) if history_path.exists() else {}
    trace = history_payload.get("trace") or []
    masks = np.load(best_masks_path) if best_masks_path.exists() else np.zeros((0, X_pool.shape[0]), dtype=bool)

    if trace:
        _plot_timeseries(outdir, trace, metric_name=str(metric))

    if masks.size:
        rows = _compute_split_metrics_for_masks(
            X_pool,
            y_pool,
            X_val,
            y_val,
            X_test,
            y_test,
            masks,
            svm_cfg,
            metric=str(metric),
            every=max(1, int(args.every)),
        )
        (outdir / "metrics_over_generations.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _plot_split_metrics(outdir, rows, metric_name=str(metric))
        _plot_best_individual_frames(
            outdir,
            X_pool,
            y_pool,
            masks,
            svm_cfg,
            every=max(1, int(args.every)),
            mesh_step=args.mesh_step,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
