from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from ga.models.svm import SVMConfig, evaluate_fitted_svm, fit_svm_model
from ga.reporting import load_datasets_from_config


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_numpy(value):
    if value is None:
        return None
    return value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)


def svc_from_model(model):
    if hasattr(model, "named_steps") and "svc" in model.named_steps:
        return model.named_steps["svc"]
    return model


def resolve_run_path(path_value: str | Path, summary_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    local = (summary_path.parent / path).resolve()
    if local.exists():
        return local
    return (REPO_ROOT / path).resolve()


def plot_model(
    *,
    output_path: Path,
    title: str,
    X_background: np.ndarray,
    y_background: np.ndarray,
    X_train_model: np.ndarray,
    y_train_model: np.ndarray,
    model,
    support_indices: np.ndarray,
    subset_label: str | None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.unique(y_background)
    colors = {labels[0]: "#3b82f6", labels[1]: "#ef4444"}

    x_min, x_max = X_background[:, 0].min(), X_background[:, 0].max()
    y_min, y_max = X_background[:, 1].min(), X_background[:, 1].max()
    pad_x = 0.04 * (x_max - x_min)
    pad_y = 0.04 * (y_max - y_min)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 420),
        np.linspace(y_min, y_max, 420),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], colors=["#dbeafe", "#fee2e2"], alpha=0.45)
    ax.contour(xx, yy, pred, levels=[0.5], colors="#111827", linewidths=1.2)

    for label in labels:
        idx = y_background == label
        ax.scatter(
            X_background[idx, 0],
            X_background[idx, 1],
            s=4,
            c=colors[label],
            alpha=0.18,
            linewidths=0,
        )

    if subset_label is not None:
        for label in labels:
            idx = y_train_model == label
            ax.scatter(
                X_train_model[idx, 0],
                X_train_model[idx, 1],
                s=34,
                facecolors="none",
                edgecolors=colors[label],
                linewidths=1.2,
            )

    X_sv = X_train_model[support_indices]
    y_sv = y_train_model[support_indices]
    for label in labels:
        idx = y_sv == label
        ax.scatter(
            X_sv[idx, 0],
            X_sv[idx, 1],
            s=42,
            marker="x",
            c="#111827",
            linewidths=1.1,
        )

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[labels[0]], alpha=0.45, markersize=7, label=f"class {labels[0]} (all train)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[labels[1]], alpha=0.45, markersize=7, label=f"class {labels[1]} (all train)"),
        Line2D([0], [0], color="#111827", linewidth=1.2, label="decision boundary"),
        Line2D([0], [0], marker="x", color="#111827", linestyle="none", markersize=7, label="support vector"),
    ]
    if subset_label is not None:
        legend_items.insert(
            2,
            Line2D([0], [0], marker="o", color="#111827", markerfacecolor="none", linestyle="none", markersize=8, label=subset_label),
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.12)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_best_progression(
    *,
    output_path: Path,
    title: str,
    X_background: np.ndarray,
    y_background: np.ndarray,
    masks: np.ndarray,
    svm_cfg: SVMConfig,
    generations: list[int],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.unique(y_background)
    colors = {labels[0]: "#3b82f6", labels[1]: "#ef4444"}

    x_min, x_max = X_background[:, 0].min(), X_background[:, 0].max()
    y_min, y_max = X_background[:, 1].min(), X_background[:, 1].max()
    pad_x = 0.04 * (x_max - x_min)
    pad_y = 0.04 * (y_max - y_min)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 260),
        np.linspace(y_min, y_max, 260),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    cols = 3
    rows = int(np.ceil(len(generations) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, generation in zip(axes, generations):
        gen_idx = min(max(generation, 1), masks.shape[0]) - 1
        mask = masks[gen_idx].astype(bool, copy=False)
        X_subset = X_background[mask]
        y_subset = y_background[mask]

        model, meta = fit_svm_model(X_subset, y_subset, svm_cfg, return_metadata=True)
        pred = model.predict(grid).reshape(xx.shape)
        support_indices = svc_from_model(model).support_
        X_sv = X_subset[support_indices]
        y_sv = y_subset[support_indices]

        ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], colors=["#dbeafe", "#fee2e2"], alpha=0.42)
        ax.contour(xx, yy, pred, levels=[0.5], colors="#111827", linewidths=1.0)

        for label in labels:
            idx = y_background == label
            ax.scatter(
                X_background[idx, 0],
                X_background[idx, 1],
                s=2.5,
                c=colors[label],
                alpha=0.12,
                linewidths=0,
            )

        for label in labels:
            idx = y_subset == label
            ax.scatter(
                X_subset[idx, 0],
                X_subset[idx, 1],
                s=24,
                facecolors="none",
                edgecolors=colors[label],
                linewidths=1.0,
            )

        for label in labels:
            idx = y_sv == label
            ax.scatter(
                X_sv[idx, 0],
                X_sv[idx, 1],
                s=32,
                marker="x",
                c="#111827",
                linewidths=1.0,
            )

        ax.set_title(f"gen {generation} | |T'|={len(X_subset)} | SV={meta['num_support_vectors']}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.1)

    for ax in axes[len(generations):]:
        ax.axis("off")

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[labels[0]], alpha=0.45, markersize=7, label=f"class {labels[0]} (all train)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[labels[1]], alpha=0.45, markersize=7, label=f"class {labels[1]} (all train)"),
        Line2D([0], [0], marker="o", color="#111827", markerfacecolor="none", linestyle="none", markersize=8, label="selected T'"),
        Line2D([0], [0], marker="x", color="#111827", linestyle="none", markersize=7, label="support vector"),
        Line2D([0], [0], color="#111827", linewidth=1.2, label="decision boundary"),
    ]
    fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncols=5, fontsize=9)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_k_trace(output_path: Path, history_path: Path) -> None:
    import matplotlib.pyplot as plt

    payload = read_json(history_path)
    trace = payload.get("trace") or []
    generations = [int(row["generation"]) for row in trace]
    k_values = [int(row["k_per_class"]) for row in trace]
    cs_values = [int(row["cs"]) for row in trace]
    sv_ratios = [float(row["best_sv_ratio"]) for row in trace]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].step(generations, k_values, where="post", color="#1f2937")
    axes[0].set_ylabel("K")
    axes[0].grid(alpha=0.2)
    axes[1].plot(generations, sv_ratios, color="#2563eb")
    axes[1].axhline(0.25, color="#dc2626", linestyle="--", linewidth=1, label="TSV=0.25")
    axes[1].set_ylabel("SV ratio")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    axes[2].step(generations, cs_values, where="post", color="#7c3aed")
    axes[2].axhline(3, color="#dc2626", linestyle="--", linewidth=1, label="Ts=3")
    axes[2].set_xlabel("generation")
    axes[2].set_ylabel("cs")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)
    fig.suptitle("DAGASVM adaptation trace: K, SV ratio, stable-size counter")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def load_run(summary_path: Path):
    summary = read_json(summary_path)
    cfg = read_json(summary_path.parent / "config_snapshot.json")
    datasets = load_datasets_from_config(cfg, apply_limit=False)
    X_train = as_numpy(datasets["train"]["X"])
    y_train = as_numpy(datasets["train"]["y"])
    X_test = as_numpy(datasets["test"]["X"])
    y_test = as_numpy(datasets["test"]["y"])
    svm_section = cfg["svm"]
    svm_cfg = SVMConfig(
        kernel=svm_section["kernel"],
        C=svm_section["C"],
        gamma=svm_section["gamma"],
        class_weight=svm_section.get("class_weight"),
        probability=svm_section.get("probability", False),
        standardize=svm_section.get("standardize", False),
    )
    return summary, cfg, X_train, y_train, X_test, y_test, svm_cfg


def train_subset_from_summary(summary: dict, X_train: np.ndarray, y_train: np.ndarray):
    mask_path = Path(summary["global_best"]["train_mask_path"])
    if not mask_path.is_absolute():
        mask_path = REPO_ROOT / mask_path
    mask = np.load(mask_path).astype(bool)
    return X_train[mask], y_train[mask]


def masks_from_summary(summary: dict, summary_path: Path) -> np.ndarray:
    mask_path = summary["cycles"][0]["train_ga"]["best_masks_per_generation_path"]
    resolved = resolve_run_path(mask_path, summary_path)
    return np.load(resolved).astype(bool)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gasvm-summary", required=True, type=Path)
    parser.add_argument("--dagasvm-summary", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("reports/assets/2026-06-13_dagasvm_corrected"), type=Path)
    args = parser.parse_args(argv)

    outdir = args.output_dir
    gasvm_summary, _, X_train, y_train, X_test, y_test, gasvm_svm_cfg = load_run(args.gasvm_summary)
    dagasvm_summary, _, _, _, _, _, dagasvm_svm_cfg = load_run(args.dagasvm_summary)
    gasvm_masks = masks_from_summary(gasvm_summary, args.gasvm_summary)
    dagasvm_masks = masks_from_summary(dagasvm_summary, args.dagasvm_summary)

    full_model, full_meta = fit_svm_model(X_train, y_train, gasvm_svm_cfg, return_metadata=True)
    full_eval = evaluate_fitted_svm(full_model, X_test, y_test, metric="balanced_accuracy")
    full_support = svc_from_model(full_model).support_
    plot_model(
        output_path=outdir / "full_svm_corrected_gamma_0_002.png",
        title=f"Full SVM, RBF, C=10, gamma=0.002 | test bAcc={full_eval['metric']:.4f}, SV={full_meta['num_support_vectors']}",
        X_background=X_train,
        y_background=y_train,
        X_train_model=X_train,
        y_train_model=y_train,
        model=full_model,
        support_indices=full_support,
        subset_label=None,
    )

    gasvm_X, gasvm_y = train_subset_from_summary(gasvm_summary, X_train, y_train)
    gasvm_model, gasvm_meta = fit_svm_model(gasvm_X, gasvm_y, gasvm_svm_cfg, return_metadata=True)
    gasvm_eval = evaluate_fitted_svm(gasvm_model, X_test, y_test, metric="balanced_accuracy")
    plot_model(
        output_path=outdir / "gasvm_tprime_sv.png",
        title=f"GASVM, K=57 | test bAcc={gasvm_eval['metric']:.4f}, |T'|={len(gasvm_X)}, SV={gasvm_meta['num_support_vectors']}",
        X_background=X_train,
        y_background=y_train,
        X_train_model=gasvm_X,
        y_train_model=gasvm_y,
        model=gasvm_model,
        support_indices=svc_from_model(gasvm_model).support_,
        subset_label="selected T'",
    )
    plot_best_progression(
        output_path=outdir / "gasvm_best_progression.png",
        title="GASVM: best individual over generations",
        X_background=X_train,
        y_background=y_train,
        masks=gasvm_masks,
        svm_cfg=gasvm_svm_cfg,
        generations=[1, 5, 10, 15, 20, 30],
    )

    dagasvm_X, dagasvm_y = train_subset_from_summary(dagasvm_summary, X_train, y_train)
    dagasvm_model, dagasvm_meta = fit_svm_model(dagasvm_X, dagasvm_y, dagasvm_svm_cfg, return_metadata=True)
    dagasvm_eval = evaluate_fitted_svm(dagasvm_model, X_test, y_test, metric="balanced_accuracy")
    plot_model(
        output_path=outdir / "dagasvm_tprime_sv.png",
        title=f"DAGASVM | test bAcc={dagasvm_eval['metric']:.4f}, |T'|={len(dagasvm_X)}, SV={dagasvm_meta['num_support_vectors']}",
        X_background=X_train,
        y_background=y_train,
        X_train_model=dagasvm_X,
        y_train_model=dagasvm_y,
        model=dagasvm_model,
        support_indices=svc_from_model(dagasvm_model).support_,
        subset_label="selected T'",
    )
    plot_best_progression(
        output_path=outdir / "dagasvm_best_progression.png",
        title="DAGASVM: best individual over generations",
        X_background=X_train,
        y_background=y_train,
        masks=dagasvm_masks,
        svm_cfg=dagasvm_svm_cfg,
        generations=[1, 5, 10, 15, 20, 30],
    )

    history_path = args.dagasvm_summary.parent / "cycle_01_daga_train_history.json"
    plot_k_trace(outdir / "dagasvm_k_svratio_cs.png", history_path)

    metrics = {
        "full_svm": {
            "test_balanced_accuracy": full_eval["metric"],
            "test_accuracy": full_eval["all"]["accuracy"],
            "support_vectors": full_meta["num_support_vectors"],
        },
        "gasvm": {
            "test_balanced_accuracy": gasvm_eval["metric"],
            "test_accuracy": gasvm_eval["all"]["accuracy"],
            "training_rows": len(gasvm_X),
            "support_vectors": gasvm_meta["num_support_vectors"],
        },
        "dagasvm": {
            "test_balanced_accuracy": dagasvm_eval["metric"],
            "test_accuracy": dagasvm_eval["all"]["accuracy"],
            "training_rows": len(dagasvm_X),
            "support_vectors": dagasvm_meta["num_support_vectors"],
        },
    }
    (outdir / "diagnostic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
