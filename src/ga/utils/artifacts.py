# src/ga/utils/artifacts.py
from pathlib import Path
import json
import numpy as np
from ga.encoding import decode
from ga.models.svm import train_eval_svm

def save_ga_artifacts(history, best_mask, X, y, svm_cfg, metric, log, outdir="artifacts"):
    """Zapisuje wyniki GA i ocenia najlepszy split."""
    Path(f"{outdir}/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"{outdir}/checkpoints").mkdir(parents=True, exist_ok=True)

    # historia fitnessu
    hist_path = Path(f"{outdir}/metrics/history.json")
    hist_path.write_text(json.dumps([float(x) for x in history]), encoding="utf-8")

    # najlepszy chromosom (maska)
    np.save(f"{outdir}/checkpoints/best_mask.npy", best_mask)

    # ewaluacja najlepszego podziału
    X_tr, y_tr, X_va, y_va = decode(best_mask, X, y)
    best_score = train_eval_svm(X_tr, y_tr, X_va, y_va, svm_cfg, metric=metric)

    log.info(f"GA finished. Best fitness (gen best): {history[-1]:.4f}")
    log.info(f"Best-split SVM ({svm_cfg.kernel}) {metric}: {best_score:.4f}")

    return best_score
