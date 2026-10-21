"""Weryfikacja zgodności wstecznej: stare konfiguracje badania V' (bez nowych parametrów => wartości domyślne)
uruchomione NOWYM kodem muszą dać identyczne trajektorie i metryki jak zapisane przebiegi historyczne.

Porównywane są wszystkie kolumny cycles.csv istniejące w obu plikach (poza czasami) oraz końcowe metryki
(test / pełne V / V') z summary.json. Wynik: reports/followup_stop_rule/tables/legacy_reproduction.csv.

    python scripts/followup_stop_rule/verify_legacy.py -w 8
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARTIFACT_ROOT, LEGACY_ARTIFACTS, REPO_ROOT, TABLES  # noqa: E402
from run_queue import THREAD_ENV  # noqa: E402

VERIFY_ROOT = ARTIFACT_ROOT / "_legacy_reproduction"
DEFAULT_CASES = (
    [("moons", v, s) for v in ("evo_mem", "evo_nomem", "rand_growth", "heur_mem", "rand_fixed") for s in (42, 43, 44)]
    + [("classification2d", v, 42) for v in ("evo_mem", "evo_nomem", "rand_growth", "heur_mem", "rand_fixed")]
    + [("digits", "rand_growth", 42), ("digits", "heur_mem", 43), ("circles", "rand_growth", 42), ("circles", "rand_fixed", 43)]
)


def legacy_run_dir(dataset: str, variant: str, seed: int) -> Path:
    runs = sorted((LEGACY_ARTIFACTS / dataset / variant / "runs").glob(f"*_study_{dataset}_{variant}_s{seed}"))
    return runs[-1]


def rerun(case):
    dataset, variant, seed = case
    cfg = yaml.safe_load((LEGACY_ARTIFACTS / dataset / variant / "configs" / f"seed_{seed}.yaml").read_text(encoding="utf-8"))
    assert "min_vprime_per_class" not in cfg["daga"]  # stara konfiguracja: nowe parametry => wartości domyślne
    root = VERIFY_ROOT / dataset / variant
    cfg["artifacts_root"] = str(root.relative_to(REPO_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / f"seed_{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log = root / f"seed_{seed}.log"
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run([sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
                              cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ, **THREAD_ENV))
    new_dirs = sorted(root.glob(f"runs/*_study_{dataset}_{variant}_s{seed}"))
    return case, proc.returncode, new_dirs[-1] if new_dirs else None


def compare(case, rc, new_dir):
    dataset, variant, seed = case
    old_dir = legacy_run_dir(dataset, variant, seed)
    row = {"dataset": dataset, "variant": variant, "seed": seed, "returncode": rc, "identical": False, "detail": ""}
    if rc != 0 or new_dir is None:
        row["detail"] = "przebieg nieudany"
        return row
    a = pd.read_csv(old_dir / "metrics" / "cycles.csv")
    b = pd.read_csv(new_dir / "metrics" / "cycles.csv")
    cols = [c for c in a.columns if c in b.columns and "duration" not in c and not c.endswith("_time")]
    diffs = [c for c in cols if len(a) != len(b) or not (a[c].fillna(-999).astype(str) == b[c].fillna(-999).astype(str)).all()]
    sa = json.loads((old_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((new_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    for key in ("metrics_test", "metrics_full_validation", "metrics_val"):
        if sa["global_best"][key]["metric"] != sb["global_best"][key]["metric"]:
            diffs.append(f"summary:{key}")
    row.update({"cycles_old": len(a), "cycles_new": len(b), "columns_compared": len(cols),
                "stop_old": sa["phase_study"]["stop_decision"], "stop_new": sb["phase_study"]["stop_decision"],
                "identical": not diffs, "detail": ", ".join(diffs)})
    return row


@click.command()
@click.option("--workers", "-w", default=6)
def main(workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(rerun, DEFAULT_CASES))
    rows = [compare(*r) for r in results]
    frame = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / "legacy_reproduction.csv", index=False)
    print(frame.to_string(index=False))
    n_ok = int(frame.identical.sum())
    print(f"\nzgodnych z historią: {n_ok}/{len(frame)}")
    sys.exit(0 if n_ok == len(frame) else 1)


if __name__ == "__main__":
    main()
