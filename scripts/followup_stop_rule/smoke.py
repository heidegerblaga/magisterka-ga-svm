"""Krótkie testy smoke badania `followup_stop_rule`: każdy wariant na wybranym zbiorze, jeden seed.

Artefakty trafiają do artifacts/followup_stop_rule/_smoke/ (nie mieszają się z seriami badania).
Sprawdzane są niezmienniki: kod wyjścia 0, obecność nowych kolumn, zgodność ze regułą stopu
(stop przez BA=1,0 tylko po spełnieniu obu warunków), maks. jeden wzrost K na cykl, brak wzrostu K przy BA=1,0.
`rand_fixed` dostaje rozmiar V' z przebiegu `evo_mem` TEGO smoke'a (tylko na potrzeby testu).

    python scripts/followup_stop_rule/smoke.py -d moons              # wszystkie 8 wariantów
    python scripts/followup_stop_rule/smoke.py -d chessboard --light # lekki test ścieżki danych Szachownicy
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
from common import ARTIFACT_ROOT, BASE_SPLIT, MAIN_VARIANTS, REPO_ROOT, BUDGET_VARIANTS, build_config  # noqa: E402
from run_queue import THREAD_ENV, find_run_dir  # noqa: E402

SMOKE_ROOT = ARTIFACT_ROOT / "_smoke"
REQUIRED_COLUMNS = [
    "min_vprime_per_class_required", "min_vprime_size_required", "vprime_min_class_count", "min_vprime_size_reached",
    "perfect_rechecks_required", "perfect_rechecks_done", "perfect_gate_pending", "perfect_stop_blocked",
    "perfect_stop_blocked_reason", "stop_decision", "stop_category", "vprime_candidates", "vprime_scored_pool",
    "vprime_increment_rows", "selection_predict_rows", "selection_decision_rows", "val_ga_nominal_subset_rows_scored",
    "diagnostic_scored_rows",
]


def run_one(dataset: str, variant: str, seed: int, split: int, light: bool, fixed_size: int | None):
    cfg = build_config(split, dataset, variant, seed)
    cfg["experiment_name"] = f"smoke_{dataset}_{variant}_seed{seed}"
    root = SMOKE_ROOT / f"split_{split}" / dataset / variant
    cfg["artifacts_root"] = str(root.relative_to(REPO_ROOT))
    if variant == "rand_fixed":
        cfg["vprime"]["fixed_size"] = int(fixed_size)
    if light:  # tylko dla sprawdzenia ścieżek danych/konfiguracji: mały GA i kilka cykli
        cfg["ga"].update({"population_size": 6, "generations": 3})
        cfg["val_ga"].update({"population_size": 6, "generations": 3})
        cfg["daga"]["max_cycles"] = 4
        cfg["data"]["max_rows"] = 0
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / f"seed_{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log = root / f"seed_{seed}.log"
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run([sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
                              cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ, **THREAD_ENV))
    run_dir = find_run_dir(root, cfg["experiment_name"])
    return proc.returncode, run_dir


def check(run_dir: Path, variant: str, cfg_rule: dict) -> list[str]:
    problems = []
    cycles = pd.read_csv(run_dir / "metrics" / "cycles.csv")
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    for c in REQUIRED_COLUMNS:
        if c not in cycles.columns:
            problems.append(f"brak kolumny {c}")
    if (cycles["k_increases_this_cycle"] > 1).any():
        problems.append("wzrost K > 1 na cykl")
    if ((cycles["perfect_on_current_val"].astype(bool)) & (cycles["k_increased_this_cycle"].astype(bool))).any():
        problems.append("K wzrosło przy BA(V')=1,0")
    ps = summary["phase_study"]
    if ps["stop_category"] == "perfect_ba" and variant != "rand_fixed":
        last = cycles.iloc[-1]
        if not bool(last["min_vprime_size_reached"]):
            problems.append("stop BA=1,0 przed osiągnięciem minimalnego rozmiaru V'")
        if int(last["perfect_rechecks_done"]) < int(last["perfect_rechecks_required"]):
            problems.append("stop BA=1,0 bez wymaganego ponownego sprawdzenia")
    if summary.get("phase_study", {}).get("used_full_validation_for_control") is not False:
        problems.append("pełne V steruje")
    return problems


@click.command()
@click.option("--dataset", "-d", default="moons")
@click.option("--split", "-s", default=BASE_SPLIT, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--variants", "-v", multiple=True)
@click.option("--light", is_flag=True)
@click.option("--workers", "-w", default=4)
@click.option("--save", is_flag=True, help="Dopisz wyniki do reports/followup_stop_rule/tables/smoke_results.json")
def main(dataset, split, seed, variants, light, workers, save):
    variants = list(variants) or (MAIN_VARIANTS + BUDGET_VARIANTS)
    first = [v for v in variants if v != "rand_fixed"]
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {v: pool.submit(run_one, dataset, v, seed, split, light, None) for v in first}
        for v, fut in futures.items():
            results[v] = fut.result()
    if "rand_fixed" in variants:
        evo_dir = results.get("evo_mem", (1, None))[1]
        size = json.loads((evo_dir / "metrics" / "summary.json").read_text())["representative_model"]["validation_rows"]
        results["rand_fixed"] = run_one(dataset, "rand_fixed", seed, split, light, size)
    bad = 0
    saved = []
    for v in variants:
        rc, run_dir = results[v]
        if rc != 0 or run_dir is None or not (run_dir / "metrics" / "summary.json").exists():
            print(f"{dataset}/{v}: BŁĄD (rc={rc})")
            bad += 1
            continue
        problems = check(run_dir, v, {})
        summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
        ps = summary["phase_study"]
        saved.append({"dataset": dataset, "variant": v, "split": split, "seed": seed, "light": bool(light), "ok": not problems,
                      "cycles": ps["cycles"], "stop_category": ps["stop_category"], "problems": problems})
        print(f"{dataset}/{v}: OK={not problems} cykle={ps['cycles']} stop={ps['stop_category']} "
              f"|V'|={summary['representative_model']['validation_rows']} K={summary['global_best']['k_per_class']} "
              f"test BA={summary['global_best']['metrics_test']['metric']:.4f} czas={summary['experiment_duration_sec']:.0f}s "
              f"{'; '.join(problems)}")
        bad += int(bool(problems))
    if save:
        from common import TABLES

        TABLES.mkdir(parents=True, exist_ok=True)
        path = TABLES / "smoke_results.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        keep = [r for r in old if (r["dataset"], r["variant"], r["light"], r["split"]) not in {(x["dataset"], x["variant"], x["light"], x["split"]) for x in saved}]
        path.write_text(json.dumps(keep + saved, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
