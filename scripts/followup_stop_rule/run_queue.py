"""Kolejka uruchomień badania `followup_stop_rule`: (podział × zbiór × wariant × seed), równolegle i z wznowieniem.

Przykłady:
    python scripts/followup_stop_rule/run_queue.py --all --workers 10                 # cały plan (bez pilotażu magic_full)
    python scripts/followup_stop_rule/run_queue.py -s 2026 -d moons -v evo_mem -w 4  # wybrane serie
    python scripts/followup_stop_rule/run_queue.py -s 2026 -d moons -v evo_mem --seeds 42 --dry-run
    python scripts/followup_stop_rule/run_queue.py --all --include-pilot              # + pilotaż pełnego MAGIC

Każdy job zapisuje: artifacts/followup_stop_rule/split_<s>/<zbiór>/<wariant>/{configs,logs,jobs}/seed_<seed>.* oraz zwykły
przebieg w .../runs/. Ukończone joby (status=completed) są pomijane (chyba że --force). Zadania `rand_fixed` startują dopiero po
ukończeniu serii `evo_mem` TEGO SAMEGO podziału i zbioru (rozmiar stałego V' = średni końcowy |V'| z tej serii; wariant
uprzywilejowany). Kolejność: najdroższe joby najpierw (mniejszy „ogon” na końcu).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ARTIFACT_ROOT, BUDGET_TWIN, REPO_ROOT, build_config, planned_series, series_root, seeds_for,
)

PRINT_LOCK = threading.Lock()
LOCK_PATH = ARTIFACT_ROOT / ".queue.lock"
LEGACY_PER_RUN = REPO_ROOT / "reports" / "vprime_study" / "tables" / "per_run.csv"
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS")}


def say(message: str) -> None:
    with PRINT_LOCK:
        print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {message}", flush=True)


def completed_summaries(split: int, dataset: str, variant: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted((series_root(split, dataset, variant) / "runs").glob("*/metrics/summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[int(summary["seed"])] = summary
    return out


def job_status(split: int, dataset: str, variant: str, seed: int) -> str | None:
    path = series_root(split, dataset, variant) / "jobs" / f"seed_{seed}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def mean_final_vprime_size(split: int, dataset: str, seeds: list[int]) -> tuple[int, dict]:
    summaries = completed_summaries(split, dataset, "evo_mem")
    missing = [s for s in seeds if s not in summaries]
    if missing:
        raise RuntimeError(f"rand_fixed(split {split}/{dataset}) wymaga ukończonej serii evo_mem; brakuje seedów: {missing}")
    sizes = [int(summaries[s]["representative_model"]["validation_rows"]) for s in seeds]
    mean = float(np.mean(sizes))
    return int(round(mean)), {"source_variant": "evo_mem", "source_split": split, "source_seeds": seeds,
                              "source_final_vprime_sizes": sizes, "mean": mean}


def build_job_config(split: int, dataset: str, variant: str, seed: int) -> dict:
    cfg = build_config(split, dataset, variant, seed)
    vp = cfg["vprime"]
    if vp.get("mode") == "fixed" and not vp.get("fixed_size"):
        size, provenance = mean_final_vprime_size(split, dataset, seeds_for(split, dataset, "evo_mem"))
        n_classes = 10 if dataset == "digits" else 2
        required = cfg["daga"]["min_vprime_per_class"] * n_classes
        if size < required:
            raise RuntimeError(f"rand_fixed: rozmiar V' {size} < wymagane minimum {required}")
        vp["fixed_size"] = size
        vp["fixed_size_provenance"] = provenance
    return cfg


def cost_table() -> dict[tuple[str, str], float]:
    table: dict[tuple[str, str], float] = {}
    if LEGACY_PER_RUN.exists():
        frame = pd.read_csv(LEGACY_PER_RUN)
        table = frame.groupby(["dataset", "variant"]).duration_sec.mean().to_dict()
    return table


def expected_cost(table, dataset: str, variant: str) -> float:
    if dataset == "magic_full":
        return 6000.0
    base = BUDGET_TWIN.get(variant, variant)
    value = table.get((dataset, base), 120.0)
    return float(value) * (0.85 if variant in BUDGET_TWIN else 1.0) * (1.3 if variant != "rand_fixed" else 1.0)


def priority_tier(dataset: str, variant: str) -> int:
    if dataset == "magic_full":
        return 4
    budget = variant in BUDGET_TWIN
    if dataset == "chessboard":
        return 3 if budget else 2
    return 1 if budget else 0


def find_run_dir(root: Path, experiment_name: str) -> Path | None:
    """Katalog przebiegu: `<znacznik czasu>_<experiment_name>` (opcjonalnie z sufiksem `_NN` przy kolizji)."""
    candidates = []
    for path in (root / "runs").glob(f"*_{experiment_name}*"):
        tail = path.name.split(experiment_name, 1)[1] if experiment_name in path.name else None
        if tail is not None and (tail == "" or (len(tail) == 3 and tail[0] == "_" and tail[1:].isdigit())):
            candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def run_job(split: int, dataset: str, variant: str, seed: int, dry_run: bool) -> tuple[str, str]:
    root = series_root(split, dataset, variant)
    tag = f"split{split}/{dataset}/{variant}/seed={seed}"
    cfg = build_job_config(split, dataset, variant, seed)
    if dry_run:
        rule = {k: v for k, v in cfg["daga"].items() if "vprime" in k or "recheck" in k}
        return tag, f"dry-run (K start={cfg['daga']['k_start']}, vprime={cfg['vprime']}, rule={rule})"
    for sub in ("configs", "logs", "jobs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cfg_path = root / "configs" / f"seed_{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log_path = root / "logs" / f"seed_{seed}.log"
    status_path = root / "jobs" / f"seed_{seed}.json"
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    status_path.write_text(json.dumps({"status": "running", "seed": seed, "started_at": started}), encoding="utf-8")
    env = dict(os.environ, **THREAD_ENV)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            [sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
            cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env,
        )
    duration = time.perf_counter() - t0
    run_dir = find_run_dir(root, cfg["experiment_name"])
    ok = proc.returncode == 0 and run_dir is not None and (run_dir / "metrics" / "summary.json").exists()
    status_path.write_text(
        json.dumps(
            {"status": "completed" if ok else "failed", "seed": seed, "returncode": proc.returncode, "started_at": started,
             "finished_at": datetime.now().isoformat(timespec="seconds"), "wall_duration_sec": round(duration, 2),
             "run_dir": None if run_dir is None else str(run_dir), "config_path": str(cfg_path), "log_path": str(log_path)},
            indent=2),
        encoding="utf-8",
    )
    return tag, f"{'OK' if ok else 'FAILED'} {duration:.0f}s"


def acquire_lock() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        else:
            raise SystemExit(f"Inna kolejka działa (PID {pid}); wyjdź albo usuń {LOCK_PATH}.")
    LOCK_PATH.write_text(str(os.getpid()))


@click.command()
@click.option("--splits", "-s", multiple=True, type=int, help="Ziarna podziału (domyślnie: wszystkie z planu).")
@click.option("--datasets", "-d", multiple=True)
@click.option("--variants", "-v", multiple=True)
@click.option("--seeds", multiple=True, type=int, help="Nadpisuje seedy z planu.")
@click.option("--all", "run_all", is_flag=True, help="Cały plan badania (bez pilotażu magic_full).")
@click.option("--include-pilot", is_flag=True, help="Dołącz pilotaż pełnego MAGIC (magic_full).")
@click.option("--workers", "-w", default=8, show_default=True)
@click.option("--force", is_flag=True, help="Uruchom ponownie nawet ukończone joby.")
@click.option("--dry-run", is_flag=True)
@click.option("--no-lock", is_flag=True, help="Pozwól na drugą kolejkę równolegle (tylko dla ROZŁĄCZNYCH zbiorów jobów).")
def main(splits, datasets, variants, seeds, run_all, include_pilot, workers, force, dry_run, no_lock):
    plan = planned_series(include_pilot=include_pilot or "magic_full" in datasets)
    if not (run_all or datasets or variants or splits):
        raise SystemExit("Podaj --all albo filtry (-s/-d/-v).")
    jobs = []
    for split, dataset, variant, planned_seeds in plan:
        if splits and split not in splits:
            continue
        if datasets and dataset not in datasets:
            continue
        if variants and variant not in variants:
            continue
        for seed in (list(seeds) or planned_seeds):
            jobs.append((split, dataset, variant, seed))
    table = cost_table()
    # priorytet: (0) zbiory bez Szachownicy, serie główne -> (1) serie z budżetem -> (2) Szachownica główne
    # -> (3) Szachownica z budżetem -> (4) pilotaż pełnego MAGIC; w obrębie poziomu najdroższe joby najpierw
    jobs.sort(key=lambda j: (priority_tier(j[1], j[2]), -expected_cost(table, j[1], j[2])))
    pending = [job for job in jobs if force or job_status(*job) != "completed"]
    say(f"serie w planie: {len({j[:3] for j in jobs})}, jobów: {len(jobs)}, do wykonania: {len(pending)}, workers={workers}")
    if not dry_run and not no_lock:
        acquire_lock()

    def deps_ready(job) -> bool:
        split, dataset, variant, _ = job
        if variant != "rand_fixed":
            return True
        need = [(split, dataset, "evo_mem", s) for s in seeds_for(split, dataset, "evo_mem")]
        return all(job_status(*n) == "completed" for n in need)

    failed = 0
    finished = 0
    total = len(pending)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            running = {}
            while pending or running:
                while pending and len(running) < workers:
                    ready = next((j for j in pending if dry_run or deps_ready(j)), None)
                    if ready is None:
                        break
                    pending.remove(ready)
                    running[pool.submit(run_job, *ready, dry_run)] = ready
                if not running:
                    say(f"BLOKADA: {len(pending)} jobów czeka na niespełnione zależności (rand_fixed bez pełnego evo_mem): "
                        f"{sorted({j[:3] for j in pending})}")
                    failed += len(pending)
                    break
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for future in done:
                    job = running.pop(future)
                    try:
                        tag, outcome = future.result()
                    except Exception as exc:  # noqa: BLE001
                        tag, outcome = "/".join(map(str, job)), f"ERROR {exc!r}"
                    finished += 1
                    failed += int("FAILED" in outcome or "ERROR" in outcome)
                    say(f"[{finished}/{total}] {tag}: {outcome}")
    finally:
        if not dry_run and not no_lock and LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(os.getpid()):
            LOCK_PATH.unlink()
    say(f"koniec: {finished - failed} ok, {failed} nieudanych/zablokowanych")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
