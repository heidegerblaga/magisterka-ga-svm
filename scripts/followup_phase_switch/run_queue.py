"""Kolejka przebiegów `followup_phase_switch` (seria × zbiór × wariant × seed), równolegle, z wznowieniem.

Seria v2 musi mieć prefiks ``v2_`` (serie bez prefiksu to zamrożone wyniki v1). Każdy seed = osobna flaga ``--seeds``.

Przykłady:
    python scripts/followup_phase_switch/run_queue.py --series smoke_f010 -d moons -v evo_switch -v rand_switch \\
        --seeds 42 --seeds 43 --seeds 44 --set max_vprime_fraction=0.10 -w 6
    python scripts/followup_phase_switch/run_queue.py --series chess_r1000 -d chessboard -v evo_switch -v rand_switch \\
        --seeds 42 --seeds 43 --set max_vprime_rows=1000

Zapis: artifacts/followup_phase_switch/<seria>/<zbiór>/<wariant>/{configs,logs,jobs}/seed_<seed>.* + runs/.
series_spec.json serii przechowuje nadpisania; uruchomienie tej samej serii z innymi nadpisaniami jest błędem.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import ARTIFACT_ROOT, REPLAY_SOURCE, REPO_ROOT, SPLIT, build_config, series_root  # noqa: E402

THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS")}


def _parse_value(text: str):
    value = yaml.safe_load(text)
    return value


def _check_series(series: str, overrides: dict, split: int = SPLIT) -> None:
    path = ARTIFACT_ROOT / series / "series_spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        spec = json.loads(path.read_text(encoding="utf-8"))
        stored = spec["phase_switch_overrides"]
        if stored != overrides:
            raise click.UsageError(f"seria {series} ma inne nadpisania: {stored} != {overrides}")
        if int(spec.get("split", SPLIT)) != int(split):
            raise click.UsageError(f"seria {series} ma inny podział danych: {spec.get('split', SPLIT)} != {split}")
    else:
        path.write_text(json.dumps({"series": series, "split": int(split), "phase_switch_overrides": overrides,
                                    "created_at": datetime.now().isoformat(timespec="seconds")}, indent=2),
                        encoding="utf-8")


def _run_job(series: str, dataset: str, variant: str, seed: int, overrides: dict, force: bool,
             split: int = SPLIT) -> tuple[str, float]:
    root = series_root(series, dataset, variant)
    job_path = root / "jobs" / f"seed_{seed}.json"
    if job_path.exists() and not force and json.loads(job_path.read_text()).get("status") == "completed":
        return "skipped", 0.0
    cfg = build_config(series, dataset, variant, seed, overrides, split=split)
    for sub in ("configs", "logs", "jobs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cfg_path = root / "configs" / f"seed_{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    t0 = time.time()
    job_path.write_text(json.dumps({"status": "running", "started": datetime.now().isoformat(timespec="seconds")}))
    with (root / "logs" / f"seed_{seed}.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
                              cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **THREAD_ENV})
    status = "completed" if proc.returncode == 0 else "failed"
    dt = time.time() - t0
    job_path.write_text(json.dumps({"status": status, "returncode": proc.returncode, "duration_sec": dt,
                                    "finished": datetime.now().isoformat(timespec="seconds")}))
    return status, dt


@click.command()
@click.option("--series", required=True)
@click.option("--dataset", "-d", "datasets", multiple=True, required=True)
@click.option("--variant", "-v", "variants", multiple=True, required=True)
@click.option("--seeds", type=int, multiple=True, required=True)
@click.option("--set", "sets", multiple=True, help="nadpisanie phase_switch, np. max_vprime_fraction=0.1")
@click.option("--split", type=int, default=SPLIT, show_default=True,
              help="podział danych z followup_stop_rule (2026 = bazowy; 2027-2029 dodatkowe, tylko zbiory zależne od podziału)")
@click.option("--workers", "-w", default=4)
@click.option("--force", is_flag=True)
def main(series, datasets, variants, seeds, sets, split, workers, force):
    overrides = {}
    for item in sets:
        key, _, val = item.partition("=")
        overrides[key.strip()] = _parse_value(val)
    _check_series(series, overrides, split)
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    print(f"[{series}] {len(jobs)} jobów, podział={split}, nadpisania={overrides}, workers={workers}", flush=True)
    # warianty replay startują dopiero po ukończeniu wariantu źródłowego (ich harmonogram jest jego wynikiem)
    passes = [[j for j in jobs if j[1] not in REPLAY_SOURCE], [j for j in jobs if j[1] in REPLAY_SOURCE]]
    for batch in passes:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_run_job, series, d, v, s, overrides, force, split): (d, v, s) for d, v, s in batch}
            for fut in as_completed(futs):
                d, v, s = futs[fut]
                status, dt = fut.result()
                print(f"[{datetime.now():%H:%M:%S}] {series} {d}/{v}/seed{s}: {status} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
