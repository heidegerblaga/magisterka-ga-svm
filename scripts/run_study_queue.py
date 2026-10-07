"""Kolejka uruchomień badania V': (zbiór × wariant × seed) wykonywana równolegle, z możliwością wznowienia.

Przykłady:
    python scripts/run_study_queue.py --datasets chessboard --variants evo_mem --workers 10
    python scripts/run_study_queue.py --datasets moons circles --variants evo_mem evo_nomem rand_growth heur_mem
    python scripts/run_study_queue.py --datasets moons --variants rand_fixed   # po ukończeniu evo_mem
    python scripts/run_study_queue.py --datasets moons --variants evo_mem --seeds 42 43 --dry-run

Każdy job zapisuje: artifacts/study/<zbiór>/<wariant>/{configs,logs,jobs}/seed_<seed>.* oraz zwykły run
w artifacts/study/<zbiór>/<wariant>/runs/. Ukończone joby (status=completed) są pomijane, chyba że --force.
Wariant rand_fixed wymaga ukończonej serii evo_mem: rozmiar stałego losowego V' = średni końcowy rozmiar
V' z tej serii (zaokrąglony); wartość i jej źródło trafiają do konfiguracji każdego przebiegu.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "study"
STUDY_ROOT = REPO_ROOT / "artifacts" / "study"
PRINT_LOCK = threading.Lock()


def _say(message: str) -> None:
    with PRINT_LOCK:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def load_spec() -> dict:
    return yaml.safe_load((CONFIG_ROOT / "study_spec.yaml").read_text(encoding="utf-8"))


def series_root(dataset: str, variant: str) -> Path:
    return STUDY_ROOT / dataset / variant


def completed_summaries(dataset: str, variant: str) -> dict[int, dict]:
    """seed -> summary.json dla ukończonych przebiegów serii (ostatni przebieg na seed)."""
    out: dict[int, dict] = {}
    for path in sorted((series_root(dataset, variant) / "runs").glob("*/metrics/summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[int(summary["seed"])] = summary
    return out


def mean_final_vprime_size(dataset: str, seeds: list[int], source_variant: str) -> tuple[int, dict]:
    summaries = completed_summaries(dataset, source_variant)
    missing = [s for s in seeds if s not in summaries]
    if missing:
        raise RuntimeError(
            f"rand_fixed({dataset}) wymaga ukończonej serii {source_variant}; brakuje seedów: {missing}"
        )
    sizes = [int(summaries[s]["representative_model"]["validation_rows"]) for s in seeds]
    mean = float(np.mean(sizes))
    return int(round(mean)), {
        "source_variant": source_variant,
        "source_seeds": seeds,
        "source_final_vprime_sizes": sizes,
        "mean": mean,
    }


def build_job_config(dataset: str, variant: str, seed: int, spec: dict) -> dict:
    cfg = yaml.safe_load((CONFIG_ROOT / dataset / f"{variant}.yaml").read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    cfg["experiment_name"] = f"study_{dataset}_{variant}_s{seed}"
    cfg["artifacts_root"] = str(series_root(dataset, variant).relative_to(REPO_ROOT))
    vp = cfg["vprime"]
    if vp.get("mode") == "fixed" and not vp.get("fixed_size"):
        size, provenance = mean_final_vprime_size(dataset, spec["seeds"], spec["fixed_size_source_variant"])
        vp["fixed_size"] = size
        vp["fixed_size_provenance"] = provenance
    return cfg


def run_job(dataset: str, variant: str, seed: int, spec: dict, force: bool, dry_run: bool) -> tuple[str, str]:
    root = series_root(dataset, variant)
    status_path = root / "jobs" / f"seed_{seed}.json"
    tag = f"{dataset}/{variant}/seed={seed}"
    if status_path.exists() and not force:
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if status == "completed":
            return tag, "skipped"
        if status == "running":  # inna kolejka właśnie wykonuje ten job (po awarii: użyj --force)
            return tag, "skipped (running elsewhere)"
    cfg = build_job_config(dataset, variant, seed, spec)
    if dry_run:
        return tag, f"dry-run (K start={cfg['daga']['k_start']}, vprime={cfg['vprime']})"
    for sub in ("configs", "logs", "jobs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cfg_path = root / "configs" / f"seed_{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log_path = root / "logs" / f"seed_{seed}.log"
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    status_path.write_text(json.dumps({"status": "running", "seed": seed, "started_at": started}), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            [sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"), "-c", str(cfg_path)],
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    duration = time.perf_counter() - t0
    run_dirs = sorted((root / "runs").glob(f"*_study_{dataset}_{variant}_s{seed}*"))
    ok = proc.returncode == 0 and bool(run_dirs) and (run_dirs[-1] / "metrics" / "summary.json").exists()
    status_path.write_text(
        json.dumps(
            {
                "status": "completed" if ok else "failed",
                "seed": seed,
                "returncode": proc.returncode,
                "started_at": started,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "wall_duration_sec": round(duration, 2),
                "run_dir": str(run_dirs[-1]) if run_dirs else None,
                "config_path": str(cfg_path),
                "log_path": str(log_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tag, f"{'OK' if ok else 'FAILED'} {duration:.0f}s"


@click.command()
@click.option("--datasets", "-d", multiple=True, required=True)
@click.option("--variants", "-v", multiple=True, required=True)
@click.option("--seeds", "-s", multiple=True, type=int, help="Domyślnie: wszystkie seedy ze specyfikacji.")
@click.option("--workers", "-w", default=8, show_default=True)
@click.option("--force", is_flag=True, help="Uruchom ponownie nawet ukończone joby.")
@click.option("--dry-run", is_flag=True)
def main(datasets, variants, seeds, workers, force, dry_run):
    spec = load_spec()
    seeds = list(seeds) or list(spec["seeds"])
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    _say(f"jobs={len(jobs)} workers={workers} seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})")
    failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, d, v, s, spec, force, dry_run): (d, v, s) for d, v, s in jobs}
        for future in as_completed(futures):
            try:
                tag, outcome = future.result()
            except Exception as exc:  # noqa: BLE001
                tag, outcome = "/".join(map(str, futures[future])), f"ERROR {exc!r}"
            done += 1
            failed += int("FAILED" in outcome or "ERROR" in outcome)
            _say(f"[{done}/{len(jobs)}] {tag}: {outcome}")
    _say(f"koniec: {len(jobs) - failed} ok, {failed} nieudanych")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
