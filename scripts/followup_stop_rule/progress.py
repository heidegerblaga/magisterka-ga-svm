"""Podsumowanie postępu kolejki: ukończone/zaplanowane joby per poziom oraz szacunek pozostałego czasu.

    python scripts/followup_stop_rule/progress.py [--workers 10]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUDGET_TWIN, REPO_ROOT, planned_series, series_root  # noqa: E402
from run_queue import priority_tier  # noqa: E402

LEGACY = REPO_ROOT / "reports" / "vprime_study" / "tables" / "per_run.csv"


@click.command()
@click.option("--workers", "-w", default=10)
@click.option("--include-pilot", is_flag=True)
def main(workers, include_pilot):
    legacy = pd.read_csv(LEGACY).groupby(["dataset", "variant"]).duration_sec.mean().to_dict() if LEGACY.exists() else {}
    done = defaultdict(list)  # (dataset, variant) -> czasy wykonania nowych przebiegów
    tiers = defaultdict(lambda: [0, 0, 0.0])  # tier -> [ukończone, zaplanowane, szac. pozostałe core-sekundy]
    failed, running = [], []
    for split, dataset, variant, seeds in planned_series(include_pilot=include_pilot):
        for seed in seeds:
            p = series_root(split, dataset, variant) / "jobs" / f"seed_{seed}.json"
            st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            if st.get("status") == "completed":
                done[(dataset, variant)].append(float(st.get("wall_duration_sec", 0)))
    ratio = {}
    for (ds, v), times in done.items():
        base = legacy.get((ds, v if v not in BUDGET_TWIN else BUDGET_TWIN[v]))
        if base and times:
            ratio[ds] = (sum(times) / len(times)) / base
    for split, dataset, variant, seeds in planned_series(include_pilot=include_pilot):
        tier = priority_tier(dataset, variant)
        for seed in seeds:
            p = series_root(split, dataset, variant) / "jobs" / f"seed_{seed}.json"
            st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            tiers[tier][1] += 1
            if st.get("status") == "completed":
                tiers[tier][0] += 1
                continue
            if st.get("status") == "failed":
                failed.append((split, dataset, variant, seed))
            if st.get("status") == "running":
                running.append((split, dataset, variant, seed))
            times = done.get((dataset, variant))
            if times:
                est = sum(times) / len(times)
            else:
                est = legacy.get((dataset, BUDGET_TWIN.get(variant, variant)), 120.0) * ratio.get(dataset, 1.0)
            tiers[tier][2] += est
    names = {0: "0: zbiory bez Szachownicy, serie główne", 1: "1: zbiory bez Szachownicy, serie z budżetem",
             2: "2: Szachownica, serie główne", 3: "3: Szachownica, serie z budżetem", 4: "4: pilotaż pełnego MAGIC"}
    total_left = 0.0
    for t in sorted(tiers):
        d, n, left = tiers[t]
        total_left += left
        print(f"poziom {names[t]:50s} {d:5d}/{n:<5d} pozostało ≈ {left / 3600:6.1f} godz. CPU = {left / 3600 / workers:5.1f} godz. zegarowo ({workers} procesów)")
    print(f"RAZEM pozostało ≈ {total_left / 3600:.1f} godz. CPU = {total_left / 3600 / workers:.1f} godz. zegarowo")
    print(f"w trakcie: {len(running)}; nieudane: {len(failed)} {failed[:5]}")
    if ratio:
        print("stosunek czasu nowa/stara reguła (zegarowo, wg ukończonych):", {k: round(v, 2) for k, v in ratio.items()})


if __name__ == "__main__":
    main()
