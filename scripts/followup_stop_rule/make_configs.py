"""Zapisuje szablony konfiguracji badania `followup_stop_rule` (podział bazowy) oraz specyfikację badania.

    configs/followup_stop_rule/<zbiór>/<wariant>.yaml     szablon (seed 42, podział 2026, reguła rozszerzona)
    configs/followup_stop_rule/study_spec.yaml           seedy, podziały, warianty, parametry reguły stopu

Konfiguracje poszczególnych przebiegów (z właściwym podziałem, (C, gamma) i seedem) zapisuje kolejka
(run_queue.py) w artifacts/followup_stop_rule/split_<seed>/<zbiór>/<wariant>/configs/seed_<seed>.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ALGO_SEEDS_BASE, ALGO_SEEDS_BUDGET_CHESS, ALGO_SEEDS_EXTRA, ALGO_SEEDS_PILOT, BASE_SPLIT, BUDGET_FACTOR,
    BUDGET_VARIANTS, CONFIG_ROOT, DATASETS, MAIN_DATASETS, MAIN_VARIANTS, RECHECK_EXPANSIONS, SPLIT_SEEDS, VARIANTS,
    build_config, planned_series, stop_rule_params,
)


def main() -> None:
    for dataset in MAIN_DATASETS + ["magic_full"]:
        variants = MAIN_VARIANTS + (BUDGET_VARIANTS if dataset != "magic_full" else [])
        for variant in variants:
            cfg = build_config(BASE_SPLIT, dataset, variant)
            out = CONFIG_ROOT / dataset / f"{variant}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    spec = {
        "study": "followup_stop_rule",
        "base_split": BASE_SPLIT,
        "split_seeds": SPLIT_SEEDS,
        "algorithm_seeds": {
            "base_split": ALGO_SEEDS_BASE, "extra_splits": ALGO_SEEDS_EXTRA,
            "budget_chessboard": ALGO_SEEDS_BUDGET_CHESS, "magic_full_pilot": ALGO_SEEDS_PILOT,
        },
        "datasets": {k: {"label": v["label"], **stop_rule_params(k)} for k, v in DATASETS.items()},
        "variants": VARIANTS,
        "budget": {"pool_budget_factor": BUDGET_FACTOR, "variants": BUDGET_VARIANTS},
        "fixed_size_source_variant": "evo_mem",
        "perfect_recheck_expansions": RECHECK_EXPANSIONS,
        "planned_series": [
            {"split": s, "dataset": d, "variant": v, "seeds": seeds} for s, d, v, seeds in planned_series(include_pilot=True)
        ],
    }
    (CONFIG_ROOT / "study_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"zapisano szablony w {CONFIG_ROOT}; zaplanowanych serii: {len(spec['planned_series'])}")


if __name__ == "__main__":
    main()
