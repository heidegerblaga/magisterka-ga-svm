"""Zapisuje szablony konfiguracji `followup_phase_switch` v2 i specyfikację badania.

    configs/followup_phase_switch/v2/<zbiór>/<wariant>.yaml   szablon (seed 42, wartości domyślne automatu v2)
    configs/followup_phase_switch/v2/study_spec.yaml          domyślne parametry automatu, warianty, wykonane serie v2

Szablony v1 (configs/followup_phase_switch/<zbiór>/<wariant>.yaml, study_spec.yaml) NIE są nadpisywane.

Szablon `rand_replay` nie istnieje: jego harmonogram (``phase_switch.schedule_replay``) wskazuje cycles.csv przebiegu
evo_switch z tym samym seedem i jest wpisywany przez run_queue.py do konfiguracji przebiegu.
Konfiguracje faktycznie uruchomione: artifacts/followup_phase_switch/<seria>/<zbiór>/<wariant>/configs/seed_<s>.yaml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import (  # noqa: E402
    ARTIFACT_ROOT, CONFIG_ROOT_V2 as CONFIG_ROOT, PHASE_SWITCH_DEFAULTS, REPLAY_SOURCE, SPLIT, V2_PREFIX, VARIANT_LABELS,
    VARIANTS, build_config, fsr,
)


def main() -> None:
    for dataset in fsr.MAIN_DATASETS:
        for variant in VARIANTS:
            if variant in REPLAY_SOURCE or variant == "heur_switch":
                continue  # rand_replay: harmonogram z przebiegu evo; heurystyka poza zakresem etapu v2 (evo vs losowanie)
            cfg = build_config(f"{V2_PREFIX}template", dataset, variant, seed=42)
            out = CONFIG_ROOT / dataset / f"{variant}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    series = {}
    for path in sorted(ARTIFACT_ROOT.glob(f"{V2_PREFIX}*/series_spec.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        datasets = sorted({p.parent.parent.name for p in path.parent.glob("*/*/jobs")})
        series[spec["series"]] = {"phase_switch_overrides": spec["phase_switch_overrides"], "datasets": datasets,
                                  **({"note": spec["note"]} if "note" in spec else {})}
    spec = {
        "study": "followup_phase_switch",
        "version": "v2",
        "algorithm_mode": "dagasvm_phase_switch",
        "split": SPLIT,
        "phase_switch_defaults": PHASE_SWITCH_DEFAULTS,
        "variants": {k: {**v, "label": VARIANT_LABELS[k]} for k, v in VARIANTS.items() if k != "heur_switch"},
        "replay_source": REPLAY_SOURCE,
        "executed_series": series,
    }
    (CONFIG_ROOT / "study_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"zapisano szablony i specyfikację w {CONFIG_ROOT}; serii: {len(series)}")


if __name__ == "__main__":
    main()
