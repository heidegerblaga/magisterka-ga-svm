import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga import study_analysis as sa


def test_describe_uses_sample_standard_deviation():
    d = sa.describe([1.0, 2.0, 3.0, 4.0])
    assert d["mean"] == pytest.approx(2.5)
    assert d["sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert (d["min"], d["max"], d["n"]) == (1.0, 4.0, 4)


def test_paired_comparison_aligns_by_seed_and_detects_shift():
    seeds = list(range(42, 62))
    rng = np.random.default_rng(0)
    a = pd.Series(0.97 + 0.002 * rng.standard_normal(20), index=seeds)
    b = a - 0.01
    res = sa.paired_comparison(a, b.iloc[::-1])  # kolejność nie ma znaczenia: wyrównanie po seedach
    assert res["n"] == 20
    assert res["mean_diff"] == pytest.approx(0.01)
    assert res["a_better_count"] == 20 and res["ci95_low"] > 0


def test_validate_series_flags_missing_seeds_and_multi_k_growth():
    seeds = list(range(42, 62))
    rows = pd.DataFrame(
        {"seed": seeds[:-1], "max_k_increases_in_cycle": 1, "job_status": "completed"}
    )
    check = sa.validate_series(rows, seeds)
    assert not check["complete"] and check["missing_seeds"] == [61]
    rows = pd.DataFrame({"seed": seeds, "max_k_increases_in_cycle": 1, "job_status": "completed"})
    assert sa.validate_series(rows, seeds)["complete"]
    rows.loc[3, "max_k_increases_in_cycle"] = 2
    assert not sa.validate_series(rows, seeds)["complete"]
