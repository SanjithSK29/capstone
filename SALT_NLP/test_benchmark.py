import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import REAL_PILOT, _sample_table, run_benchmark


def test_sanjith_pilot_values_are_preserved():
    assert REAL_PILOT[0]["minus_F"] == 0.31
    assert REAL_PILOT[0]["minus_all"] == 0.25
    assert REAL_PILOT[1]["minus_F"] == 1.00


def test_sample_table_is_complete_and_reproducible():
    first = _sample_table("partial_feedback", 16, __import__("random").Random(7))
    second = _sample_table("partial_feedback", 16, __import__("random").Random(7))
    assert first == second
    assert len(first) == 8
    assert all(n == 16 for _, n in first.values())


def test_benchmark_has_all_scenarios_and_sample_sizes():
    rows, pilot = run_benchmark()
    assert len(rows) == 5 * 4
    assert len(pilot) == 4
    assert {row["sample_size"] for row in rows} == {8, 16, 32, 64}
