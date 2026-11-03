import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from policy_experiment import _best, run_policy_experiment


def test_policy_experiment_has_both_temporal_stages():
    rows = run_policy_experiment()
    assert {row["stage"] for row in rows} == {"onset", "persistence"}
    assert {row["parameter_set"] for row in rows} == {
        "pilot_facing", "moderate_learning", "message_sensitive"
    }


def test_feedback_prevents_onset_and_state_reset_recovers_persistence():
    best = _best(run_policy_experiment())
    assert best["pilot_facing:onset"]["policy"] == "remove_feedback"
    assert best["pilot_facing:persistence"]["policy"] == "reset_persistent_state"
