import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from temporal_attribution import classify_profile, temporal_profile
from temporal_benchmark import run


def test_distinguishes_initiation_from_persistence():
    stages = {
        "onset": {"factual": 1.0, "interventions": {"M": 1.0, "F": 0.2}},
        "persistence": {"factual": 1.0, "interventions": {"M": 0.2, "F": 1.0}},
    }
    profile = temporal_profile(stages, ("M", "F"))
    labels = classify_profile(profile, effect_threshold=0.1)
    assert labels == {"M": "maintainer_only", "F": "initiator_only"}


def test_sanjith_pilot_profile():
    stages = {
        "onset": {"factual": 1.0, "interventions": {"M": 1.0, "F": 0.31, "R": 1.0}},
        "persistence": {"factual": 1.0, "interventions": {"M": 1.0, "F": 1.0, "R": 1.0}},
    }
    profile = temporal_profile(stages, ("M", "F", "R"))
    labels = classify_profile(profile)
    assert profile["F"]["onset_effect"] == 0.69
    assert profile["F"]["persistence_effect"] == 0.0
    assert labels["F"] == "initiator_only"
    assert labels["M"] == "inactive"


def test_temporal_benchmark_covers_all_mechanisms():
    rows = run()
    assert len(rows) == 12
    assert {row["sample_size"] for row in rows} == {16, 32, 64}
