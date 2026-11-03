import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from causal_estimands import direct_effect, mediated_state_effect, minimal_interventions, total_effect
from scm import StructuralParameters, TemporalSCM


def test_feedback_has_mediated_effect_in_onset_model():
    scm = TemporalSCM()
    inputs = {"M_t": 1, "F_t": 1, "R_t": 1, "C_prev": 0}
    assert total_effect(scm, inputs, "F_t", samples=2000) > 0.5
    assert mediated_state_effect(scm, inputs, "F_t", samples=2000) > 0.5
    assert abs(direct_effect(scm, inputs, "F_t", samples=2000)) < 0.1


def test_direct_message_effect_can_be_represented_separately():
    scm = TemporalSCM(StructuralParameters(message_accept_effect=0.4))
    inputs = {"M_t": 1, "F_t": 0, "R_t": 0, "C_prev": 0}
    assert direct_effect(scm, inputs, "M_t", samples=2000) > 0.25
    assert abs(mediated_state_effect(scm, inputs, "M_t", samples=2000)) < 0.1


def test_minimal_prevention_and_creation_interventions():
    scm = TemporalSCM()
    inputs = {"M_t": 1, "F_t": 1, "R_t": 1, "C_prev": 0}
    assert minimal_interventions(scm, inputs, ("M_t", "F_t", "R_t"), 0) == [("F_t",)]
    assert minimal_interventions(scm, inputs, ("M_t", "F_t", "R_t"), 1) == [()]
