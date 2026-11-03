import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scm import Exogenous, TemporalSCM, graph_dot, graph_mermaid


def test_feedback_can_create_persistent_state():
    scm = TemporalSCM()
    onset = scm.evaluate(M_t=1, F_t=1, R_t=1, C_prev=0, exogenous=Exogenous())
    later = scm.evaluate(M_t=0, F_t=0, R_t=0, C_prev=onset["C_t"], exogenous=Exogenous())
    assert onset["C_t"] == 1
    assert onset["A_t"] == 1
    assert later["C_t"] == 1
    assert later["A_t"] == 1


def test_intervening_feedback_blocks_onset_in_declared_model():
    scm = TemporalSCM()
    result = scm.evaluate(
        M_t=1, F_t=1, R_t=1, C_prev=0, exogenous=Exogenous(verdict=0.5),
        intervention={"F_t": 0},
    )
    assert result["C_t"] == 0
    assert result["A_t"] == 0


def test_graph_contains_declared_dependencies():
    dot = graph_dot()
    mermaid = graph_mermaid()
    assert '"F_t" -> "C_t";' in dot
    assert '"C_t" -> "A_t";' in dot
    assert "Ft[F_t] --> Ct[C_t]" in mermaid
