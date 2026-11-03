"""Export the declared temporal SCM graph for reports."""
from pathlib import Path

from scm import graph_dot, graph_mermaid

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "temporal_scm.dot").write_text(graph_dot(), encoding="utf-8")
    (RESULTS / "temporal_scm.mmd").write_text(graph_mermaid(), encoding="utf-8")
    print("Wrote temporal_scm.dot and temporal_scm.mmd to", RESULTS)


if __name__ == "__main__":
    main()
