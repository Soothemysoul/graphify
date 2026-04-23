"""Regression test for beads-c6r: __main__.py:1147 tuple-unpack drift.

_score_nodes (serve.py) returns list[tuple[float, str, list[str]]] (3-tuple).
The CLI 'query' command unpacked as 2-tuple → ValueError: too many values to unpack.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest
from networkx.readwrite import json_graph


def _make_graph_file(tmp_path: Path) -> Path:
    G = nx.Graph()
    G.add_node("n1", label="extract", source_file="a.py", source_location="L1", community=0)
    G.add_node("n2", label="cluster", source_file="b.py", source_location="L1", community=0)
    G.add_edge("n1", "n2", relation="calls", confidence="EXTRACTED")
    data = json_graph.node_link_data(G, edges="links")
    gf = tmp_path / "graph.json"
    gf.write_text(json.dumps(data), encoding="utf-8")
    return gf


def test_query_cmd_handles_3tuple_from_score_nodes(tmp_path, monkeypatch):
    """CLI 'query' must unpack 3-tuple (score, nid, aliases) returned by _score_nodes.

    RED (before fix):  ValueError: too many values to unpack (expected 2)
    GREEN (after fix): main() completes without error
    """
    graph_file = _make_graph_file(tmp_path)

    # _score_nodes returns 3-tuple since the EN↔RU matching patch (beads-c6r root cause)
    mock_scored: list[tuple[float, str, list[str]]] = [
        (0.9, "n1", ["label"]),
        (0.7, "n2", ["source"]),
    ]

    monkeypatch.setattr(
        sys, "argv", ["graphify", "query", "extract", "--graph", str(graph_file)]
    )

    with patch("graphify.serve._score_nodes", return_value=mock_scored):
        from graphify.__main__ import main

        # Must NOT raise ValueError about tuple unpacking
        main()
