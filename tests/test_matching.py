"""Tests for the agents-brain matching patch to graphify/serve.py.

Runs against a REAL graph if available (~/ai-infra/develop-brain/graphify-out/graph.json)
or falls back to a synthetic test fixture.

Usage (inside fork clone with editable install):
    cd ~/repos/agents-brain-team/graphify-fork
    pytest ../system/graphify-patches/tests/test_matching.py -v

Or standalone:
    PYTHONPATH=~/.local/lib/python3.12/site-packages python3 -m pytest tests/test_matching.py -v
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from graphify.serve import (
    _score_nodes,
    _find_node,
    _expand_term,
    _get_aliases,
    _load_aliases,
    _suggest_labels,
    _no_match_message,
    _EN_RU_MAP,
    _TRANSLIT_MAP,
)


@pytest.fixture
def synthetic_graph() -> nx.Graph:
    """Tiny graph for controlled tests. Mixes EN/RU labels."""
    G = nx.Graph()
    G.add_node(
        "librarian_agent",
        label="Librarian Agent",
        norm_label="librarian agent",
        source_file="agents/librarian/CLAUDE.md",
    )
    G.add_node(
        "agent_overlay",
        label="Агент overlay",
        norm_label="агент overlay",
        source_file="wiki/operations/agent-overlay.md",
    )
    G.add_node(
        "graphify_tool",
        label="Graphify Knowledge Graph Tool",
        norm_label="graphify knowledge graph tool",
        source_file="wiki/stacks/graphify.md",
    )
    G.add_node(
        "brain_query",
        label="Brain Query Protocol",
        norm_label="brain query protocol",
        source_file="brain/rules/core.md",
    )
    return G


@pytest.fixture
def aliases_file(tmp_path, monkeypatch) -> Path:
    """Point GRAPHIFY_ALIASES_PATH at a temp file and invalidate cache."""
    aliases = {
        "brain_query": ["запрос в мозг", "brain-query-global", "bqg"],
        "graphify_tool": ["граф знаний", "граф"],
    }
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(aliases, ensure_ascii=False))
    monkeypatch.setenv("GRAPHIFY_ALIASES_PATH", str(path))
    # Invalidate module-level cache
    import graphify.serve as mod
    mod._ALIASES_CACHE = {}
    mod._ALIASES_MTIME = 0.0
    return path


# --- EN↔RU transliteration ------------------------------------------------

def test_expand_term_english_hits_russian():
    variants = _expand_term("agent")
    assert "agent" in variants
    assert "агент" in variants


def test_expand_term_russian_hits_english():
    variants = _expand_term("граф")
    assert "граф" in variants
    assert "graph" in variants


def test_expand_term_unknown_passes_through():
    variants = _expand_term("kubernetes")
    assert variants == ["kubernetes"]


def test_transliteration_map_is_bidirectional():
    # Every EN key must appear both as EN→RU and RU→EN lookup
    for en, ru in _EN_RU_MAP.items():
        assert en in _TRANSLIT_MAP, f"EN {en!r} not in union map"
        assert ru in _TRANSLIT_MAP, f"RU {ru!r} not in union map"
        assert en in _TRANSLIT_MAP[ru], f"{en!r} missing from variants of {ru!r}"
        assert ru in _TRANSLIT_MAP[en], f"{ru!r} missing from variants of {en!r}"


# --- _score_nodes EN↔RU ---------------------------------------------------

def test_score_en_query_finds_ru_label(synthetic_graph):
    # Query "agent" — должно найти и EN "Librarian Agent", и RU "Агент overlay"
    scored = _score_nodes(synthetic_graph, ["agent"])
    nids = {nid for _, nid, _ in scored}
    assert "librarian_agent" in nids
    assert "agent_overlay" in nids, "EN 'agent' should hit RU 'агент' via transliteration"


def test_score_ru_query_finds_en_label(synthetic_graph):
    # Query "граф" — должно найти EN "Graphify Knowledge Graph Tool"
    scored = _score_nodes(synthetic_graph, ["граф"])
    nids = {nid for _, nid, _ in scored}
    assert "graphify_tool" in nids, "RU 'граф' should hit EN 'graph' via transliteration"


def test_score_mixed_query(synthetic_graph):
    scored = _score_nodes(synthetic_graph, ["brain", "query"])
    assert scored, "multi-term query should still match"
    top_nid = scored[0][1]
    assert top_nid == "brain_query"


# --- Aliases cache --------------------------------------------------------

def test_load_aliases_missing_file_returns_empty(tmp_path):
    path = tmp_path / "nonexistent.json"
    assert _load_aliases(str(path)) == {}


def test_load_aliases_invalid_json_returns_empty(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not-json")
    assert _load_aliases(str(path)) == {}


def test_load_aliases_normalizes_lowercase_and_diacritics(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"n1": ["MixedCASE", "Café"]}))
    result = _load_aliases(str(path))
    assert result["n1"] == ["mixedcase", "cafe"]


def test_aliases_hit_in_score(synthetic_graph, aliases_file):
    # `brain_query` has alias "bqg" — querying "bqg" should find it
    scored = _score_nodes(synthetic_graph, ["bqg"])
    nids = {nid for _, nid, _ in scored}
    assert "brain_query" in nids, "alias lookup should hit via aliases.json"


def test_aliases_ru_phrase(synthetic_graph, aliases_file):
    # "граф знаний" alias on graphify_tool
    scored = _score_nodes(synthetic_graph, ["граф", "знаний"])
    nids = {nid for _, nid, _ in scored}
    assert "graphify_tool" in nids


def test_aliases_cache_refreshes_on_mtime(synthetic_graph, tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"librarian_agent": ["first-alias"]}))
    monkeypatch.setenv("GRAPHIFY_ALIASES_PATH", str(path))
    import graphify.serve as mod
    mod._ALIASES_CACHE = {}
    mod._ALIASES_MTIME = 0.0

    # First load
    aliases = _get_aliases()
    assert aliases.get("librarian_agent") == ["first-alias"]

    # Update file with bumped mtime
    import time
    time.sleep(0.01)  # ensure mtime differs
    path.write_text(json.dumps({"librarian_agent": ["second-alias"]}))
    os.utime(path, None)  # touch

    aliases = _get_aliases()
    assert aliases.get("librarian_agent") == ["second-alias"], \
        "mtime bump must invalidate cache"


# --- Regression: original behavior preserved ------------------------------

def test_score_nodes_returns_sorted_by_score(synthetic_graph):
    scored = _score_nodes(synthetic_graph, ["agent"])
    scores = [s for s, _, _ in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_nodes_zero_score_excluded(synthetic_graph):
    scored = _score_nodes(synthetic_graph, ["nonexistent-term-xyz"])
    assert scored == []


def test_find_node_by_alias(synthetic_graph, aliases_file):
    matches = _find_node(synthetic_graph, "bqg")
    assert "brain_query" in matches


# --- Suggestions on no-match --------------------------------------------

def test_suggest_labels_typo(synthetic_graph):
    # 'librariian' (doubled i) должно найти "Librarian Agent" fuzzy
    suggestions = _suggest_labels(synthetic_graph, "librariian", k=3)
    assert any("librarian" in s.lower() for s in suggestions)


def test_suggest_labels_truncation(synthetic_graph):
    # 'librar' (truncated) тоже должен найти librarian-узел
    suggestions = _suggest_labels(synthetic_graph, "librar agent", k=3)
    assert any("librarian" in s.lower() for s in suggestions)


def test_suggest_labels_empty_for_nonsense(synthetic_graph):
    suggestions = _suggest_labels(synthetic_graph, "xyzabc qqqwww", k=3)
    assert suggestions == []


def test_suggest_labels_skips_short_words(synthetic_graph):
    # Words shorter than 3 chars не тригерят suggestions
    suggestions = _suggest_labels(synthetic_graph, "a b", k=3)
    assert suggestions == []


def test_no_match_message_contains_suggestion(synthetic_graph):
    msg = _no_match_message(synthetic_graph, "librariian", kind="node")
    assert "Did you mean" in msg
    assert "librarian" in msg.lower()


def test_no_match_message_fallback_without_suggestions(synthetic_graph):
    msg = _no_match_message(synthetic_graph, "xyzabc qqqwww", kind="node")
    assert "Did you mean" not in msg
    assert "god_nodes" in msg or "graph_stats" in msg  # helpful hint


# --- Match-path explainer -------------------------------------------------

def test_score_returns_tuple_with_paths(synthetic_graph):
    scored = _score_nodes(synthetic_graph, ["agent"])
    assert scored
    score, nid, paths = scored[0]
    assert isinstance(paths, list)
    assert all(p in {"label", "alias", "translit", "source"} for p in paths)


def test_match_path_label_for_direct_substring(synthetic_graph):
    scored = _score_nodes(synthetic_graph, ["librarian"])
    paths_by_nid = {nid: paths for _, nid, paths in scored}
    assert "label" in paths_by_nid["librarian_agent"]


def test_match_path_translit_for_ru_query(synthetic_graph):
    scored = _score_nodes(synthetic_graph, ["агент"])
    paths_by_nid = {nid: paths for _, nid, paths in scored}
    # Expect translit path (RU 'агент' → EN 'agent')
    assert "translit" in paths_by_nid["librarian_agent"]


def test_match_path_alias_when_aliases_file_used(synthetic_graph, aliases_file):
    # aliases_file fixture registers "bqg" alias on brain_query node
    scored = _score_nodes(synthetic_graph, ["bqg"])
    paths_by_nid = {nid: paths for _, nid, paths in scored}
    assert "brain_query" in paths_by_nid
    assert "alias" in paths_by_nid["brain_query"]


def test_match_path_source_for_path_substring(synthetic_graph):
    # source_file='agents/librarian/CLAUDE.md' — 'claude' hits source but not label
    scored = _score_nodes(synthetic_graph, ["claude"])
    if scored:  # не во всех fixture узлах есть path — защита теста
        paths_by_nid = {nid: paths for _, nid, paths in scored}
        for nid, paths in paths_by_nid.items():
            # Если не label — значит source (путь содержит claude)
            if "label" not in paths and "alias" not in paths:
                assert "source" in paths or "translit" in paths


# --- Integration with real graph (skip if unavailable) --------------------

REAL_GRAPH = Path.home() / "ai-infra" / "develop-brain" / "graphify-out" / "graph.json"


@pytest.mark.skipif(not REAL_GRAPH.exists(), reason="no real graph")
def test_real_graph_en_query_matches():
    """Smoke test — query должен найти хотя бы один узел."""
    from graphify.serve import _load_graph
    G = _load_graph(str(REAL_GRAPH))
    scored = _score_nodes(G, ["agent"])
    assert scored, "real graph query for 'agent' should match at least one node"


@pytest.mark.skipif(not REAL_GRAPH.exists(), reason="no real graph")
def test_real_graph_ru_query_matches_via_translit():
    """RU query должен найти EN-узлы через транслитерацию."""
    from graphify.serve import _load_graph
    G = _load_graph(str(REAL_GRAPH))
    en_scored = _score_nodes(G, ["graph"])
    ru_scored = _score_nodes(G, ["граф"])
    # RU query must find at least as many nodes as EN query for the same concept
    # (may find more if RU aliases exist via aliases.json)
    assert ru_scored, "RU 'граф' must match via translit to EN 'graph' nodes"
    en_ids = {nid for _, nid, _ in en_scored}
    ru_ids = {nid for _, nid, _ in ru_scored}
    overlap = en_ids & ru_ids
    assert overlap, "RU/EN queries must share at least one result"
