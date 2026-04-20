# MCP stdio server - exposes graph query tools to Claude and other agents
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph
from graphify.security import sanitize_label


def _load_graph(graph_path: str) -> nx.Graph:
    try:
        resolved = Path(graph_path).resolve()
        if resolved.suffix != ".json":
            raise ValueError(f"Graph path must be a .json file, got: {graph_path!r}")
        if not resolved.exists():
            raise FileNotFoundError(f"Graph file not found: {resolved}")
        safe = resolved
        data = json.loads(safe.read_text(encoding="utf-8"))
        try:
            return json_graph.node_link_graph(data, edges="links")
        except TypeError:
            return json_graph.node_link_graph(data)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"error: graph.json is corrupted ({exc}). Re-run /graphify to rebuild.", file=sys.stderr)
        sys.exit(1)


def _communities_from_graph(G: nx.Graph) -> dict[int, list[str]]:
    """Reconstruct community dict from community property stored on nodes."""
    communities: dict[int, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        cid = data.get("community")
        if cid is not None:
            communities.setdefault(int(cid), []).append(node_id)
    return communities


def _strip_diacritics(text: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# --- agents-brain fork: EN↔RU matching ------------------------------------
#
# Built-in bidirectional tech-term map. Keep small — only dev/infra terms that
# show up as god-nodes across projects. For everything else, librarian writes
# aliases.json per node (see _load_aliases).

_EN_RU_MAP: dict[str, str] = {
    # Core concepts
    "agent": "агент",
    "agents": "агенты",
    "graph": "граф",
    "node": "узел",
    "nodes": "узлы",
    "edge": "ребро",
    "project": "проект",
    "task": "задача",
    "role": "роль",
    "code": "код",
    "file": "файл",
    "repo": "репо",
    "repository": "репозиторий",
    "branch": "ветка",
    "commit": "коммит",
    "memory": "память",
    "library": "библиотека",
    "module": "модуль",
    "service": "сервис",
    "config": "конфиг",
    # Domain-specific (agents-brain)
    "brain": "мозг",
    "librarian": "библиотекарь",
    "director": "директор",
    "worker": "воркер",
    "head": "хэд",
    "drafter": "драфтер",
    # Actions
    "query": "запрос",
    "write": "запись",
    "read": "чтение",
    "review": "ревью",
    "audit": "аудит",
    "decision": "решение",
    "lesson": "урок",
    "pattern": "паттерн",
    "convention": "конвенция",
}

# Union map: term (either lang, lowercase) → list of equivalents including self.
_TRANSLIT_MAP: dict[str, list[str]] = {}
for _en, _ru in _EN_RU_MAP.items():
    _TRANSLIT_MAP.setdefault(_en, []).extend([_en, _ru])
    _TRANSLIT_MAP.setdefault(_ru, []).extend([_ru, _en])


def _expand_term(term: str) -> list[str]:
    """Return lower-cased variants for a search term — original + EN↔RU
    equivalent if in _TRANSLIT_MAP. Deduplicated, preserves order."""
    term_low = _strip_diacritics(term).lower()
    variants = _TRANSLIT_MAP.get(term_low)
    if variants:
        return list(dict.fromkeys(_strip_diacritics(v).lower() for v in variants))
    return [term_low]


# Aliases cache: module-level, re-read on file mtime change. Librarian writes
# the file on each cycle; MCP picks up updates without restart.
_ALIASES_CACHE: dict[str, list[str]] = {}
_ALIASES_MTIME: float = 0.0


def _aliases_path() -> str:
    return os.environ.get(
        "GRAPHIFY_ALIASES_PATH",
        os.path.expanduser("~/ai-infra/ops/graphify/aliases.json"),
    )


def _load_aliases(path: str) -> dict[str, list[str]]:
    """Parse aliases.json (node_id → [alias strings]). Returns {} on any
    failure — missing file, invalid JSON, wrong schema are all non-fatal."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for node_id, aliases in raw.items():
        if not isinstance(aliases, list):
            continue
        out[node_id] = [
            _strip_diacritics(a).lower()
            for a in aliases if isinstance(a, str) and a
        ]
    return out


def _get_aliases() -> dict[str, list[str]]:
    global _ALIASES_CACHE, _ALIASES_MTIME
    path = _aliases_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return _ALIASES_CACHE
    if mtime > _ALIASES_MTIME:
        _ALIASES_CACHE = _load_aliases(path)
        _ALIASES_MTIME = mtime
    return _ALIASES_CACHE
# --- end agents-brain fork ------------------------------------------------


def _score_nodes(G: nx.Graph, terms: list[str]) -> list[tuple[float, str, list[str]]]:
    """Оценивает узлы графа по похожести на query terms.

    Возвращает tuple (score, node_id, match_paths). match_paths — список
    уникальных путей совпадения (в порядке приоритета label > alias >
    translit > source), по которым term'ы зацепились. Используется для
    «match explainer» в query_graph output — агент видит ПОЧЕМУ узел нашёлся.

    Backward compat: вызывающий код, который разрабатывался под
    (score, nid), может делать `_, nid = ...` через unpack — получит
    ошибку. Все in-tree вызовы обновлены.
    """
    scored = []
    expanded_terms: list[tuple[str, list[str]]] = [
        (_strip_diacritics(t).lower(), _expand_term(t)) for t in terms
    ]
    aliases_map = _get_aliases()
    for nid, data in G.nodes(data=True):
        norm_label = data.get("norm_label") or _strip_diacritics(data.get("label") or "").lower()
        source = (data.get("source_file") or "").lower()
        node_aliases = aliases_map.get(nid, [])
        score = 0.0
        paths: set[str] = set()
        for orig_norm, variants in expanded_terms:
            # Проверяем в порядке приоритета: label > alias > source.
            # Для каждого — перебираем variants; если матч случился на variant
            # отличном от original, помечаем translit (substring через EN↔RU).
            matched = False

            for v in variants:
                if v in norm_label:
                    score += 1.0
                    paths.add("translit" if v != orig_norm else "label")
                    matched = True
                    break
            if matched:
                continue

            for v in variants:
                if node_aliases and any(v in a for a in node_aliases):
                    score += 0.75
                    paths.add("translit" if v != orig_norm else "alias")
                    matched = True
                    break
            if matched:
                continue

            for v in variants:
                if v in source:
                    score += 0.5
                    paths.add("translit" if v != orig_norm else "source")
                    break
        if score > 0:
            # Стабильный порядок: label, alias, translit, source
            order = {"label": 0, "alias": 1, "translit": 2, "source": 3}
            path_list = sorted(paths, key=lambda p: order.get(p, 99))
            scored.append((score, nid, path_list))
    return sorted(scored, reverse=True, key=lambda x: (x[0], x[1]))


def _bfs(G: nx.Graph, start_nodes: list[str], depth: int) -> tuple[set[str], list[tuple]]:
    visited: set[str] = set(start_nodes)
    frontier = set(start_nodes)
    edges_seen: list[tuple] = []
    for _ in range(depth):
        next_frontier: set[str] = set()
        for n in frontier:
            for neighbor in G.neighbors(n):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    edges_seen.append((n, neighbor))
        visited.update(next_frontier)
        frontier = next_frontier
    return visited, edges_seen


def _dfs(G: nx.Graph, start_nodes: list[str], depth: int) -> tuple[set[str], list[tuple]]:
    visited: set[str] = set()
    edges_seen: list[tuple] = []
    stack = [(n, 0) for n in reversed(start_nodes)]
    while stack:
        node, d = stack.pop()
        if node in visited or d > depth:
            continue
        visited.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                stack.append((neighbor, d + 1))
                edges_seen.append((node, neighbor))
    return visited, edges_seen


def _subgraph_to_text(G: nx.Graph, nodes: set[str], edges: list[tuple], token_budget: int = 2000) -> str:
    """Render subgraph as text, cutting at token_budget (approx 3 chars/token)."""
    char_budget = token_budget * 3
    lines = []
    for nid in sorted(nodes, key=lambda n: G.degree(n), reverse=True):
        d = G.nodes[nid]
        line = f"NODE {sanitize_label(d.get('label', nid))} [src={d.get('source_file', '')} loc={d.get('source_location', '')} community={d.get('community', '')}]"
        lines.append(line)
    for u, v in edges:
        if u in nodes and v in nodes:
            raw = G[u][v]
            d = next(iter(raw.values()), {}) if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) else raw
            line = f"EDGE {sanitize_label(G.nodes[u].get('label', u))} --{d.get('relation', '')} [{d.get('confidence', '')}]--> {sanitize_label(G.nodes[v].get('label', v))}"
            lines.append(line)
    output = "\n".join(lines)
    if len(output) > char_budget:
        output = output[:char_budget] + f"\n... (truncated to ~{token_budget} token budget)"
    return output


def _find_node(G: nx.Graph, label: str) -> list[str]:
    """Return node IDs whose label, ID, or alias matches the search term
    (diacritic-insensitive, EN↔RU aware)."""
    variants = _expand_term(label)
    aliases_map = _get_aliases()
    matches: list[str] = []
    for nid, d in G.nodes(data=True):
        norm_label = d.get("norm_label") or _strip_diacritics(d.get("label") or "").lower()
        nid_lower = nid.lower()
        if any(v in norm_label or v == nid_lower for v in variants):
            matches.append(nid)
            continue
        node_aliases = aliases_map.get(nid, [])
        if node_aliases and any(any(v in a for a in node_aliases) for v in variants):
            matches.append(nid)
    return matches


def _suggest_labels(G: nx.Graph, query: str, k: int = 3) -> list[str]:
    """Token-level fuzzy suggestions: для каждого слова query'я находит
    label'ы где есть похожее слово через SequenceMatcher ratio >= 0.7.
    Считаем score labels по числу matched tokens, возвращаем top-k.

    Это NOT замена aliases — aliases уже работали в _score_nodes. Это
    fallback от typo'ов и near-miss'ов: 'librariian' → 'Librarian',
    'repowie' → 'Repowire Mesh Communication', 'pre tool' → 'PreToolUse ...'.
    """
    from difflib import SequenceMatcher

    q_tokens = [
        _strip_diacritics(w).lower()
        for w in query.split()
        if len(w) >= 3
    ]
    if not q_tokens:
        return []

    # Расширяем через EN↔RU map (pow'im 'agent' если искали 'агент')
    expanded: set[str] = set(q_tokens)
    for w in list(q_tokens):
        for v in _TRANSLIT_MAP.get(w, []):
            expanded.add(_strip_diacritics(v).lower())

    # Индекс: label → [token, token, ...] один раз за вызов
    import re as _re
    labels_tokens: list[tuple[str, list[str]]] = []
    seen_labels: set[str] = set()
    for nid, d in G.nodes(data=True):
        label = d.get("label") or nid
        if label in seen_labels:
            continue
        seen_labels.add(label)
        norm = d.get("norm_label") or _strip_diacritics(label).lower()
        tokens = [t for t in _re.split(r"\W+", norm) if len(t) >= 3]
        if tokens:
            labels_tokens.append((label, tokens))

    # Score: для каждого слова query — max ratio против label-words
    scored: list[tuple[float, str]] = []
    for label, tokens in labels_tokens:
        score = 0.0
        for qt in expanded:
            best = 0.0
            for lt in tokens:
                # Быстрый skip для очевидно далёких слов
                if abs(len(lt) - len(qt)) > 3 and qt not in lt and lt not in qt:
                    continue
                r = SequenceMatcher(None, qt, lt).ratio()
                if r > best:
                    best = r
                    if best >= 0.95:
                        break
            if best >= 0.7:
                score += best
        if score > 0:
            scored.append((score, label))
    scored.sort(reverse=True)
    return [label for _, label in scored[:k]]


def _no_match_message(G: nx.Graph, query: str, kind: str = "node") -> str:
    """Унифицированное сообщение о промахе с suggestions."""
    suggestions = _suggest_labels(G, query, k=3)
    base = f"No matching {kind} found for {query!r}."
    if suggestions:
        return base + " Did you mean: " + ", ".join(f"'{s}'" for s in suggestions) + "?"
    return base + " Try: god_nodes() to browse top concepts, or graph_stats() for scope."


def _filter_blank_stdin() -> None:
    """Filter blank lines from stdin before MCP reads it.

    Some MCP clients (Claude Desktop, etc.) send blank lines between JSON
    messages. The MCP stdio transport tries to parse every line as a
    JSONRPCMessage, so a bare newline triggers a Pydantic ValidationError.
    This installs an OS-level pipe that relays stdin while dropping blanks.
    """
    import os
    import threading

    r_fd, w_fd = os.pipe()
    saved_fd = os.dup(sys.stdin.fileno())

    def _relay() -> None:
        try:
            with open(saved_fd, "rb") as src, open(w_fd, "wb") as dst:
                for line in src:
                    if line.strip():
                        dst.write(line)
                        dst.flush()
        except Exception:
            pass

    threading.Thread(target=_relay, daemon=True).start()
    os.dup2(r_fd, sys.stdin.fileno())
    os.close(r_fd)
    sys.stdin = open(0, "r", closefd=False)


def serve(graph_path: str = "graphify-out/graph.json") -> None:
    """Start the MCP server. Requires pip install mcp."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types
    except ImportError as e:
        raise ImportError("mcp not installed. Run: pip install mcp") from e

    G = _load_graph(graph_path)
    communities = _communities_from_graph(G)

    server = Server("graphify")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="query_graph",
                description="Search the knowledge graph using BFS or DFS. Returns relevant nodes and edges as text context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Natural language question or keyword search"},
                        "mode": {"type": "string", "enum": ["bfs", "dfs"], "default": "bfs",
                                 "description": "bfs=broad context, dfs=trace a specific path"},
                        "depth": {"type": "integer", "default": 3, "description": "Traversal depth (1-6)"},
                        "token_budget": {"type": "integer", "default": 2000, "description": "Max output tokens"},
                    },
                    "required": ["question"],
                },
            ),
            types.Tool(
                name="get_node",
                description="Get full details for a specific node by label or ID.",
                inputSchema={
                    "type": "object",
                    "properties": {"label": {"type": "string", "description": "Node label or ID to look up"}},
                    "required": ["label"],
                },
            ),
            types.Tool(
                name="get_neighbors",
                description="Get all direct neighbors of a node with edge details.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "relation_filter": {"type": "string", "description": "Optional: filter by relation type"},
                    },
                    "required": ["label"],
                },
            ),
            types.Tool(
                name="get_community",
                description="Get all nodes in a community by community ID.",
                inputSchema={
                    "type": "object",
                    "properties": {"community_id": {"type": "integer", "description": "Community ID (0-indexed by size)"}},
                    "required": ["community_id"],
                },
            ),
            types.Tool(
                name="god_nodes",
                description="Return the most connected nodes - the core abstractions of the knowledge graph.",
                inputSchema={"type": "object", "properties": {"top_n": {"type": "integer", "default": 10}}},
            ),
            types.Tool(
                name="graph_stats",
                description="Return summary statistics: node count, edge count, communities, confidence breakdown.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="shortest_path",
                description="Find the shortest path between two concepts in the knowledge graph.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source concept label or keyword"},
                        "target": {"type": "string", "description": "Target concept label or keyword"},
                        "max_hops": {"type": "integer", "default": 8, "description": "Maximum hops to consider"},
                    },
                    "required": ["source", "target"],
                },
            ),
        ]

    def _tool_query_graph(arguments: dict) -> str:
        question = arguments["question"]
        mode = arguments.get("mode", "bfs")
        depth = min(int(arguments.get("depth", 3)), 6)
        budget = int(arguments.get("token_budget", 2000))
        terms = [t.lower() for t in question.split() if len(t) > 2]
        scored = _score_nodes(G, terms)
        start_tuples = scored[:3]   # [(score, nid, paths), ...]
        start_nodes = [nid for _, nid, _ in start_tuples]
        if not start_nodes:
            return _no_match_message(G, question, kind="nodes")
        nodes, edges = _dfs(G, start_nodes, depth) if mode == "dfs" else _bfs(G, start_nodes, depth)

        # Match explainer: для каждого start-node выдаём как он нашёлся —
        # substring на label, через alias (aliases.json), через EN↔RU
        # translit, или через source-path. Помогает агенту понять почему
        # именно эти узлы выбраны и как переформулировать в случае noise.
        start_annotated = []
        for score, nid, paths in start_tuples:
            label = G.nodes[nid].get('label', nid)
            path_str = "+".join(paths) if paths else "?"
            start_annotated.append(f"'{label}' [via={path_str} score={score:.2f}]")

        header = (
            f"Traversal: {mode.upper()} depth={depth} | "
            f"{len(nodes)} nodes found\n"
            f"Start ranked: {', '.join(start_annotated)}\n\n"
        )
        return header + _subgraph_to_text(G, nodes, edges, budget)

    def _tool_get_node(arguments: dict) -> str:
        label = arguments["label"]
        matches = _find_node(G, label)
        if not matches:
            return _no_match_message(G, label, kind="node")
        nid = matches[0]
        d = G.nodes[nid]
        return "\n".join([
            f"Node: {d.get('label', nid)}",
            f"  ID: {nid}",
            f"  Source: {d.get('source_file', '')} {d.get('source_location', '')}",
            f"  Type: {d.get('file_type', '')}",
            f"  Community: {d.get('community', '')}",
            f"  Degree: {G.degree(nid)}",
        ])

    def _tool_get_neighbors(arguments: dict) -> str:
        label = arguments["label"].lower()
        rel_filter = arguments.get("relation_filter", "").lower()
        matches = _find_node(G, label)
        if not matches:
            return _no_match_message(G, label, kind="node")
        nid = matches[0]
        lines = [f"Neighbors of {G.nodes[nid].get('label', nid)}:"]
        for neighbor in G.neighbors(nid):
            d = G.edges[nid, neighbor]
            rel = d.get("relation", "")
            if rel_filter and rel_filter not in rel.lower():
                continue
            lines.append(f"  --> {G.nodes[neighbor].get('label', neighbor)} [{rel}] [{d.get('confidence', '')}]")
        return "\n".join(lines)

    def _tool_get_community(arguments: dict) -> str:
        cid = int(arguments["community_id"])
        nodes = communities.get(cid, [])
        if not nodes:
            return f"Community {cid} not found."
        lines = [f"Community {cid} ({len(nodes)} nodes):"]
        for n in nodes:
            d = G.nodes[n]
            lines.append(f"  {d.get('label', n)} [{d.get('source_file', '')}]")
        return "\n".join(lines)

    def _tool_god_nodes(arguments: dict) -> str:
        from .analyze import god_nodes as _god_nodes
        nodes = _god_nodes(G, top_n=int(arguments.get("top_n", 10)))
        lines = ["God nodes (most connected):"]
        lines += [f"  {i}. {n['label']} - {n['degree']} edges" for i, n in enumerate(nodes, 1)]
        return "\n".join(lines)

    def _tool_graph_stats(_: dict) -> str:
        confs = [d.get("confidence", "EXTRACTED") for _, _, d in G.edges(data=True)]
        total = len(confs) or 1
        return (
            f"Nodes: {G.number_of_nodes()}\n"
            f"Edges: {G.number_of_edges()}\n"
            f"Communities: {len(communities)}\n"
            f"EXTRACTED: {round(confs.count('EXTRACTED')/total*100)}%\n"
            f"INFERRED: {round(confs.count('INFERRED')/total*100)}%\n"
            f"AMBIGUOUS: {round(confs.count('AMBIGUOUS')/total*100)}%\n"
        )

    def _tool_shortest_path(arguments: dict) -> str:
        src_scored = _score_nodes(G, [t.lower() for t in arguments["source"].split()])
        tgt_scored = _score_nodes(G, [t.lower() for t in arguments["target"].split()])
        if not src_scored:
            return _no_match_message(G, arguments["source"], kind="source node")
        if not tgt_scored:
            return _no_match_message(G, arguments["target"], kind="target node")
        src_nid, tgt_nid = src_scored[0][1], tgt_scored[0][1]
        max_hops = int(arguments.get("max_hops", 8))
        try:
            path_nodes = nx.shortest_path(G, src_nid, tgt_nid)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return f"No path found between '{G.nodes[src_nid].get('label', src_nid)}' and '{G.nodes[tgt_nid].get('label', tgt_nid)}'."
        hops = len(path_nodes) - 1
        if hops > max_hops:
            return f"Path exceeds max_hops={max_hops} ({hops} hops found)."
        segments = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            edata = G.edges[u, v]
            rel = edata.get("relation", "")
            conf = edata.get("confidence", "")
            conf_str = f" [{conf}]" if conf else ""
            if i == 0:
                segments.append(G.nodes[u].get("label", u))
            segments.append(f"--{rel}{conf_str}--> {G.nodes[v].get('label', v)}")
        return f"Shortest path ({hops} hops):\n  " + " ".join(segments)

    _handlers = {
        "query_graph": _tool_query_graph,
        "get_node": _tool_get_node,
        "get_neighbors": _tool_get_neighbors,
        "get_community": _tool_get_community,
        "god_nodes": _tool_god_nodes,
        "graph_stats": _tool_graph_stats,
        "shortest_path": _tool_shortest_path,
    }

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        handler = _handlers.get(name)
        if not handler:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            return [types.TextContent(type="text", text=handler(arguments))]
        except Exception as exc:
            return [types.TextContent(type="text", text=f"Error executing {name}: {exc}")]

    import asyncio

    async def main() -> None:
        async with stdio_server() as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    _filter_blank_stdin()
    asyncio.run(main())


if __name__ == "__main__":
    graph_path = sys.argv[1] if len(sys.argv) > 1 else "graphify-out/graph.json"
    serve(graph_path)
