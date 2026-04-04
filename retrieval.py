"""
retrieval.py — Graph traversal + context assembly for GraphRAG queries.

Two public functions:
    graph_context(query, conn, top_k, hops) -> str
        Traverses the knowledge graph, weights by confidence × recency,
        filters superseded edges, and returns a structured context block.

    flat_context(conn) -> str
        Dumps the full adjacency list (all edges, including superseded)
        as a plain text block — used for the baseline comparison.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

# ── recency decay ─────────────────────────────────────────────────────────────
# Half-life 90 days: weight = e^(-λ·days), λ = ln(2)/90 ≈ 0.0077
_LAMBDA = math.log(2) / 90


def _recency_weight(created_at: datetime) -> float:
    now = datetime.now(timezone.utc)
    days = (now - created_at).total_seconds() / 86400
    return math.exp(-_LAMBDA * days)


def _composite_score(confidence: float, created_at: datetime) -> float:
    return confidence * _recency_weight(created_at)


# ── keyword extraction (lightweight, no NLP dependency) ──────────────────────
_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","need","dare","ought","used",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","mine","yours","hers","ours","theirs",
    "this","that","these","those","what","which","who","whom","whose",
    "when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","just","but","and","or","for","of","to","in",
    "on","at","by","from","with","about","as","into","through","during",
    "before","after","above","below","between","out","up","down","over",
    "under","again","then","once","here","there",
}

def _keywords(query: str, min_len: int = 4) -> list[str]:
    tokens = query.lower().replace("?", " ").replace(",", " ").split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= min_len]


# ── graph traversal (psycopg2 path, calls the SQL function) ──────────────────

def _fetch_graph_rows(
    conn: psycopg2.extensions.connection,
    keywords: list[str],
    hops: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """Call the graph_context() Postgres function and return rows."""
    sql = """
        SELECT
            id::text,
            entity,
            relationship_type,
            target_entity,
            confidence,
            chat_source,
            created_at,
            hop,
            recency_score,
            composite_score
        FROM graph_context(%s::text[], %s)
        ORDER BY composite_score DESC
        LIMIT %s;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (keywords, hops, top_k))
        return [dict(r) for r in cur.fetchall()]


def _fetch_all_rows(conn: psycopg2.extensions.connection) -> list[dict[str, Any]]:
    """Fetch ALL edges (including superseded) for flat baseline."""
    sql = """
        SELECT
            id::text,
            entity,
            relationship_type,
            target_entity,
            confidence,
            superseded_by::text,
            chat_source,
            created_at
        FROM knowledge_graph
        ORDER BY created_at ASC;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(r) for r in cur.fetchall()]


# ── context formatters ────────────────────────────────────────────────────────

def _format_graph_context(rows: list[dict], query: str) -> str:
    if not rows:
        return "[Graph context: no relevant edges found]"

    lines = [
        "=== GRAPH-ASSEMBLED CONTEXT ===",
        f"Query : {query}",
        f"Edges : {len(rows)} (active, confidence-recency weighted, superseded excluded)",
        "",
    ]

    # Group by composite_score tiers: high (>=0.7), medium (>=0.4), low (<0.4)
    tiers: dict[str, list[dict]] = {"HIGH": [], "MED": [], "LOW": []}
    for r in rows:
        score = float(r["composite_score"])
        if score >= 0.70:
            tiers["HIGH"].append(r)
        elif score >= 0.40:
            tiers["MED"].append(r)
        else:
            tiers["LOW"].append(r)

    for tier_name, tier_rows in tiers.items():
        if not tier_rows:
            continue
        lines.append(f"── {tier_name} CONFIDENCE ──────────────────────────")
        for r in tier_rows:
            dt = r["created_at"]
            if isinstance(dt, datetime):
                dt_str = dt.strftime("%Y-%m-%d")
            else:
                dt_str = str(dt)[:10]
            score = float(r["composite_score"])
            conf  = float(r["confidence"])
            lines.append(
                f"  [{r['chat_source']} | {dt_str} | conf={conf:.2f} | score={score:.3f}]"
            )
            lines.append(f"  {r['entity']} --[{r['relationship_type']}]--> {r['target_entity']}")
            lines.append("")

    lines.append("=== END OF GRAPH CONTEXT ===")
    return "\n".join(lines)


def _format_flat_context(rows: list[dict]) -> str:
    if not rows:
        return "[Flat context: no edges found]"

    lines = [
        "=== FLAT ADJACENCY LIST (ALL EDGES, UNFILTERED) ===",
        f"Total edges: {len(rows)}",
        "",
    ]
    for r in rows:
        sup = "(SUPERSEDED)" if r.get("superseded_by") else ""
        dt = r["created_at"]
        dt_str = dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else str(dt)[:10]
        lines.append(
            f"[{r['chat_source']} | {dt_str} | conf={float(r['confidence']):.2f}] {sup}"
        )
        lines.append(f"  {r['entity']} --[{r['relationship_type']}]--> {r['target_entity']}")
    lines.append("")
    lines.append("=== END OF FLAT LIST ===")
    return "\n".join(lines)


# ── public API ────────────────────────────────────────────────────────────────

def graph_context(
    query: str,
    conn: psycopg2.extensions.connection,
    top_k: int = 15,
    hops: int = 2,
) -> str:
    """
    Traverse the knowledge graph for *query*, weight results by
    confidence × recency, exclude superseded edges, and return a
    structured context string ready for inclusion in an LLM prompt.
    """
    keywords = _keywords(query)
    if not keywords:
        keywords = [query[:30]]

    rows = _fetch_graph_rows(conn, keywords, hops, top_k)
    return _format_graph_context(rows, query)


def flat_context(conn: psycopg2.extensions.connection) -> str:
    """
    Return a flat adjacency-list dump of ALL edges (including superseded)
    as a plain-text context block.  Simulates what a naive memory system
    that stores raw chat exports would give an LLM.
    """
    rows = _fetch_all_rows(conn)
    return _format_flat_context(rows)


# ── in-memory versions (no Supabase needed — used by demo_offline.py) ─────────

def graph_context_from_data(
    query: str,
    edges: list[dict],
    top_k: int = 15,
    hops: int = 2,
) -> str:
    """
    Pure-Python version of graph_context that works on a list of dicts
    (same schema as the DB table).  Used for offline demos and unit tests.
    """
    keywords = _keywords(query)

    # Step 1: filter superseded edges
    active = [e for e in edges if not e.get("superseded_by")]

    # Step 2: seed — edges whose entity or target matches a keyword
    def matches_query(edge: dict) -> bool:
        combined = (edge["entity"] + " " + edge["target_entity"]).lower()
        return any(kw in combined for kw in keywords)

    visited_ids = set()
    scored: list[dict] = []

    # BFS up to `hops` hops
    frontier = [e for e in active if matches_query(e)]
    for hop in range(hops + 1):
        next_frontier = []
        for e in frontier:
            if e["id"] in visited_ids:
                continue
            visited_ids.add(e["id"])
            score = _composite_score(e["confidence"], e["created_at"])
            scored.append({**e, "hop": hop, "composite_score": score,
                           "recency_score": _recency_weight(e["created_at"])})
            # Expand neighbours
            for n in active:
                if n["id"] not in visited_ids and hop < hops:
                    if (n["entity"] == e["target_entity"] or
                            n["target_entity"] == e["entity"]):
                        next_frontier.append(n)
        frontier = next_frontier

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return _format_graph_context(scored[:top_k], query)


def flat_context_from_data(edges: list[dict]) -> str:
    """Pure-Python version of flat_context. Works on a list of dicts."""
    return _format_flat_context(sorted(edges, key=lambda e: e["created_at"]))
