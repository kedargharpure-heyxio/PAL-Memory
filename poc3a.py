"""
poc3a.py — Seven-call comparison: cold baseline → chat summaries → graph-augmented.

Prerequisites:
    1. workstream_output.py has been run → summaries_poc3a.json exists
    2. knowledge_graph is seeded (seed_poc3a.py)
    3. ANTHROPIC_API_KEY is set (or replace the placeholder below)

Usage:
    python poc3a.py

Seven calls against:
    'What is the current recommended hub city and market size for the engagement?'

    Call 0   — Cold, no context, no system prompt  (baseline zero)
    Call 1A  — Chat 1 summary only, no system prompt
    Call 1B  — Chat 1 summary + graph retrieval, with context-assembly system prompt
    Call 2A  — Chat 2 summary only, no system prompt
    Call 2B  — Chat 2 summary + graph retrieval, with context-assembly system prompt
    Call 3A  — Chat 3 summary only, no system prompt
    Call 3B  — Chat 3 summary + graph retrieval, with context-assembly system prompt
"""

import json
import os
import sys
from typing import Any

import anthropic
import psycopg2
import psycopg2.extras

from retrieval import graph_context_from_data

# ── API key ───────────────────────────────────────────────────────────────────
api_key = 'YOUR_ANTHROPIC_API_KEY_HERE'
os.environ['ANTHROPIC_API_KEY'] = api_key

# ── Supabase connection ───────────────────────────────────────────────────────
DB_CONFIG = dict(
    host="db.hezkypxgfbcwiqprpoyp.supabase.co",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="P4l4TheWin@2",
)

MODEL        = "claude-haiku-4-5-20251001"
SUMMARIES_PATH = "summaries_poc3a.json"

TEST_QUERY = "What is the current recommended hub city and market size for the engagement?"

# ── System prompts ────────────────────────────────────────────────────────────
# Used for all B calls
CONTEXT_ASSEMBLY_SYSTEM = """\
You are answering a query on behalf of a consulting team.
The context below contains memory assembled from prior work
on this engagement. Use it as follows:

- CONFIRMED DECISIONS: treat as final, do not reopen or hedge
- ACTIVE CONSTRAINTS: treat as non-negotiable boundaries
- CURRENT VALUES: where a value supersedes an earlier one,
  use only the current value, do not reference the older one
- OPEN ASSUMPTIONS: flag these explicitly in your answer
  if relevant to the query
- FALSIFIED PATHS: do not resurface these as options

If the context does not contain sufficient information to
answer the query, say so explicitly rather than inferring
or fabricating.

CONTEXT:
{assembled_context}

QUERY:
{user_query}"""


# ── DB helper ─────────────────────────────────────────────────────────────────
def fetch_all_edges(conn: psycopg2.extensions.connection) -> list[dict[str, Any]]:
    """Fetch all edges from knowledge_graph for in-memory graph retrieval."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id::text, entity, relationship_type, target_entity,
                   confidence, superseded_by::text, chat_source, created_at
            FROM   knowledge_graph
            ORDER  BY created_at ASC;
        """)
        return [dict(r) for r in cur.fetchall()]


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(
    client: anthropic.Anthropic,
    user_content: str,
    system: str | None = None,
) -> str:
    kwargs: dict[str, Any] = dict(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": user_content}],
    )
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text.strip()


# ── Pretty printing ───────────────────────────────────────────────────────────
def banner(label: str, width: int = 70) -> None:
    bar = "═" * width
    print(f"\n{bar}\n  {label}\n{bar}\n")


def divider(label: str) -> None:
    print(f"\n── {label} {'─' * max(0, 64 - len(label))}")


# ── The seven calls ───────────────────────────────────────────────────────────
def run_calls(
    client: anthropic.Anthropic,
    summaries: dict[str, str],
    edges: list[dict],
) -> dict[str, str]:
    """
    Execute all seven API calls and return {label: answer}.
    """
    results: dict[str, str] = {}

    # Pre-build graph context (same retrieval for all B calls)
    graph_ctx = graph_context_from_data(TEST_QUERY, edges, top_k=15, hops=2)

    # ── Call 0: cold, no context ───────────────────────────────────────────
    print("Running Call 0 (cold baseline) …")
    results["Call 0 — Cold (no context)"] = call_llm(client, TEST_QUERY)

    # ── Calls 1A / 1B ─────────────────────────────────────────────────────
    summary_1 = summaries.get("chat_1", "[chat_1 summary not found]")

    print("Running Call 1A …")
    results["Call 1A — Chat 1 summary only"] = call_llm(
        client,
        f"CONTEXT:\n{summary_1}\n\nQUESTION:\n{TEST_QUERY}",
    )

    print("Running Call 1B …")
    assembled_1b = f"CHAT 1 SUMMARY:\n{summary_1}\n\nGRAPH CONTEXT:\n{graph_ctx}"
    system_1b = CONTEXT_ASSEMBLY_SYSTEM.format(
        assembled_context=assembled_1b,
        user_query=TEST_QUERY,
    )
    results["Call 1B — Chat 1 + graph retrieval"] = call_llm(
        client, TEST_QUERY, system=system_1b
    )

    # ── Calls 2A / 2B ─────────────────────────────────────────────────────
    summary_2 = summaries.get("chat_2", "[chat_2 summary not found]")

    print("Running Call 2A …")
    results["Call 2A — Chat 2 summary only"] = call_llm(
        client,
        f"CONTEXT:\n{summary_2}\n\nQUESTION:\n{TEST_QUERY}",
    )

    print("Running Call 2B …")
    assembled_2b = f"CHAT 2 SUMMARY:\n{summary_2}\n\nGRAPH CONTEXT:\n{graph_ctx}"
    system_2b = CONTEXT_ASSEMBLY_SYSTEM.format(
        assembled_context=assembled_2b,
        user_query=TEST_QUERY,
    )
    results["Call 2B — Chat 2 + graph retrieval"] = call_llm(
        client, TEST_QUERY, system=system_2b
    )

    # ── Calls 3A / 3B ─────────────────────────────────────────────────────
    summary_3 = summaries.get("chat_3", "[chat_3 summary not found]")

    print("Running Call 3A …")
    results["Call 3A — Chat 3 summary only"] = call_llm(
        client,
        f"CONTEXT:\n{summary_3}\n\nQUESTION:\n{TEST_QUERY}",
    )

    print("Running Call 3B …")
    assembled_3b = f"CHAT 3 SUMMARY:\n{summary_3}\n\nGRAPH CONTEXT:\n{graph_ctx}"
    system_3b = CONTEXT_ASSEMBLY_SYSTEM.format(
        assembled_context=assembled_3b,
        user_query=TEST_QUERY,
    )
    results["Call 3B — Chat 3 + graph retrieval"] = call_llm(
        client, TEST_QUERY, system=system_3b
    )

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    # Load summaries written by workstream_output.py
    if not os.path.exists(SUMMARIES_PATH):
        print(
            f"ERROR: {SUMMARIES_PATH} not found.\n"
            f"Run workstream_output.py first to generate chat summaries.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(SUMMARIES_PATH, "r", encoding="utf-8") as fh:
        summaries: dict[str, str] = json.load(fh)

    client = anthropic.Anthropic()

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        edges = fetch_all_edges(conn)
    finally:
        conn.close()

    print(f"\nLoaded {len(summaries)} chat summary/ies from {SUMMARIES_PATH}")
    print(f"Loaded {len(edges)} graph edge(s) from Supabase")
    print(f"\nQuery: {TEST_QUERY}\n")

    results = run_calls(client, summaries, edges)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    banner("COMPARISON — All Seven Calls")
    print(f"Query: {TEST_QUERY}\n")

    for label, answer in results.items():
        divider(label)
        print(answer)

    # ── Alignment table ───────────────────────────────────────────────────────
    banner("ALIGNMENT TABLE")
    hub_kw    = ["munich", "berlin", "hamburg", "frankfurt", "düsseldorf"]
    size_kw   = ["20b", "20 b", "18b", "18 b", "15b", "15 b",
                 "eur 20", "eur 18", "eur 15", "€20", "€18", "€15"]

    print(f"  {'Call':<40} {'Hub city detected':<22} {'Market size detected'}")
    print("  " + "─" * 76)
    for label, answer in results.items():
        low = answer.lower()
        hub  = next((k.title() for k in hub_kw  if k in low), "—")
        size = next((k.upper() for k in size_kw if k in low), "—")
        print(f"  {label:<40} {hub:<22} {size}")

    print()


if __name__ == "__main__":
    main()
