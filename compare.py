"""
compare.py — Run the GraphRAG vs flat-list comparison against Claude.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python compare.py

The script:
  1. Connects to Supabase (psycopg2).
  2. Assembles a graph-traversal context block for the test query.
  3. Assembles a flat adjacency-list context block for the same data.
  4. Sends two separate Claude API calls with identical system prompts
     but different context blocks.
  5. Prints both answers side-by-side for comparison.
"""

import os
import sys
import textwrap

import anthropic
import psycopg2
import psycopg2.extras

from retrieval import graph_context

# ── config ────────────────────────────────────────────────────────────────────
DB_CONFIG = dict(
    host="db.hezkypxgfbcwiqprpoyp.supabase.co",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="P4l4TheWin@2",
)

MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for POC comparisons

TEST_QUERY = (
    "What is the current recommended market entry strategy for Germany, "
    "including market size, legal entity setup, and best hub city?"
)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior strategy consultant.
    The user will provide a context block drawn from a project memory system,
    followed by a question.
    Answer concisely using ONLY the information in the context block.
    If data was updated or superseded, reflect the most current information.
    Highlight any contradictions or uncertainty you detect in the context.
    Do not invent information that is not in the context.
""").strip()


# ── flat export (raw edges only — no PAL intelligence) ───────────────────────
def flat_context_raw(conn: psycopg2.extensions.connection) -> str:
    """
    Dump every edge as a plain triple: entity → relationship_type → target_entity.
    No confidence scores, no timestamps, no SUPERSEDED labels, no chat source.
    This is what a naive adjacency-list export looks like.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT entity, relationship_type, target_entity "
            "FROM knowledge_graph ORDER BY created_at ASC;"
        )
        rows = cur.fetchall()

    lines = ["=== FLAT ADJACENCY LIST ==="]
    for r in rows:
        lines.append(f"{r['entity']} → {r['relationship_type']} → {r['target_entity']}")
    lines.append("=== END ===")
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────
def ask_claude(context_block: str, query: str) -> str:
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    user_message = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{query}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ── pretty print ──────────────────────────────────────────────────────────────
def banner(title: str) -> str:
    line = "═" * 70
    return f"\n{line}\n  {title}\n{line}"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    print("Connecting to Supabase …")
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        print("Assembling graph context …")
        g_ctx = graph_context(TEST_QUERY, conn, top_k=15, hops=2)

        print("Assembling flat context …")
        f_ctx = flat_context_raw(conn)
    finally:
        conn.close()

    print(banner("GRAPH CONTEXT BLOCK"))
    print(g_ctx)

    print(banner("FLAT CONTEXT BLOCK"))
    print(f_ctx)

    print(banner("CALLING CLAUDE — GRAPH CONTEXT"))
    graph_answer = ask_claude(g_ctx, TEST_QUERY)
    print(graph_answer)

    print(banner("CALLING CLAUDE — FLAT CONTEXT"))
    flat_answer = ask_claude(f_ctx, TEST_QUERY)
    print(flat_answer)

    print(banner("SIDE-BY-SIDE COMPARISON"))
    print(f"\nQUERY: {TEST_QUERY}\n")
    print("── GRAPH ANSWER ─────────────────────────────────────────────────────")
    print(graph_answer)
    print("\n── FLAT ANSWER ──────────────────────────────────────────────────────")
    print(flat_answer)
    print("\n" + "═" * 70)

    # Key differences to check
    print("\n── SCORING CRITERIA (manually verify) ──────────────────────────────")
    checks = [
        ("Market size", "Graph: EUR 20B (latest). Flat: may mention all three (15B/18B/20B)"),
        ("Hub city", "Graph: Munich (confirmed). Flat: may mention both Berlin and Munich"),
        ("Supersession awareness", "Graph: cites current value. Flat: may present conflicting data"),
        ("Confidence propagation", "Graph: high-confidence facts ranked higher"),
    ]
    for criterion, expectation in checks:
        print(f"  [{criterion}] {expectation}")


if __name__ == "__main__":
    main()
