"""
workstream_output.py — Four workstream outputs for a seeded knowledge_graph.

Usage:
    python workstream_output.py chats.json

Reads:
  - chats.json  (same format as seed_poc3a.py) — for Output 2a transcript summaries
  - Supabase knowledge_graph table              — for Outputs 2b, 2c, 2d

Writes:
  - summaries_poc3a.json  (chat_id → summary text; consumed by poc3a.py)

Outputs printed:
  2a  Chat-level decision-critical summaries (one per chat, from transcript)
  2b  Raw workstream graph export (all nodes, chat_1–chat_6)
  2c  Human-legible workstream summary (LLM over raw graph)
  2d  Query-specific assembled context + LLM answer
"""

import json
import os
import sys
import textwrap
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

MODEL = "claude-haiku-4-5-20251001"

SUMMARIES_PATH = "summaries_poc3a.json"

QUERY_2D = "What is the current recommended hub city and market size for the engagement?"

# ── Prompts ───────────────────────────────────────────────────────────────────
CHAT_SUMMARY_PROMPT = """\
You are extracting decision-critical signal from a consulting chat.
Extract only the following five categories. Exclude frameworks,
process logic, methodology, and anything reconstructible by a cold model.

CLOSED RULINGS: decisions made and no longer open
HARD CONSTRAINTS: non-negotiable boundaries
NUMERICAL ANCHORS: specific figures that carry forward
OPEN ASSUMPTIONS: what is assumed and what would change it
FALSIFIED PATHS: approaches considered and ruled out with reasons

Be concise. Only include what is explicitly stated.

TRANSCRIPT:
{transcript}"""

WORKSTREAM_SUMMARY_PROMPT = """\
You are reviewing the memory PAL has built for a consulting workstream.
Below is the full workstream graph in raw format.

Produce a plain English summary organised as follows:

CONFIRMED DECISIONS
List every decision marked as current (not superseded).
For each: state what was decided, when, and confidence level
(high above 0.85, medium 0.60-0.85, low below 0.60).

SUPERSEDED FACTS
List every fact or assumption that was overridden.
For each: state the original fact, what replaced it, and when.

ACTIVE CONSTRAINTS
List every constraint currently in force.

OPEN ASSUMPTIONS
List every assumption not yet verified or closed.

PERIPHERAL TOPICS
List topics present in the graph not load-bearing to the
current strategic direction.

Do not invent anything not present in the graph.
Do not interpret or recommend. Only describe what the graph contains.

WORKSTREAM GRAPH:
{raw_graph}"""

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


# ── DB helpers ────────────────────────────────────────────────────────────────
def fetch_edges(conn: psycopg2.extensions.connection) -> list[dict[str, Any]]:
    """Fetch all edges for chat_1 through chat_6."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id::text, entity, relationship_type, target_entity,
                   confidence, superseded_by::text, chat_source, created_at
            FROM   knowledge_graph
            WHERE  chat_source IN (
                       'chat_1','chat_2','chat_3',
                       'chat_4','chat_5','chat_6'
                   )
            ORDER  BY chat_source, created_at ASC;
        """)
        return [dict(r) for r in cur.fetchall()]


# ── LLM helper ────────────────────────────────────────────────────────────────
def call_llm(
    client: anthropic.Anthropic,
    user_content: str,
    system: str | None = None,
) -> str:
    kwargs: dict[str, Any] = dict(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": user_content}],
    )
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text.strip()


# ── Pretty printing ───────────────────────────────────────────────────────────
def section(title: str, width: int = 70) -> None:
    bar = "═" * width
    print(f"\n{bar}\n  {title}\n{bar}\n")


def subsection(title: str) -> None:
    print(f"\n── {title} {'─' * (60 - len(title))}")


# ── Output 2a ─────────────────────────────────────────────────────────────────
def output_2a(
    client: anthropic.Anthropic,
    chats: list[dict],
) -> dict[str, str]:
    """
    Summarise each chat transcript into five decision-critical categories.
    Returns {chat_id: summary_text} and saves to summaries_poc3a.json.
    """
    section("OUTPUT 2a — Chat-Level Decision-Critical Summaries")
    summaries: dict[str, str] = {}

    for chat in chats:
        chat_id    = chat["identifier"]
        transcript = chat["transcript"]

        subsection(f"[{chat_id}]")
        prompt = CHAT_SUMMARY_PROMPT.format(transcript=transcript)
        summary = call_llm(client, prompt)
        print(summary)
        summaries[chat_id] = summary

    # Persist for poc3a.py
    with open(SUMMARIES_PATH, "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, ensure_ascii=False)
    print(f"\n[Summaries saved to {SUMMARIES_PATH}]")

    return summaries


# ── Output 2b ─────────────────────────────────────────────────────────────────
def output_2b(edges: list[dict]) -> str:
    """
    Print every edge as:
      entity → relationship_type → target_entity | confidence: X.XX | superseded_by: UUID or NULL
    Returns the raw text block (reused by 2c).
    """
    section("OUTPUT 2b — Raw Workstream Graph Export")

    if not edges:
        msg = "[No edges found for chat_1–chat_6]"
        print(msg)
        return msg

    lines: list[str] = []
    current_chat = None
    for e in edges:
        if e["chat_source"] != current_chat:
            current_chat = e["chat_source"]
            print(f"\n  [{current_chat}]")
            lines.append(f"[{current_chat}]")

        sup = e.get("superseded_by") or "NULL"
        line = (
            f"  {e['entity']} → {e['relationship_type']} → {e['target_entity']}"
            f" | confidence: {float(e['confidence']):.2f}"
            f" | superseded_by: {sup}"
        )
        print(line)
        lines.append(line.strip())

    return "\n".join(lines)


# ── Output 2c ─────────────────────────────────────────────────────────────────
def output_2c(client: anthropic.Anthropic, raw_graph: str) -> None:
    """Pass the full raw graph to the LLM and print a human-legible summary."""
    section("OUTPUT 2c — Human-Legible Workstream Summary")
    prompt = WORKSTREAM_SUMMARY_PROMPT.format(raw_graph=raw_graph)
    print(call_llm(client, prompt))


# ── Output 2d ─────────────────────────────────────────────────────────────────
def output_2d(
    client: anthropic.Anthropic,
    edges: list[dict],
) -> None:
    """
    Graph-retrieve context for the test query, print it, then pass to LLM
    with the context-assembly system prompt.
    """
    section("OUTPUT 2d — Query-Specific Assembled Context")
    print(f"Query: {QUERY_2D}\n")

    assembled = graph_context_from_data(QUERY_2D, edges, top_k=15, hops=2)

    subsection("Assembled context block")
    print(assembled)

    subsection("LLM answer")
    system = CONTEXT_ASSEMBLY_SYSTEM.format(
        assembled_context=assembled,
        user_query=QUERY_2D,
    )
    # The system prompt already embeds context + query; send a minimal user turn
    answer = call_llm(client, QUERY_2D, system=system)
    print(answer)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run all four workstream outputs for the POC 3a pipeline"
    )
    parser.add_argument("json_file", help="Path to JSON file containing chat transcripts")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as fh:
        chats = json.load(fh)

    client = anthropic.Anthropic()

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        edges = fetch_edges(conn)
    finally:
        conn.close()

    # Run outputs in order
    output_2a(client, chats)
    raw_graph = output_2b(edges)
    output_2c(client, raw_graph)
    output_2d(client, edges)


if __name__ == "__main__":
    main()
