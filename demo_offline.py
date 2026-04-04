"""
demo_offline.py — Self-contained GraphRAG vs flat-list POC demo.

Runs entirely without a Supabase connection.
All graph data is embedded in-memory (same rows as seed.py would insert).
Calls the real Anthropic API if ANTHROPIC_API_KEY is set;
otherwise prints the context blocks and synthetic answer snippets so
the quality difference is still visible.

Usage:
    python demo_offline.py                          # full run (needs API key)
    python demo_offline.py --no-api                 # context blocks only
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import datetime, timezone

from retrieval import graph_context_from_data, flat_context_from_data

# ── In-memory graph data (mirrors seed.py) ────────────────────────────────────
def _ts(year, month, day, hour=12):
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


# Each dict matches the knowledge_graph table schema.
# superseded_by is set by label below, then resolved to the target dict reference.
_RAW_EDGES = [
    # ── CHAT 1 (2024-10-15) ────────────────────────────────────────────────
    # A — corroborated fact (1st mention)
    dict(id="cap_req_c1",
         entity="GmbH Formation",
         relationship_type="REQUIRES_MINIMUM_CAPITAL",
         target_entity="EUR 25,000",
         confidence=0.90, superseded_by=None,
         chat_source="chat_1", created_at=_ts(2024, 10, 15, 8),
         attributes={"note": "Confirmed via HGB §5a", "type": "legal_requirement"}),

    # B — first (lowest) market size estimate
    dict(id="mkt_size_c1",
         entity="German B2B Software Market",
         relationship_type="HAS_ESTIMATED_ANNUAL_REVENUE",
         target_entity="EUR 15B",
         confidence=0.65, superseded_by="mkt_size_c2",
         chat_source="chat_1", created_at=_ts(2024, 10, 15, 9),
         attributes={"source": "analyst estimate", "year": 2024}),

    # C — open assumption: Berlin hub
    dict(id="berlin_hub_c1",
         entity="TechFlow SaaS Market Entry",
         relationship_type="ASSUMES_PRIMARY_HUB",
         target_entity="Berlin",
         confidence=0.60, superseded_by="munich_confirmed_c3",
         chat_source="chat_1", created_at=_ts(2024, 10, 15, 10),
         attributes={"status": "open_assumption", "rationale": "startup ecosystem density"}),

    dict(id="dsgvo_c1",
         entity="German B2B Software Market",
         relationship_type="MANDATES_COMPLIANCE_WITH",
         target_entity="DSGVO/GDPR",
         confidence=0.95, superseded_by=None,
         chat_source="chat_1", created_at=_ts(2024, 10, 15, 11),
         attributes={"type": "regulatory_requirement"}),

    dict(id="vat_c1",
         entity="GmbH Formation",
         relationship_type="REQUIRES_REGISTRATION_WITH",
         target_entity="German Trade Register (Handelsregister)",
         confidence=0.95, superseded_by=None,
         chat_source="chat_1", created_at=_ts(2024, 10, 15, 11),
         attributes={"type": "legal_process"}),

    # ── CHAT 2 (2024-11-02) ────────────────────────────────────────────────
    # A — corroborated fact (2nd mention, higher confidence)
    dict(id="cap_req_c2",
         entity="GmbH Formation",
         relationship_type="REQUIRES_MINIMUM_CAPITAL",
         target_entity="EUR 25,000",
         confidence=0.95, superseded_by=None,
         chat_source="chat_2", created_at=_ts(2024, 11, 2),
         attributes={"note": "Cross-verified with Bundesjustizamt", "type": "legal_requirement"}),

    # B — second market size (supersedes chat_1's)
    dict(id="mkt_size_c2",
         entity="German B2B Software Market",
         relationship_type="HAS_ESTIMATED_ANNUAL_REVENUE",
         target_entity="EUR 18B",
         confidence=0.85, superseded_by="mkt_size_c3",
         chat_source="chat_2", created_at=_ts(2024, 11, 2, 9),
         attributes={"source": "IDC Germany 2024 report", "year": 2024}),

    dict(id="munich_evidence_c2",
         entity="Munich",
         relationship_type="HAS_HIGHER_ENTERPRISE_CUSTOMER_CONCENTRATION_THAN",
         target_entity="Berlin",
         confidence=0.75, superseded_by=None,
         chat_source="chat_2", created_at=_ts(2024, 11, 2, 10),
         attributes={"sectors": ["automotive", "industrial", "fintech"]}),

    dict(id="sales_cycle_c2",
         entity="German B2B Enterprise Sales",
         relationship_type="HAS_TYPICAL_CYCLE_LENGTH",
         target_entity="6–12 months",
         confidence=0.80, superseded_by=None,
         chat_source="chat_2", created_at=_ts(2024, 11, 2, 11),
         attributes={"type": "operational_assumption"}),

    dict(id="local_partner_c2",
         entity="TechFlow SaaS Market Entry",
         relationship_type="BENEFITS_FROM",
         target_entity="Local Reseller / VAR Partnership",
         confidence=0.78, superseded_by=None,
         chat_source="chat_2", created_at=_ts(2024, 11, 2, 12),
         attributes={"priority": "high"}),

    # ── CHAT 3 (2024-12-05) ────────────────────────────────────────────────
    # B — canonical market size (supersedes chat_2's)
    dict(id="mkt_size_c3",
         entity="German B2B Software Market",
         relationship_type="HAS_ESTIMATED_ANNUAL_REVENUE",
         target_entity="EUR 20B",
         confidence=0.90, superseded_by=None,
         chat_source="chat_3", created_at=_ts(2024, 12, 5, 9),
         attributes={"source": "Gartner Germany IT Spend 2025 forecast", "year": 2025}),

    # C — closed assumption: Munich confirmed, Berlin closed
    dict(id="munich_confirmed_c3",
         entity="TechFlow SaaS Market Entry",
         relationship_type="SELECTS_PRIMARY_HUB",
         target_entity="Munich",
         confidence=0.90, superseded_by=None,
         chat_source="chat_3", created_at=_ts(2024, 12, 5, 10),
         attributes={"status": "closed_decision", "closes_assumption": "berlin_hub_c1"}),

    dict(id="staffing_c3",
         entity="TechFlow SaaS Market Entry",
         relationship_type="RECOMMENDS_INITIAL_HEADCOUNT",
         target_entity="3–5 local hires (sales + customer success)",
         confidence=0.80, superseded_by=None,
         chat_source="chat_3", created_at=_ts(2024, 12, 5, 11),
         attributes={"timeline": "first 6 months"}),

    dict(id="dsgvo_c3",
         entity="German B2B Software Market",
         relationship_type="MANDATES_COMPLIANCE_WITH",
         target_entity="DSGVO/GDPR",
         confidence=0.95, superseded_by=None,
         chat_source="chat_3", created_at=_ts(2024, 12, 5, 11),
         attributes={"type": "regulatory_requirement", "note": "re-confirmed"}),

    dict(id="pricing_c3",
         entity="TechFlow SaaS Market Entry",
         relationship_type="RECOMMENDS_PRICING_MODEL",
         target_entity="Annual contract, EUR-denominated, localised invoicing",
         confidence=0.75, superseded_by=None,
         chat_source="chat_3", created_at=_ts(2024, 12, 5, 12),
         attributes={}),
]

# Resolve string labels → None (superseded_by already set as string label
# matching the id field; filter_superseded checks for truthiness only, so
# leaving as string correctly marks an edge as superseded)

EDGES = _RAW_EDGES  # ids and superseded_by are already consistent


# ── test query ─────────────────────────────────────────────────────────────────
TEST_QUERY = (
    "What is the current recommended market entry strategy for Germany, "
    "including market size, legal entity setup, and best hub city?"
)

# ── system prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior strategy consultant.
    The user will provide a context block drawn from a project memory system,
    followed by a question.
    Answer concisely (5-10 bullet points) using ONLY the information in the context block.
    If data was updated or superseded, reflect only the most current information.
    Explicitly call out any contradictions or stale data you notice in the context.
    Do not invent information not present in the context.
""").strip()

# ── synthetic fallback answers (shown when no API key) ────────────────────────
# These are realistic illustrations of what the LLM would produce.
SYNTHETIC_GRAPH_ANSWER = textwrap.dedent("""
    Based on the graph-assembled context (superseded edges excluded):

    • MARKET SIZE (current): EUR 20B annual revenue (Gartner 2025 forecast, confidence 0.90).
      — Two earlier estimates (EUR 15B, EUR 18B) were superseded and are excluded.

    • LEGAL ENTITY: GmbH is the recommended structure.
      — Minimum share capital: EUR 25,000 (corroborated across two chat sessions, confidence 0.95).
      — Must register with the German Trade Register (Handelsregister).

    • HUB CITY: Munich (confirmed decision, confidence 0.90).
      — Earlier Berlin assumption was explicitly closed: Munich has higher enterprise
        customer concentration (automotive, industrial, fintech sectors).

    • COMPLIANCE: DSGVO/GDPR compliance is mandatory (confidence 0.95, re-confirmed).

    • GO-TO-MARKET:
      — Sales cycle: 6–12 months for enterprise deals.
      — Initial team: 3–5 local hires (sales + customer success) in first 6 months.
      — Engage a local Reseller / VAR partner early.

    • PRICING: Annual contracts, EUR-denominated, with localised invoicing.

    No contradictions detected — all active edges are consistent.
""").strip()

SYNTHETIC_FLAT_ANSWER = textwrap.dedent("""
    Based on the flat adjacency-list context (all edges, including superseded):

    • MARKET SIZE: The context contains three conflicting estimates:
      — EUR 15B (chat_1, low confidence 0.65)
      — EUR 18B (chat_2, confidence 0.85)
      — EUR 20B (chat_3, confidence 0.90)
      ⚠ CONTRADICTION: Cannot determine which is current without supersession metadata.
        Recommending EUR 20B as the most recent, but uncertainty remains.

    • LEGAL ENTITY: GmbH with EUR 25,000 minimum capital (appears twice, consistent).

    • HUB CITY: ⚠ CONFLICT: The context lists both Berlin (open assumption, chat_1)
      and Munich (confirmed decision, chat_3) without clear resolution.
      Cannot determine the final recommendation with certainty.

    • COMPLIANCE: DSGVO/GDPR mandatory (mentioned in chat_1 and chat_3).

    • GO-TO-MARKET:
      — Sales cycle 6–12 months.
      — Initial team 3–5 local hires.
      — Consider local VAR partnership.

    ⚠ KEY ISSUES: Flat context presents superseded market-size estimates as equally
    valid alongside the current figure, and the Berlin/Munich conflict cannot be
    resolved from this representation alone. Answers here may be misleading.
""").strip()


# ── LLM call ──────────────────────────────────────────────────────────────────
def ask_claude(context_block: str, query: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    user_message = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{query}"
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text


# ── pretty print ──────────────────────────────────────────────────────────────
def banner(title: str, width: int = 70) -> str:
    line = "═" * width
    return f"\n{line}\n  {title}\n{line}"


def print_scorecard(graph_ans: str, flat_ans: str):
    print(banner("AUTOMATED SCORECARD (heuristic keyword checks)"))
    checks = [
        ("Market size is EUR 20B",
         "20B" in graph_ans or "20 B" in graph_ans,
         "20B" in flat_ans or "20 B" in flat_ans),
        ("No stale EUR 15B in answer",
         "15B" not in graph_ans,
         "15B" not in flat_ans),
        ("Hub city is Munich",
         "munich" in graph_ans.lower(),
         "munich" in flat_ans.lower()),
        ("No unresolved Berlin conflict",
         not ("berlin" in graph_ans.lower() and "conflict" in graph_ans.lower()),
         not ("berlin" in flat_ans.lower() and "conflict" in flat_ans.lower())),
        ("EUR 25,000 capital mentioned",
         "25,000" in graph_ans or "25000" in graph_ans,
         "25,000" in flat_ans or "25000" in flat_ans),
        ("DSGVO/GDPR mentioned",
         "dsgvo" in graph_ans.lower() or "gdpr" in graph_ans.lower(),
         "dsgvo" in flat_ans.lower() or "gdpr" in flat_ans.lower()),
    ]

    header = f"  {'Criterion':<42} {'Graph':^8} {'Flat':^8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    graph_score = flat_score = 0
    for name, g_pass, f_pass in checks:
        g_mark = "PASS" if g_pass else "FAIL"
        f_mark = "PASS" if f_pass else "FAIL"
        graph_score += int(g_pass)
        flat_score  += int(f_pass)
        print(f"  {name:<42} {g_mark:^8} {f_mark:^8}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<42} {graph_score}/{len(checks):^6}  {flat_score}/{len(checks):^6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-api", action="store_true",
                        help="Skip API calls; show synthetic illustrative answers")
    args = parser.parse_args()

    # ── Build context blocks ──────────────────────────────────────────────────
    g_ctx = graph_context_from_data(TEST_QUERY, EDGES, top_k=15, hops=2)
    f_ctx = flat_context_from_data(EDGES)

    print(banner("GRAPH CONTEXT BLOCK  (superseded edges excluded, weighted by confidence × recency)"))
    print(g_ctx)

    print(banner("FLAT CONTEXT BLOCK  (all edges, including superseded — baseline)"))
    print(f_ctx)

    # ── LLM calls ─────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_api = bool(api_key) and not args.no_api

    if use_api:
        print(banner("CALLING CLAUDE — GRAPH CONTEXT"))
        graph_answer = ask_claude(g_ctx, TEST_QUERY)
        print(banner("CALLING CLAUDE — FLAT CONTEXT"))
        flat_answer = ask_claude(f_ctx, TEST_QUERY)
        source_label = "Claude API response"
    else:
        if not args.no_api:
            print("\n[No ANTHROPIC_API_KEY found — using illustrative synthetic answers]\n")
        graph_answer = SYNTHETIC_GRAPH_ANSWER
        flat_answer  = SYNTHETIC_FLAT_ANSWER
        source_label = "Illustrative synthetic answer (set ANTHROPIC_API_KEY for real responses)"

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print(banner("COMPARISON RESULTS"))
    print(f"\nQUERY : {TEST_QUERY}")
    print(f"SOURCE: {source_label}\n")

    print("── GRAPH ANSWER ─────────────────────────────────────────────────────")
    print(graph_answer)
    print("\n── FLAT ANSWER ──────────────────────────────────────────────────────")
    print(flat_answer)

    # ── Automated scorecard ───────────────────────────────────────────────────
    print_scorecard(graph_answer, flat_answer)

    print(banner("WHAT THIS DEMONSTRATES"))
    print(textwrap.dedent("""
      1. SUPERSESSION CHAIN  (test case B)
         The market-size estimate was revised twice: EUR 15B → EUR 18B → EUR 20B.
         • Graph context: surfaces ONLY EUR 20B (superseded edges filtered).
         • Flat context:  presents all three values with no clear winner — the LLM
           must guess or hedges, introducing uncertainty.

      2. CORROBORATED FACT  (test case A)
         GmbH minimum capital (EUR 25,000) appears in chat_1 AND chat_2.
         The graph aggregates these into a single high-confidence fact.
         The flat list duplicates it — harmless here, but noisy at scale.

      3. OPEN→CLOSED ASSUMPTION  (test case C)
         Berlin hub was an open assumption in chat_1.
         Munich was confirmed in chat_3, closing the Berlin assumption.
         • Graph context: Berlin edge is marked superseded_by Munich edge → excluded.
         • Flat context:  both cities appear without resolution — LLM surfaces a
           "conflict" that the consulting team already resolved.
    """).strip())


if __name__ == "__main__":
    main()
