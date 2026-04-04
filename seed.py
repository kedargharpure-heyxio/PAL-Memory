"""
seed.py — Populate knowledge_graph with synthetic Germany market-entry data.

Scenario: "TechFlow SaaS" evaluating entry into the German B2B software market.
Three consulting chat sessions spanning Oct–Dec 2024.

Embedded test cases:
  A) Corroborated fact   — GmbH minimum capital requirement (appears in chat_1 & chat_2, high confidence)
  B) Contradicted fact   — Market size estimate superseded twice (€15B → €18B → €20B)
  C) Open→Closed assumption — Berlin hub assumption raised in chat_1, closed in chat_3 by Munich

Usage:
    python seed.py
    # or pass --dry-run to print SQL without executing
"""

import argparse
import sys
from datetime import datetime, timezone
from typing import Optional
import psycopg2
import psycopg2.extras

# ── connection ──────────────────────────────────────────────────────────────
DB_CONFIG = dict(
    host="db.hezkypxgfbcwiqprpoyp.supabase.co",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="P4l4TheWin@2",
)

# ── helpers ─────────────────────────────────────────────────────────────────
def ts(year, month, day, hour=12):
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


def build_rows():
    """
    Returns (rows, supersession_pairs) where supersession_pairs is a list of
    (older_label, newer_label) tuples that must be resolved to UUIDs after insert.
    """
    rows = []

    # ── CHAT 1  (2024-10-15) — Initial market research  ─────────────────────
    # A — Corroborated fact (first mention)
    rows.append(dict(
        label="cap_req_c1",
        entity="GmbH Formation",
        relationship_type="REQUIRES_MINIMUM_CAPITAL",
        target_entity="EUR 25,000",
        confidence=0.90,
        chat_source="chat_1",
        created_at=ts(2024, 10, 15),
        attributes={"note": "Confirmed via HGB §5a", "type": "legal_requirement"},
    ))

    # B — First (lowest-confidence) market size estimate
    rows.append(dict(
        label="mkt_size_c1",
        entity="German B2B Software Market",
        relationship_type="HAS_ESTIMATED_ANNUAL_REVENUE",
        target_entity="EUR 15B",
        confidence=0.65,
        chat_source="chat_1",
        created_at=ts(2024, 10, 15, 9),
        attributes={"source": "analyst estimate", "year": 2024},
    ))

    # C — Open assumption: Berlin is the best hub
    rows.append(dict(
        label="berlin_hub_c1",
        entity="TechFlow SaaS Market Entry",
        relationship_type="ASSUMES_PRIMARY_HUB",
        target_entity="Berlin",
        confidence=0.60,
        chat_source="chat_1",
        created_at=ts(2024, 10, 15, 10),
        attributes={"status": "open_assumption", "rationale": "startup ecosystem density"},
    ))

    # Supporting edges — chat 1
    rows.append(dict(
        label="dsgvo_c1",
        entity="German B2B Software Market",
        relationship_type="MANDATES_COMPLIANCE_WITH",
        target_entity="DSGVO/GDPR",
        confidence=0.95,
        chat_source="chat_1",
        created_at=ts(2024, 10, 15, 11),
        attributes={"type": "regulatory_requirement"},
    ))
    rows.append(dict(
        label="vat_c1",
        entity="GmbH Formation",
        relationship_type="REQUIRES_REGISTRATION_WITH",
        target_entity="German Trade Register (Handelsregister)",
        confidence=0.95,
        chat_source="chat_1",
        created_at=ts(2024, 10, 15, 11),
        attributes={"type": "legal_process"},
    ))

    # ── CHAT 2  (2024-11-02) — Deeper competitive analysis ───────────────────
    # A — Corroborated fact (second mention, higher confidence)
    rows.append(dict(
        label="cap_req_c2",
        entity="GmbH Formation",
        relationship_type="REQUIRES_MINIMUM_CAPITAL",
        target_entity="EUR 25,000",
        confidence=0.95,
        chat_source="chat_2",
        created_at=ts(2024, 11, 2),
        attributes={"note": "Cross-verified with Bundesjustizamt", "type": "legal_requirement"},
    ))

    # B — Second market size estimate (supersedes chat_1's)
    rows.append(dict(
        label="mkt_size_c2",
        entity="German B2B Software Market",
        relationship_type="HAS_ESTIMATED_ANNUAL_REVENUE",
        target_entity="EUR 18B",
        confidence=0.85,
        chat_source="chat_2",
        created_at=ts(2024, 11, 2, 9),
        attributes={"source": "IDC Germany 2024 report", "year": 2024},
    ))

    # Munich evidence emerges
    rows.append(dict(
        label="munich_evidence_c2",
        entity="Munich",
        relationship_type="HAS_HIGHER_ENTERPRISE_CUSTOMER_CONCENTRATION_THAN",
        target_entity="Berlin",
        confidence=0.75,
        chat_source="chat_2",
        created_at=ts(2024, 11, 2, 10),
        attributes={"sectors": ["automotive", "industrial", "fintech"]},
    ))
    rows.append(dict(
        label="sales_cycle_c2",
        entity="German B2B Enterprise Sales",
        relationship_type="HAS_TYPICAL_CYCLE_LENGTH",
        target_entity="6–12 months",
        confidence=0.80,
        chat_source="chat_2",
        created_at=ts(2024, 11, 2, 11),
        attributes={"type": "operational_assumption"},
    ))
    rows.append(dict(
        label="local_partner_c2",
        entity="TechFlow SaaS Market Entry",
        relationship_type="BENEFITS_FROM",
        target_entity="Local Reseller / VAR Partnership",
        confidence=0.78,
        chat_source="chat_2",
        created_at=ts(2024, 11, 2, 12),
        attributes={"priority": "high"},
    ))

    # ── CHAT 3  (2024-12-05) — Strategy finalization ─────────────────────────
    # B — Third (canonical) market size estimate (supersedes chat_2's)
    rows.append(dict(
        label="mkt_size_c3",
        entity="German B2B Software Market",
        relationship_type="HAS_ESTIMATED_ANNUAL_REVENUE",
        target_entity="EUR 20B",
        confidence=0.90,
        chat_source="chat_3",
        created_at=ts(2024, 12, 5, 9),
        attributes={"source": "Gartner Germany IT Spend 2025 forecast", "year": 2025},
    ))

    # C — Closed assumption: Munich confirmed, Berlin assumption closed
    rows.append(dict(
        label="munich_confirmed_c3",
        entity="TechFlow SaaS Market Entry",
        relationship_type="SELECTS_PRIMARY_HUB",
        target_entity="Munich",
        confidence=0.90,
        chat_source="chat_3",
        created_at=ts(2024, 12, 5, 10),
        attributes={"status": "closed_decision", "closes_assumption": "berlin_hub_c1"},
    ))

    # Staffing recommendation
    rows.append(dict(
        label="staffing_c3",
        entity="TechFlow SaaS Market Entry",
        relationship_type="RECOMMENDS_INITIAL_HEADCOUNT",
        target_entity="3–5 local hires (sales + customer success)",
        confidence=0.80,
        chat_source="chat_3",
        created_at=ts(2024, 12, 5, 11),
        attributes={"timeline": "first 6 months"},
    ))
    rows.append(dict(
        label="dsgvo_c3",
        entity="German B2B Software Market",
        relationship_type="MANDATES_COMPLIANCE_WITH",
        target_entity="DSGVO/GDPR",
        confidence=0.95,
        chat_source="chat_3",
        created_at=ts(2024, 12, 5, 11),
        attributes={"type": "regulatory_requirement", "note": "re-confirmed, non-negotiable"},
    ))
    rows.append(dict(
        label="pricing_c3",
        entity="TechFlow SaaS Market Entry",
        relationship_type="RECOMMENDS_PRICING_MODEL",
        target_entity="Annual contract, EUR-denominated, localised invoicing",
        confidence=0.75,
        chat_source="chat_3",
        created_at=ts(2024, 12, 5, 12),
        attributes={},
    ))

    # Supersession pairs (older_label, newer_label)
    supersession_pairs = [
        ("mkt_size_c1", "mkt_size_c2"),   # €15B → €18B
        ("mkt_size_c2", "mkt_size_c3"),   # €18B → €20B
        ("berlin_hub_c1", "munich_confirmed_c3"),  # Berlin assumption → Munich decision
    ]

    return rows, supersession_pairs


# ── insert ───────────────────────────────────────────────────────────────────
def seed(dry_run: bool = False):
    rows, supersession_pairs = build_rows()
    label_to_id: dict[str, str] = {}

    insert_sql = """
        INSERT INTO knowledge_graph
            (entity, relationship_type, target_entity, confidence,
             chat_source, created_at, attributes)
        VALUES
            (%(entity)s, %(relationship_type)s, %(target_entity)s,
             %(confidence)s, %(chat_source)s, %(created_at)s, %(attributes)s)
        RETURNING id;
    """
    supersede_sql = """
        UPDATE knowledge_graph
        SET superseded_by = %(newer_id)s
        WHERE id = %(older_id)s;
    """

    if dry_run:
        print("── DRY RUN ── rows that would be inserted:\n")
        for r in rows:
            print(f"  [{r['label']:20s}] {r['entity']} --{r['relationship_type']}--> "
                  f"{r['target_entity']}  (conf={r['confidence']}, src={r['chat_source']})")
        print(f"\n── Supersession pairs ({len(supersession_pairs)}):")
        for old, new in supersession_pairs:
            print(f"  {old} superseded_by {new}")
        return

    conn = psycopg2.connect(**DB_CONFIG)
    psycopg2.extras.register_uuid()
    try:
        with conn:
            with conn.cursor() as cur:
                # Insert all rows
                for row in rows:
                    label = row.pop("label")
                    row["attributes"] = psycopg2.extras.Json(row["attributes"])
                    cur.execute(insert_sql, row)
                    label_to_id[label] = str(cur.fetchone()[0])
                    print(f"  Inserted [{label:20s}] id={label_to_id[label]}")

                # Apply supersessions
                for old_label, new_label in supersession_pairs:
                    cur.execute(supersede_sql, {
                        "older_id": label_to_id[old_label],
                        "newer_id": label_to_id[new_label],
                    })
                    print(f"  Superseded {old_label} → {new_label}")

        print(f"\nSeeded {len(rows)} edges, {len(supersession_pairs)} supersession links.")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed knowledge_graph with Germany market-entry data")
    parser.add_argument("--dry-run", action="store_true", help="Print without executing")
    args = parser.parse_args()
    seed(dry_run=args.dry_run)
