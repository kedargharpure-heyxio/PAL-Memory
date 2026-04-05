"""
seed_poc3a.py — Extract entities/relationships from raw chat transcripts and
insert them into the Supabase knowledge_graph table.

Usage:
    python seed_poc3a.py chats.json

Input JSON format:
    [
      {
        "identifier": "chat_1",
        "days_ago": 42,
        "transcript": "full text of chat here"
      },
      ...
    ]

Behaviour:
  Pass 1 — For each chat, call the Anthropic API to extract typed graph edges.
            Insert every extracted edge into knowledge_graph.
  Pass 2 — For each edge that carries a supersedes_description, locate the most
            recent prior edge with the same entity + relationship_type and set
            its superseded_by field to the new edge's UUID.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import anthropic
import psycopg2
import psycopg2.extras

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

# ── Extraction prompt ─────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """\
Extract all entities and relationships from the following consulting chat transcript.
For each relationship output a JSON object with these fields:
- entity (text)
- relationship_type (text, uppercase with underscores)
- target_entity (text)
- confidence (float 0-1, based on how firmly the fact is stated)
- chat_source (use the chat identifier provided)
- created_at (use the date provided)
- supersedes_description (text — if this fact explicitly replaces or closes a prior fact, \
describe what it replaces. Otherwise null.)

Output a JSON array only. No explanation. No markdown.

Chat identifier: {chat_id}
Date: {created_at}

Transcript:
{transcript}"""


# ── Anthropic extraction ──────────────────────────────────────────────────────
def extract_relationships(
    client: anthropic.Anthropic,
    transcript: str,
    chat_id: str,
    created_at: datetime,
) -> list[dict]:
    """Call the API and return a list of relationship dicts."""
    prompt = EXTRACTION_PROMPT.format(
        chat_id=chat_id,
        created_at=created_at.strftime("%Y-%m-%d"),
        transcript=transcript,
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences if the model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        nodes = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [WARN] JSON parse error for {chat_id}: {exc}", file=sys.stderr)
        print(f"  Raw response:\n{raw[:500]}", file=sys.stderr)
        nodes = []

    return nodes if isinstance(nodes, list) else []


# ── DB helpers ────────────────────────────────────────────────────────────────
INSERT_SQL = """
    INSERT INTO knowledge_graph
        (entity, relationship_type, target_entity, confidence,
         chat_source, created_at, attributes)
    VALUES
        (%(entity)s, %(relationship_type)s, %(target_entity)s,
         %(confidence)s, %(chat_source)s, %(created_at)s, %(attributes)s)
    RETURNING id;
"""

FIND_PRIOR_SQL = """
    SELECT id
    FROM   knowledge_graph
    WHERE  entity            = %(entity)s
      AND  relationship_type = %(relationship_type)s
      AND  created_at        < %(created_at)s
      AND  superseded_by IS NULL
    ORDER BY created_at DESC
    LIMIT 1;
"""

SUPERSEDE_SQL = """
    UPDATE knowledge_graph
    SET    superseded_by = %(newer_id)s
    WHERE  id            = %(older_id)s;
"""


def insert_node(cur: psycopg2.extensions.cursor, node: dict, created_at: datetime) -> str:
    """Insert one node and return its UUID."""
    # Coerce confidence to float, clamp to [0,1]
    try:
        conf = float(node.get("confidence", 0.8))
    except (TypeError, ValueError):
        conf = 0.8
    conf = max(0.0, min(1.0, conf))

    cur.execute(INSERT_SQL, {
        "entity":            str(node.get("entity", "")).strip(),
        "relationship_type": str(node.get("relationship_type", "RELATED_TO")).strip().upper(),
        "target_entity":     str(node.get("target_entity", "")).strip(),
        "confidence":        conf,
        "chat_source":       str(node.get("chat_source", "")).strip(),
        "created_at":        created_at,
        "attributes":        psycopg2.extras.Json({}),
    })
    return str(cur.fetchone()[0])


# ── Main ──────────────────────────────────────────────────────────────────────
def seed(json_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as fh:
        chats = json.load(fh)

    client = anthropic.Anthropic()
    conn = psycopg2.connect(**DB_CONFIG)
    psycopg2.extras.register_uuid()

    # inserted_nodes: list of (uuid, node_dict, created_at) for the supersession pass
    inserted: list[tuple[str, dict, datetime]] = []

    today = datetime.now(timezone.utc)

    print(f"\n{'─'*60}")
    print(f"PASS 1 — Extracting and inserting nodes from {len(chats)} chat(s)")
    print(f"{'─'*60}")

    for chat in chats:
        chat_id   = chat["identifier"]
        days_ago  = int(chat["days_ago"])
        transcript = chat["transcript"]
        created_at = today - timedelta(days=days_ago)

        print(f"\n[{chat_id}] Extracting relationships (date={created_at.date()}) …")
        nodes = extract_relationships(client, transcript, chat_id, created_at)
        print(f"  Extracted {len(nodes)} node(s)")

        with conn:
            with conn.cursor() as cur:
                for node in nodes:
                    # Override chat_source and created_at with authoritative values
                    node["chat_source"] = chat_id
                    uid = insert_node(cur, node, created_at)
                    inserted.append((uid, node, created_at))
                    sup = node.get("supersedes_description")
                    sup_label = f" [supersedes: {sup[:60]}…]" if sup and len(sup) > 60 \
                               else (f" [supersedes: {sup}]" if sup else "")
                    print(f"  Inserted {uid[:8]}… "
                          f"{node.get('entity','')} "
                          f"--[{node.get('relationship_type','')}]--> "
                          f"{node.get('target_entity','')}{sup_label}")

    print(f"\n{'─'*60}")
    print(f"PASS 2 — Linking supersession chains")
    print(f"{'─'*60}")

    supersession_count = 0
    with conn:
        with conn.cursor() as cur:
            for new_uuid, node, created_at in inserted:
                sup_desc = node.get("supersedes_description")
                if not sup_desc:
                    continue

                # Find the most recent prior edge with same entity + relationship_type
                cur.execute(FIND_PRIOR_SQL, {
                    "entity":            str(node.get("entity", "")).strip(),
                    "relationship_type": str(node.get("relationship_type", "")).strip().upper(),
                    "created_at":        created_at,
                })
                row = cur.fetchone()
                if row:
                    older_uuid = str(row[0])
                    cur.execute(SUPERSEDE_SQL, {
                        "newer_id": new_uuid,
                        "older_id": older_uuid,
                    })
                    supersession_count += 1
                    print(f"  Superseded {older_uuid[:8]}… "
                          f"→ {new_uuid[:8]}… "
                          f"({node.get('entity','')} / {node.get('relationship_type','')})")
                else:
                    print(f"  [WARN] No prior node found to supersede for: "
                          f"{node.get('entity','')} / {node.get('relationship_type','')} "
                          f"— description: {sup_desc[:80]}")

    conn.close()
    total = len(inserted)
    print(f"\n{'─'*60}")
    print(f"Done. {total} node(s) inserted, {supersession_count} supersession link(s) applied.")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract graph nodes from chat transcripts and seed knowledge_graph"
    )
    parser.add_argument("json_file", help="Path to JSON file containing chat transcripts")
    args = parser.parse_args()
    seed(args.json_file)
