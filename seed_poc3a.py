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
  Pass 2 — Pass all inserted nodes to Claude in chronological order. Claude
            identifies supersession pairs semantically (same underlying fact,
            later node updates or replaces the earlier one) and returns UUIDs.
            Apply each pair by setting older_node.superseded_by = newer_uuid.
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


# ── Supersession prompt ───────────────────────────────────────────────────────
SUPERSESSION_PROMPT = """\
Below is a list of graph nodes extracted from consulting chats in chronological order.
Identify pairs where a later node supersedes an earlier node — meaning they represent
the same underlying fact but the later one updates or replaces the earlier one.

For each supersession pair output JSON:
{{
  "superseded_id": "UUID of the older node",
  "superseding_id": "UUID of the newer node",
  "reason": "brief explanation"
}}

Output a JSON array only. No explanation. No markdown.

NODES:
{nodes}"""


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

SUPERSEDE_SQL = """
    UPDATE knowledge_graph
    SET    superseded_by = %(newer_id)s
    WHERE  id            = %(older_id)s;
"""


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences if the model wraps its JSON response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def find_supersession_pairs(
    client: anthropic.Anthropic,
    inserted: list[tuple[str, dict, datetime]],
) -> list[dict]:
    """
    Pass all inserted nodes to Claude in chronological order.
    Returns a list of {superseded_id, superseding_id, reason} dicts.
    Returns [] if there are fewer than 2 nodes or the model finds no pairs.
    """
    if len(inserted) < 2:
        return []

    # Build a compact node list for the prompt, sorted by created_at
    node_lines = []
    for uid, node, created_at in sorted(inserted, key=lambda t: t[2]):
        node_lines.append(
            f"id={uid} | chat={node.get('chat_source','')} "
            f"| date={created_at.strftime('%Y-%m-%d')} "
            f"| entity={node.get('entity','')} "
            f"| relationship_type={node.get('relationship_type','')} "
            f"| target_entity={node.get('target_entity','')}"
        )

    prompt = SUPERSESSION_PROMPT.format(nodes="\n".join(node_lines))

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _strip_fences(response.content[0].text)

    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [WARN] Supersession JSON parse error: {exc}", file=sys.stderr)
        print(f"  Raw response:\n{raw[:500]}", file=sys.stderr)
        return []

    return pairs if isinstance(pairs, list) else []


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
    print(f"PASS 2 — Linking supersession chains (LLM-identified)")
    print(f"{'─'*60}")

    # Build a UUID lookup for validation
    inserted_ids = {uid for uid, _, _ in inserted}

    print(f"\n  Asking Claude to identify supersession pairs across {len(inserted)} node(s) …")
    pairs = find_supersession_pairs(client, inserted)
    print(f"  Claude identified {len(pairs)} pair(s)\n")

    supersession_count = 0
    with conn:
        with conn.cursor() as cur:
            for pair in pairs:
                older_uuid    = str(pair.get("superseded_id", "")).strip()
                newer_uuid    = str(pair.get("superseding_id", "")).strip()
                reason        = str(pair.get("reason", "")).strip()

                # Validate both UUIDs were actually inserted in this run
                if older_uuid not in inserted_ids:
                    print(f"  [WARN] superseded_id {older_uuid[:8]}… not in inserted set — skipping")
                    continue
                if newer_uuid not in inserted_ids:
                    print(f"  [WARN] superseding_id {newer_uuid[:8]}… not in inserted set — skipping")
                    continue
                if older_uuid == newer_uuid:
                    print(f"  [WARN] superseded_id == superseding_id ({older_uuid[:8]}…) — skipping")
                    continue

                cur.execute(SUPERSEDE_SQL, {
                    "newer_id": newer_uuid,
                    "older_id": older_uuid,
                })
                supersession_count += 1
                print(f"  Superseded {older_uuid[:8]}… → {newer_uuid[:8]}…  [{reason[:80]}]")

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
