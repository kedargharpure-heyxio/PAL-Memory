-- ============================================================
-- GraphRAG Memory Schema for PAL-Memory
-- Apply via: psql $DATABASE_URL -f schema.sql
--            or paste into Supabase SQL Editor
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- Core knowledge graph edge table
-- Each row is a directed, typed edge: entity -> relationship_type -> target_entity
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge_graph (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    entity          TEXT        NOT NULL,           -- source entity label
    relationship_type TEXT      NOT NULL,           -- semantic edge type (VERB-form, e.g. HAS_SIZE)
    target_entity   TEXT        NOT NULL,           -- target entity label / value
    confidence      FLOAT       NOT NULL DEFAULT 1.0
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    superseded_by   UUID        REFERENCES knowledge_graph(id) ON DELETE SET NULL,
    chat_source     TEXT        NOT NULL,           -- originating chat, e.g. "chat_1"
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    attributes      JSONB       NOT NULL DEFAULT '{}'  -- arbitrary extra metadata
);

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_kg_entity       ON knowledge_graph(entity);
CREATE INDEX IF NOT EXISTS idx_kg_target       ON knowledge_graph(target_entity);
CREATE INDEX IF NOT EXISTS idx_kg_superseded   ON knowledge_graph(superseded_by);
CREATE INDEX IF NOT EXISTS idx_kg_chat         ON knowledge_graph(chat_source);
CREATE INDEX IF NOT EXISTS idx_kg_confidence   ON knowledge_graph(confidence);
CREATE INDEX IF NOT EXISTS idx_kg_created      ON knowledge_graph(created_at DESC);
-- Partial index for active (non-superseded) edges — used by hot retrieval paths
CREATE INDEX IF NOT EXISTS idx_kg_active
    ON knowledge_graph(entity, confidence DESC)
    WHERE superseded_by IS NULL;

-- ============================================================
-- Helper view: active edges only (not superseded)
-- ============================================================
CREATE OR REPLACE VIEW knowledge_graph_active AS
SELECT *
FROM   knowledge_graph
WHERE  superseded_by IS NULL;

-- ============================================================
-- Helper view: supersession chains  (shows lineage)
-- ============================================================
CREATE OR REPLACE VIEW knowledge_graph_supersession_chains AS
WITH RECURSIVE chain AS (
    -- anchor: edges that have been superseded (start of chain)
    SELECT
        id,
        entity,
        relationship_type,
        target_entity,
        confidence,
        superseded_by,
        chat_source,
        created_at,
        id        AS chain_root,
        1         AS depth
    FROM knowledge_graph
    WHERE superseded_by IS NOT NULL

    UNION ALL

    -- recursive: follow superseded_by upward
    SELECT
        kg.id,
        kg.entity,
        kg.relationship_type,
        kg.target_entity,
        kg.confidence,
        kg.superseded_by,
        kg.chat_source,
        kg.created_at,
        chain.chain_root,
        chain.depth + 1
    FROM knowledge_graph kg
    JOIN chain ON chain.superseded_by = kg.id
)
SELECT * FROM chain ORDER BY chain_root, depth;

-- ============================================================
-- RPC: graph_context(query_entities TEXT[], hops INT)
-- Returns weighted context for the given entity set.
-- Called from the retrieval layer via PostgREST / supabase-py.
-- ============================================================
CREATE OR REPLACE FUNCTION graph_context(
    query_entities  TEXT[],
    hops            INT DEFAULT 2
)
RETURNS TABLE (
    id              UUID,
    entity          TEXT,
    relationship_type TEXT,
    target_entity   TEXT,
    confidence      FLOAT,
    chat_source     TEXT,
    created_at      TIMESTAMPTZ,
    hop             INT,
    recency_score   FLOAT,
    composite_score FLOAT
)
LANGUAGE SQL
STABLE
AS $$
WITH RECURSIVE traversal AS (
    -- Seed: edges whose entity OR target matches query
    SELECT
        kg.id,
        kg.entity,
        kg.relationship_type,
        kg.target_entity,
        kg.confidence,
        kg.chat_source,
        kg.created_at,
        0 AS hop
    FROM knowledge_graph kg
    WHERE kg.superseded_by IS NULL
      AND (
          kg.entity       ILIKE ANY (SELECT '%' || unnest || '%' FROM unnest(query_entities))
       OR kg.target_entity ILIKE ANY (SELECT '%' || unnest || '%' FROM unnest(query_entities))
      )

    UNION

    -- Expand: follow neighbours up to `hops` steps
    SELECT
        kg.id,
        kg.entity,
        kg.relationship_type,
        kg.target_entity,
        kg.confidence,
        kg.chat_source,
        kg.created_at,
        t.hop + 1
    FROM knowledge_graph kg
    JOIN traversal t
      ON (kg.entity = t.target_entity OR kg.target_entity = t.entity)
    WHERE kg.superseded_by IS NULL
      AND t.hop < hops
)
SELECT
    t.id,
    t.entity,
    t.relationship_type,
    t.target_entity,
    t.confidence,
    t.chat_source,
    t.created_at,
    MIN(t.hop)                                                          AS hop,
    -- recency: exponential decay over days, half-life ~90 days
    EXP(-0.0077 * EXTRACT(EPOCH FROM (now() - t.created_at)) / 86400)  AS recency_score,
    t.confidence *
      EXP(-0.0077 * EXTRACT(EPOCH FROM (now() - t.created_at)) / 86400) AS composite_score
FROM traversal t
GROUP BY t.id, t.entity, t.relationship_type, t.target_entity,
         t.confidence, t.chat_source, t.created_at
ORDER BY composite_score DESC;
$$;
