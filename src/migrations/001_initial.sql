-- 001_initial.sql (PostgreSQL version)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories
(
    id             TEXT PRIMARY KEY,
    content        TEXT,
    primary_sector TEXT,
    sectors        TEXT,
    tags           TEXT,
    compressed_vec BYTEA,
    meta           TEXT,
    user_id        TEXT,
    segment        INTEGER DEFAULT 0,
    created_at     TIMESTAMP,
    updated_at     TIMESTAMP,
    last_seen_at   TIMESTAMP,
    salience       DOUBLE PRECISION,
    decay_lambda   DOUBLE PRECISION,
    version        INTEGER,
    mean_dim       INTEGER,
    mean_vec       BYTEA,
    feedback_score DOUBLE PRECISION DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vectors
(
    id      TEXT,
    v       vector(1536),
    dim     INTEGER,
    sector  TEXT,
    user_id TEXT,
    PRIMARY KEY (id, sector),
    CONSTRAINT fk_vectors_id_memories_id
        FOREIGN KEY (id) REFERENCES memories (id)
);

CREATE TABLE IF NOT EXISTS users
(
    user_id          TEXT PRIMARY KEY,
    summary          TEXT,
    reflection_count INTEGER,
    created_at       TIMESTAMP,
    updated_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stats
(
    id      SERIAL PRIMARY KEY,
    ts      BIGINT,
    metrics TEXT
);

CREATE TABLE IF NOT EXISTS embed_logs
(
    id     TEXT,
    model  TEXT,
    status TEXT,
    ts     BIGINT,
    err    TEXT,
    CONSTRAINT fk_embed_logs_id_memories
        FOREIGN KEY (id) REFERENCES memories (id)
);

CREATE TABLE IF NOT EXISTS waypoints
(
    src_id     TEXT,
    dst_id     TEXT,
    user_id    TEXT,
    weight     DOUBLE PRECISION,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (src_id, dst_id),
    CONSTRAINT fk_waypoints_src_id_memories_id
        FOREIGN KEY (src_id) REFERENCES memories (id),
    CONSTRAINT fk_waypoints_dst_id_memories_id
        FOREIGN KEY (dst_id) REFERENCES memories (id)
);

CREATE TABLE IF NOT EXISTS temporal_facts
(
    id         TEXT PRIMARY KEY,
    subject    TEXT   NOT NULL,
    predicate  TEXT   NOT NULL,
    obj        TEXT   NOT NULL,
    valid_from BIGINT NOT NULL,
    valid_to   BIGINT,
    confidence DOUBLE PRECISION,
    metadata   TEXT
);

CREATE TABLE IF NOT EXISTS temporal_edges
(
    source_id  TEXT             NOT NULL,
    target_id  TEXT             NOT NULL,
    relation   TEXT             NOT NULL,
    valid_from BIGINT           NOT NULL,
    valid_to   BIGINT,
    weight     DOUBLE PRECISION NOT NULL,
    metadata   TEXT,
    CONSTRAINT fk_temporal_edges_source_id_temporal_facts_id
        FOREIGN KEY (source_id) REFERENCES temporal_facts (id),
    CONSTRAINT fk_temporal_edges_target_id_temporal_facts_id
        FOREIGN KEY (target_id) REFERENCES temporal_facts (id)
);

CREATE TABLE IF NOT EXISTS segment
(
    id              SERIAL PRIMARY KEY,
    current_segment BIGINT NOT NULL DEFAULT 0,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO segment (current_segment)
VALUES ((SELECT COALESCE(MAX(segment), 0) FROM memories))
ON CONFLICT DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_memories_sector ON memories (primary_sector);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories (last_seen_at);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_vectors_user ON vectors (user_id);
CREATE INDEX IF NOT EXISTS idx_waypoints_src ON waypoints (src_id);
CREATE INDEX IF NOT EXISTS idx_waypoints_dst ON waypoints (dst_id);
CREATE INDEX IF NOT EXISTS idx_stats_ts ON stats (ts);
CREATE INDEX IF NOT EXISTS idx_temporal_subject ON temporal_facts (subject);
