-- 001_initial.sql (PostgreSQL version)

CREATE EXTENSION IF NOT EXISTS vector;

-- 记忆表

CREATE TABLE IF NOT EXISTS memories
(
    id             TEXT PRIMARY KEY,
    tenant_id      TEXT,
    project_id     TEXT,
    user_id        TEXT             DEFAULT 'anonymous',
    content        TEXT,
    primary_sector TEXT,
    sectors        TEXT,
    tags           TEXT,
    compressed_vec BYTEA,
    meta           TEXT,
    segment        INTEGER          DEFAULT 0,
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

COMMENT ON TABLE memories IS '用户记忆及派生元数据';
COMMENT ON COLUMN memories.id IS '记忆主标识';
COMMENT ON COLUMN memories.tenant_id IS '租户标识';
COMMENT ON COLUMN memories.project_id IS '项目标识';
COMMENT ON COLUMN memories.user_id IS '用户标识';
COMMENT ON COLUMN memories.content IS '原始记忆内容';
COMMENT ON COLUMN memories.primary_sector IS '主扇区/主题标签';
COMMENT ON COLUMN memories.sectors IS '次级扇区/主题标签';
COMMENT ON COLUMN memories.tags IS '用户或系统标签';
COMMENT ON COLUMN memories.compressed_vec IS '压缩嵌入向量字节';
COMMENT ON COLUMN memories.meta IS '元数据载荷（序列化）';
COMMENT ON COLUMN memories.segment IS '分段编号';
COMMENT ON COLUMN memories.created_at IS '创建时间戳';
COMMENT ON COLUMN memories.updated_at IS '最后更新时间戳';
COMMENT ON COLUMN memories.last_seen_at IS '最后访问时间戳';
COMMENT ON COLUMN memories.salience IS '显著性得分';
COMMENT ON COLUMN memories.decay_lambda IS '衰减率（lambda）';
COMMENT ON COLUMN memories.version IS '版本标记';
COMMENT ON COLUMN memories.mean_dim IS '均值嵌入维度';
COMMENT ON COLUMN memories.mean_vec IS '均值嵌入向量字节';
COMMENT ON COLUMN memories.feedback_score IS '用户反馈得分';

-- 嵌入向量表

CREATE TABLE IF NOT EXISTS vectors
(
    id         TEXT,
    tenant_id  TEXT,
    project_id TEXT,
    user_id    TEXT,
    v          vector(1536),
    dim        INTEGER,
    sector     TEXT,
    PRIMARY KEY (id, sector),
    CONSTRAINT fk_vectors_id_memories_id
        FOREIGN KEY (id) REFERENCES memories (id)
);

COMMENT ON TABLE vectors IS '记忆的嵌入向量';
COMMENT ON COLUMN vectors.id IS '记忆标识';
COMMENT ON COLUMN vectors.tenant_id IS '租户标识';
COMMENT ON COLUMN vectors.project_id IS '项目标识';
COMMENT ON COLUMN vectors.user_id IS '用户标识';
COMMENT ON COLUMN vectors.v IS '嵌入向量';
COMMENT ON COLUMN vectors.dim IS '向量维度';
COMMENT ON COLUMN vectors.sector IS '扇区/主题标签';

-- 用户表

CREATE TABLE IF NOT EXISTS users
(
    id               TEXT PRIMARY KEY,
    tenant_id        TEXT,
    project_id       TEXT,
    user_id          TEXT,
    summary          TEXT,
    reflection_count INTEGER,
    created_at       TIMESTAMP,
    updated_at       TIMESTAMP,
    UNIQUE (tenant_id, project_id, user_id)
);

COMMENT ON TABLE users IS '用户级摘要与计数器';
COMMENT ON COLUMN users.tenant_id IS '租户标识';
COMMENT ON COLUMN users.project_id IS '项目标识';
COMMENT ON COLUMN users.user_id IS '用户标识';
COMMENT ON COLUMN users.summary IS '用户摘要文本';
COMMENT ON COLUMN users.reflection_count IS '反思计数';
COMMENT ON COLUMN users.created_at IS '创建时间戳';
COMMENT ON COLUMN users.updated_at IS '最后更新时间戳';

-- 运行指标表

CREATE TABLE IF NOT EXISTS stats
(
    id      SERIAL PRIMARY KEY,
    ts      BIGINT,
    metrics TEXT
);

COMMENT ON TABLE stats IS '运行指标快照';
COMMENT ON COLUMN stats.id IS '统计记录标识';
COMMENT ON COLUMN stats.ts IS 'Unix 时间戳（毫秒）';
COMMENT ON COLUMN stats.metrics IS '指标载荷（序列化）';

-- 嵌入生成日志表

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

COMMENT ON TABLE embed_logs IS '嵌入生成日志';
COMMENT ON COLUMN embed_logs.id IS '记忆标识';
COMMENT ON COLUMN embed_logs.model IS '嵌入模型名称';
COMMENT ON COLUMN embed_logs.status IS '嵌入任务状态';
COMMENT ON COLUMN embed_logs.ts IS 'Unix 时间戳（毫秒）';
COMMENT ON COLUMN embed_logs.err IS '失败时错误信息';

-- 路标表，表示记忆之间的关系边

CREATE TABLE IF NOT EXISTS waypoints
(
    src_id     TEXT,
    dst_id     TEXT,
    tenant_id  TEXT,
    project_id TEXT,
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

COMMENT ON TABLE waypoints IS '相关记忆之间的边';
COMMENT ON COLUMN waypoints.src_id IS '源记忆标识';
COMMENT ON COLUMN waypoints.dst_id IS '目标记忆标识';
COMMENT ON COLUMN waypoints.tenant_id IS '租户标识';
COMMENT ON COLUMN waypoints.project_id IS '项目标识';
COMMENT ON COLUMN waypoints.user_id IS '用户标识';
COMMENT ON COLUMN waypoints.weight IS '边权重';
COMMENT ON COLUMN waypoints.created_at IS '创建时间戳';
COMMENT ON COLUMN waypoints.updated_at IS '最后更新时间戳';

-- 时间事实表，表示从内容提取的有时间边界的事实，以及它们之间的关系边

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

COMMENT ON TABLE temporal_facts IS '从内容提取的有时间边界的事实';
COMMENT ON COLUMN temporal_facts.id IS '时间事实标识';
COMMENT ON COLUMN temporal_facts.subject IS '事实主体';
COMMENT ON COLUMN temporal_facts.predicate IS '事实谓词';
COMMENT ON COLUMN temporal_facts.obj IS '事实客体';
COMMENT ON COLUMN temporal_facts.valid_from IS '有效期开始（毫秒）';
COMMENT ON COLUMN temporal_facts.valid_to IS '有效期结束（毫秒）';
COMMENT ON COLUMN temporal_facts.confidence IS '事实置信度';
COMMENT ON COLUMN temporal_facts.metadata IS '元数据载荷（序列化）';

-- 时间边表，表示时间事实之间的关系边

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

COMMENT ON TABLE temporal_edges IS '事实之间的时间关系';
COMMENT ON COLUMN temporal_edges.source_id IS '源时间事实标识';
COMMENT ON COLUMN temporal_edges.target_id IS '目标时间事实标识';
COMMENT ON COLUMN temporal_edges.relation IS '边关系标签';
COMMENT ON COLUMN temporal_edges.valid_from IS '有效期开始（毫秒）';
COMMENT ON COLUMN temporal_edges.valid_to IS '有效期结束（毫秒）';
COMMENT ON COLUMN temporal_edges.weight IS '边权重';
COMMENT ON COLUMN temporal_edges.metadata IS '元数据载荷（序列化）';

-- 分段表，用于全局分段跟踪，确保在分布式环境中生成唯一的分段编号

CREATE TABLE IF NOT EXISTS segment
(
    id              SERIAL PRIMARY KEY,
    current_segment BIGINT NOT NULL DEFAULT 0,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE segment IS '全局分段跟踪';
COMMENT ON COLUMN segment.id IS '分段记录标识';
COMMENT ON COLUMN segment.current_segment IS '当前分段值';
COMMENT ON COLUMN segment.created_at IS '创建时间戳';
COMMENT ON COLUMN segment.updated_at IS '最后更新时间戳';

-- 基础数据插入

INSERT INTO segment (current_segment)
VALUES ((SELECT COALESCE(MAX(segment), 0) FROM memories))
ON CONFLICT DO NOTHING;

-- 索引

CREATE INDEX IF NOT EXISTS idx_memories_sector ON memories (primary_sector);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories (last_seen_at);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_vectors_user ON vectors (user_id);
CREATE INDEX IF NOT EXISTS idx_waypoints_src ON waypoints (src_id);
CREATE INDEX IF NOT EXISTS idx_waypoints_dst ON waypoints (dst_id);
CREATE INDEX IF NOT EXISTS idx_stats_ts ON stats (ts);
CREATE INDEX IF NOT EXISTS idx_temporal_subject ON temporal_facts (subject);
