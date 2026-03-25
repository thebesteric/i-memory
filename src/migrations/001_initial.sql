-- 001_initial.sql (PostgreSQL version)

CREATE EXTENSION IF NOT EXISTS vector;

-- 记忆表

CREATE TABLE IF NOT EXISTS memories
(
    id             TEXT PRIMARY KEY,
    user_id        TEXT             DEFAULT 'anonymous',
    qa_role        TEXT,
    qa_pair_id     TEXT,
    content        TEXT,
    summary        TEXT,
    fact_joined    BOOLEAN          DEFAULT FALSE,
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
    feedback_score DOUBLE PRECISION DEFAULT 0,
    CONSTRAINT chk_memories_qa_role
        CHECK (qa_role IS NULL OR qa_role IN ('human', 'assistant'))
);

COMMENT ON TABLE memories IS '用户记忆及派生元数据';
COMMENT ON COLUMN memories.id IS '记忆主标识';
COMMENT ON COLUMN memories.user_id IS '用户标识';
COMMENT ON COLUMN memories.qa_role IS '问答角色（human/assistant）';
COMMENT ON COLUMN memories.qa_pair_id IS '问答对标识';
COMMENT ON COLUMN memories.content IS '原始记忆内容';
COMMENT ON COLUMN memories.summary IS '记忆摘要';
COMMENT ON COLUMN memories.fact_joined IS '是否已参与事实处理';
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
    id      TEXT,
    sector  TEXT,
    user_id TEXT,
    v       VECTOR(1536),
    dim     INTEGER,
    PRIMARY KEY (id, sector),
    CONSTRAINT fk_vectors_id_memories_id
        FOREIGN KEY (id) REFERENCES memories (id)
);

COMMENT ON TABLE vectors IS '记忆的嵌入向量';
COMMENT ON COLUMN vectors.id IS '记忆标识';
COMMENT ON COLUMN vectors.sector IS '扇区/主题标签';
COMMENT ON COLUMN vectors.user_id IS '用户标识';
COMMENT ON COLUMN vectors.v IS '嵌入向量';
COMMENT ON COLUMN vectors.dim IS '向量维度';

-- 用户表

CREATE TABLE IF NOT EXISTS users
(
    id               TEXT PRIMARY KEY,
    tenant_key       TEXT,
    project_key      TEXT,
    user_key         TEXT,
    summary          TEXT,
    reflection_count INTEGER,
    created_at       TIMESTAMP,
    updated_at       TIMESTAMP,
    UNIQUE (tenant_key, project_key, user_key)
);

COMMENT ON TABLE users IS '用户级摘要与计数器';
COMMENT ON COLUMN users.id IS '用户主键';
COMMENT ON COLUMN users.tenant_key IS '租户标识';
COMMENT ON COLUMN users.project_key IS '项目标识';
COMMENT ON COLUMN users.user_key IS '用户标识';
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
COMMENT ON COLUMN waypoints.user_id IS '用户标识';
COMMENT ON COLUMN waypoints.weight IS '边权重';
COMMENT ON COLUMN waypoints.created_at IS '创建时间戳';
COMMENT ON COLUMN waypoints.updated_at IS '最后更新时间戳';

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

-- 主题表
CREATE TABLE IF NOT EXISTS topics
(
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT,
    project_id   TEXT,
    user_id      TEXT,
    name         TEXT NOT NULL,
    summary      TEXT,
    keywords     JSONB     DEFAULT '[]',
    dialogue_ids JSONB     DEFAULT '[]',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE topics IS '主题表';
COMMENT ON COLUMN topics.id IS '主题标识';
COMMENT ON COLUMN topics.tenant_id IS '租户标识';
COMMENT ON COLUMN topics.project_id IS '项目标识';
COMMENT ON COLUMN topics.user_id IS '用户标识';
COMMENT ON COLUMN topics.name IS '主题名称';
COMMENT ON COLUMN topics.summary IS '主题摘要';
COMMENT ON COLUMN topics.keywords IS '主题相关关键词列表';
COMMENT ON COLUMN topics.dialogue_ids IS '相关对话标识列表';
COMMENT ON COLUMN topics.created_at IS '创建时间戳';
COMMENT ON COLUMN topics.updated_at IS '最后更新时间戳';

-- 事实表
CREATE TABLE IF NOT EXISTS facts
(
    id             TEXT PRIMARY KEY,
    user_id        TEXT,
    topic_id       TEXT,
    what           TEXT        NOT NULL,
    when_          TEXT,
    where_         TEXT,
    who            TEXT,
    why            TEXT,
    status         VARCHAR(20)          DEFAULT 'pending',
    fact_kind      VARCHAR(20) NOT NULL DEFAULT 'conversation',
    occurred_start TIMESTAMP,
    occurred_end   TIMESTAMP,
    created_at     TIMESTAMP,
    updated_at     TIMESTAMP,
    processed_at   TIMESTAMP,
    CONSTRAINT fk_facts_topic_id_topics_id
        FOREIGN KEY (topic_id) REFERENCES topics (id)
);

COMMENT ON TABLE facts IS '事实表';
COMMENT ON COLUMN facts.id IS '事实标识';
COMMENT ON COLUMN facts.user_id IS '用户标识';
COMMENT ON COLUMN facts.topic_id IS '主题 ID';
COMMENT ON COLUMN facts.what IS '事实内容';
COMMENT ON COLUMN facts.when_ IS '事实发生的时间描述';
COMMENT ON COLUMN facts.where_ IS '事实发生的地点描述';
COMMENT ON COLUMN facts.who IS '事实相关的主体描述';
COMMENT ON COLUMN facts.why IS '事实发生的原因描述';
COMMENT ON COLUMN facts.status IS '事实状态（如 pending、processed、failed）';
COMMENT ON COLUMN facts.fact_kind IS '事实类型（如 conversation、event 等）';
COMMENT ON COLUMN facts.occurred_start IS '事实发生的开始时间';
COMMENT ON COLUMN facts.occurred_end IS '事实发生的结束时间';
COMMENT ON COLUMN facts.created_at IS '创建时间戳';
COMMENT ON COLUMN facts.updated_at IS '最后更新时间戳';
COMMENT ON COLUMN facts.processed_at IS '事实被图谱化的时间戳';

-- 规范化实体表
CREATE TABLE IF NOT EXISTS canonical_entities
(
    id               TEXT PRIMARY KEY,
    user_id          TEXT,
    name             VARCHAR(500) NOT NULL,
    entity_type      VARCHAR(30)  NOT NULL,
    entity_label     VARCHAR(50),
    vector           VECTOR(1536),
    occurrence_count INT       DEFAULT 1,
    first_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, name, entity_type)
);

COMMENT ON TABLE canonical_entities IS '规范化实体表';
COMMENT ON COLUMN canonical_entities.id IS '实体标识';
COMMENT ON COLUMN canonical_entities.user_id IS '用户标识';
COMMENT ON COLUMN canonical_entities.name IS '实体名称';
COMMENT ON COLUMN canonical_entities.entity_type IS '实体类型';
COMMENT ON COLUMN canonical_entities.vector IS '实体的嵌入向量';
COMMENT ON COLUMN canonical_entities.occurrence_count IS '实体出现次数';
COMMENT ON COLUMN canonical_entities.first_seen_at IS '实体首次出现时间';
COMMENT ON COLUMN canonical_entities.last_seen_at IS '实体最后出现时间';
COMMENT ON COLUMN canonical_entities.created_at IS '创建时间戳';
COMMENT ON COLUMN canonical_entities.updated_at IS '最后更新时间戳';

-- 实体表
CREATE TABLE IF NOT EXISTS entities
(
    id             TEXT PRIMARY KEY,
    user_id        TEXT,
    text           VARCHAR(500) NOT NULL,
    entity_type    VARCHAR(30)  NOT NULL,
    entity_label   VARCHAR(50),
    canonical_id   TEXT,
    canonical_text TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, text, entity_type),
    CONSTRAINT fk_entities_canonical_id_canonical_entities_id
        FOREIGN KEY (canonical_id) REFERENCES canonical_entities (id)
);

COMMENT ON TABLE entities IS '实体表';
COMMENT ON COLUMN entities.id IS '实体标识';
COMMENT ON COLUMN entities.user_id IS '用户标识';
COMMENT ON COLUMN entities.text IS '实体原始提及文本';
COMMENT ON COLUMN entities.entity_type IS '实体类型';
COMMENT ON COLUMN entities.canonical_id IS '规范化实体标识';
COMMENT ON COLUMN entities.canonical_text IS '规范化实体文本';
COMMENT ON COLUMN entities.created_at IS '创建时间戳';
COMMENT ON COLUMN entities.updated_at IS '最后更新时间戳';

-- 事实-实体关联表
CREATE TABLE IF NOT EXISTS fact_entities
(
    fact_id          TEXT,
    entity_id        TEXT,
    relation_to_user TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, entity_id),
    CONSTRAINT fk_fact_entities_fact_id_facts_id
        FOREIGN KEY (fact_id) REFERENCES facts (id),
    CONSTRAINT fk_fact_entities_entity_id_entities_id
        FOREIGN KEY (entity_id) REFERENCES entities (id)
);

COMMENT ON TABLE fact_entities IS '事实与实体的关联表';
COMMENT ON COLUMN fact_entities.fact_id IS '事实标识';
COMMENT ON COLUMN fact_entities.entity_id IS '实体标识';
COMMENT ON COLUMN fact_entities.relation_to_user IS '实体与用户的关系描述';
COMMENT ON COLUMN fact_entities.created_at IS '创建时间戳';
COMMENT ON COLUMN fact_entities.updated_at IS '最后更新时间戳';

-- 事实图构建过程表
CREATE TABLE IF NOT EXISTS fact_processing_logs
(
    id         SERIAL PRIMARY KEY,
    fact_id    TEXT,
    stage      VARCHAR(50) NOT NULL,
    status     VARCHAR(20) DEFAULT 'pending',
    message    TEXT,
    created_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_fact_graph_construction_fact_id_facts_id
        FOREIGN KEY (fact_id) REFERENCES facts (id)
);

COMMENT ON TABLE fact_processing_logs IS '事实图构建过程日志';
COMMENT ON COLUMN fact_processing_logs.id IS '日志记录标识';
COMMENT ON COLUMN fact_processing_logs.fact_id IS '相关事实标识';
COMMENT ON COLUMN fact_processing_logs.stage IS '处理阶段描述';
COMMENT ON COLUMN fact_processing_logs.status IS '阶段状态（如 pending、completed、failed）';
COMMENT ON COLUMN fact_processing_logs.message IS '阶段相关消息或错误信息';
COMMENT ON COLUMN fact_processing_logs.created_at IS '创建时间戳';
COMMENT ON COLUMN fact_processing_logs.updated_at IS '更新时间戳';

-- 基础数据插入

INSERT INTO segment (current_segment)
VALUES ((SELECT COALESCE(MAX(segment), 0) FROM memories))
ON CONFLICT DO NOTHING;

-- 索引

CREATE INDEX IF NOT EXISTS idx_memories_sector ON memories (primary_sector);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories (last_seen_at);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_memories_qa_pair_id ON memories (qa_pair_id);
CREATE INDEX IF NOT EXISTS idx_memories_fact_joined ON memories (fact_joined) WHERE fact_joined = FALSE;
CREATE INDEX IF NOT EXISTS idx_vectors_user ON vectors (user_id);
CREATE INDEX IF NOT EXISTS idx_waypoints_src ON waypoints (src_id);
CREATE INDEX IF NOT EXISTS idx_waypoints_dst ON waypoints (dst_id);
CREATE INDEX IF NOT EXISTS idx_stats_ts ON stats (ts);
CREATE INDEX IF NOT EXISTS idx_facts_created_at ON facts (created_at);
CREATE INDEX IF NOT EXISTS idx_facts_fact_kind ON facts (fact_kind);
CREATE INDEX IF NOT EXISTS idx_fact_entities_fact_id ON fact_entities (fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_entities_entity_id ON fact_entities (entity_id);
CREATE INDEX IF NOT EXISTS idx_entities_text ON entities (text);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_id ON entities (canonical_id);
