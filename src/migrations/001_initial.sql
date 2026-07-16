-- 001_initial.sql (PostgreSQL version)

CREATE EXTENSION IF NOT EXISTS vector;

-- 记忆表

CREATE TABLE IF NOT EXISTS memories
(
    id             TEXT PRIMARY KEY,
    user_id        TEXT             DEFAULT 'anonymous',
    role           TEXT,
    pair_id        TEXT,
    content        TEXT,
    summary        TEXT,
    primary_sector TEXT,
    sectors        TEXT,
    tags           TEXT,
    compressed_vec BYTEA,
    meta           TEXT,
    profile_joined SMALLINT         DEFAULT 0,
    session_joined SMALLINT         DEFAULT 0,
    fact_joined    SMALLINT         DEFAULT 0,
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
    CONSTRAINT chk_memories_role
        CHECK (role IS NULL OR role IN ('human', 'assistant'))
);

COMMENT ON TABLE memories IS '用户记忆及派生元数据';
COMMENT ON COLUMN memories.id IS '记忆主标识';
COMMENT ON COLUMN memories.user_id IS '用户标识';
COMMENT ON COLUMN memories.role IS '问答角色（human/assistant）';
COMMENT ON COLUMN memories.pair_id IS '问答对标识';
COMMENT ON COLUMN memories.content IS '原始记忆内容';
COMMENT ON COLUMN memories.summary IS '记忆摘要';
COMMENT ON COLUMN memories.primary_sector IS '主扇区/主题标签';
COMMENT ON COLUMN memories.sectors IS '次级扇区/主题标签';
COMMENT ON COLUMN memories.tags IS '用户或系统标签';
COMMENT ON COLUMN memories.compressed_vec IS '压缩嵌入向量字节';
COMMENT ON COLUMN memories.meta IS '元数据载荷（序列化）';
COMMENT ON COLUMN memories.profile_joined IS '是否已参与用户画像处理，0 = 否，1 = 是';
COMMENT ON COLUMN memories.session_joined IS '是否已参与会话总结处理，0 = 否，1 = 是';
COMMENT ON COLUMN memories.fact_joined IS '是否已参与事实处理，0 = 否，1 = 是';
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

-- 会话总结表
CREATE TABLE IF NOT EXISTS sessions
(
    id           TEXT PRIMARY KEY,
    user_id      TEXT,
    summary      TEXT,
    vector       VECTOR(1536),
    dialogue_ids JSONB DEFAULT '[]',
    key_facts    JSONB DEFAULT '[]',
    created_at   TIMESTAMP,
    updated_at   TIMESTAMP
);

COMMENT ON TABLE sessions IS '会话总结表';
COMMENT ON COLUMN sessions.id IS '主键';
COMMENT ON COLUMN sessions.user_id IS '用户标识';
COMMENT ON COLUMN sessions.summary IS '会话摘要';
COMMENT ON COLUMN sessions.vector IS '会话摘要的嵌入向量';
COMMENT ON COLUMN sessions.dialogue_ids IS '相关对话标识列表';
COMMENT ON COLUMN sessions.key_facts IS '会话中的关键事实列表';
COMMENT ON COLUMN sessions.created_at IS '创建时间戳';
COMMENT ON COLUMN sessions.updated_at IS '最后更新时间戳';

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
    encryption_key   TEXT,
    summary          TEXT,
    reflection_count INTEGER,
    status           SMALLINT DEFAULT 1,
    created_at       TIMESTAMP,
    updated_at       TIMESTAMP,
    UNIQUE (tenant_key, project_key, user_key)
);

COMMENT ON TABLE users IS '用户级摘要与计数器';
COMMENT ON COLUMN users.id IS '用户主键';
COMMENT ON COLUMN users.tenant_key IS '租户标识';
COMMENT ON COLUMN users.project_key IS '项目标识';
COMMENT ON COLUMN users.user_key IS '用户标识';
COMMENT ON COLUMN users.encryption_key IS '用户加密密钥（AES-256-GCM）';
COMMENT ON COLUMN users.summary IS '用户摘要文本';
COMMENT ON COLUMN users.reflection_count IS '反思计数';
COMMENT ON COLUMN users.status IS '用户状态，0 = 禁用，1 = 启用';
COMMENT ON COLUMN users.created_at IS '创建时间戳';
COMMENT ON COLUMN users.updated_at IS '最后更新时间戳';

CREATE TABLE IF NOT EXISTS user_profiles
(
    id          TEXT PRIMARY KEY,
    user_id     TEXT,
    demographic JSONB     DEFAULT '{}',
    preferences JSONB     DEFAULT '{}',
    attributes  JSONB     DEFAULT '{}',
    tags        JSONB     DEFAULT '[]',
    is_active   BOOLEAN   DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_profiles_user_id_users_id
        FOREIGN KEY (user_id) REFERENCES users (id)
);

COMMENT ON TABLE user_profiles IS '用户画像表';
COMMENT ON COLUMN user_profiles.id IS '主键';
COMMENT ON COLUMN user_profiles.user_id IS '用户标识';
COMMENT ON COLUMN user_profiles.demographic IS '用户个性特征';
COMMENT ON COLUMN user_profiles.preferences IS '用户偏好';
COMMENT ON COLUMN user_profiles.attributes IS '用户属性';
COMMENT ON COLUMN user_profiles.tags IS '用户标签列表';
COMMENT ON COLUMN user_profiles.is_active IS '是否可用';
COMMENT ON COLUMN user_profiles.created_at IS '创建时间戳';
COMMENT ON COLUMN user_profiles.updated_at IS '最后更新时间戳';

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
    id        TEXT PRIMARY KEY,
    user_id   TEXT,
    memory_id TEXT,
    model     TEXT,
    status    TEXT,
    ts        BIGINT,
    err       TEXT,
    CONSTRAINT fk_embed_logs_memory_id_memories_id
        FOREIGN KEY (memory_id) REFERENCES memories (id)
);

COMMENT ON TABLE embed_logs IS '嵌入生成日志';
COMMENT ON COLUMN embed_logs.id IS '主键';
COMMENT ON COLUMN embed_logs.user_id IS '用户标识';
COMMENT ON COLUMN embed_logs.memory_id IS '记忆标识';
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
CREATE TABLE IF NOT EXISTS graph_topics
(
    id           TEXT PRIMARY KEY,
    user_id      TEXT,
    name         TEXT NOT NULL,
    summary      TEXT,
    vector       VECTOR(1536),
    keywords     JSONB     DEFAULT '[]',
    dialogue_ids JSONB     DEFAULT '[]',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE graph_topics IS '主题表';
COMMENT ON COLUMN graph_topics.id IS '主题标识';
COMMENT ON COLUMN graph_topics.user_id IS '用户标识';
COMMENT ON COLUMN graph_topics.name IS '主题名称';
COMMENT ON COLUMN graph_topics.summary IS '主题摘要';
COMMENT ON COLUMN graph_topics.vector IS '主题摘要的嵌入向量';
COMMENT ON COLUMN graph_topics.keywords IS '主题相关关键词列表';
COMMENT ON COLUMN graph_topics.dialogue_ids IS '相关对话标识列表';
COMMENT ON COLUMN graph_topics.created_at IS '创建时间戳';
COMMENT ON COLUMN graph_topics.updated_at IS '最后更新时间戳';

-- 事实表
CREATE TABLE IF NOT EXISTS graph_facts
(
    id             TEXT PRIMARY KEY,
    user_id        TEXT,
    topic_id       TEXT,
    what           TEXT        NOT NULL,
    when_          TEXT,
    where_         TEXT,
    who            TEXT,
    why            TEXT,
    vector         VECTOR(1536),
    confidence     FLOAT                DEFAULT 0.0,
    fact_kind      VARCHAR(20) NOT NULL DEFAULT 'conversation',
    occurred_start TIMESTAMP,
    occurred_end   TIMESTAMP,
    created_at     TIMESTAMP,
    updated_at     TIMESTAMP,
    processed_at   TIMESTAMP,
    CONSTRAINT fk_graph_facts_topic_id_graph_topics_id
        FOREIGN KEY (topic_id) REFERENCES graph_topics (id)
);

COMMENT ON TABLE graph_facts IS '事实表';
COMMENT ON COLUMN graph_facts.id IS '事实标识';
COMMENT ON COLUMN graph_facts.user_id IS '用户标识';
COMMENT ON COLUMN graph_facts.topic_id IS '主题 ID';
COMMENT ON COLUMN graph_facts.what IS '事实内容';
COMMENT ON COLUMN graph_facts.when_ IS '事实发生的时间描述';
COMMENT ON COLUMN graph_facts.where_ IS '事实发生的地点描述';
COMMENT ON COLUMN graph_facts.who IS '事实相关的主体描述';
COMMENT ON COLUMN graph_facts.why IS '事实发生的原因描述';
COMMENT ON COLUMN graph_facts.vector IS '事实的语义向量（由 5W 组合生成）';
COMMENT ON COLUMN graph_facts.confidence IS '事实的置信度评分';
COMMENT ON COLUMN graph_facts.fact_kind IS '事实类型（如 conversation、event 等）';
COMMENT ON COLUMN graph_facts.occurred_start IS '事实发生的开始时间';
COMMENT ON COLUMN graph_facts.occurred_end IS '事实发生的结束时间';
COMMENT ON COLUMN graph_facts.created_at IS '创建时间戳';
COMMENT ON COLUMN graph_facts.updated_at IS '最后更新时间戳';
COMMENT ON COLUMN graph_facts.processed_at IS '事实被图谱化的时间戳';

-- 规范化实体表
CREATE TABLE IF NOT EXISTS graph_canonical_entities
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
    is_active        BOOLEAN   DEFAULT TRUE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, name, entity_type)
);

COMMENT ON TABLE graph_canonical_entities IS '规范化实体表';
COMMENT ON COLUMN graph_canonical_entities.id IS '实体标识';
COMMENT ON COLUMN graph_canonical_entities.user_id IS '用户标识';
COMMENT ON COLUMN graph_canonical_entities.name IS '实体名称';
COMMENT ON COLUMN graph_canonical_entities.entity_type IS '实体类型';
COMMENT ON COLUMN graph_canonical_entities.vector IS '实体的嵌入向量（用于进行相似度比较）';
COMMENT ON COLUMN graph_canonical_entities.occurrence_count IS '实体出现次数';
COMMENT ON COLUMN graph_canonical_entities.first_seen_at IS '实体首次出现时间';
COMMENT ON COLUMN graph_canonical_entities.last_seen_at IS '实体最后出现时间';
COMMENT ON COLUMN graph_canonical_entities.is_active IS '实体是否可用';
COMMENT ON COLUMN graph_canonical_entities.created_at IS '创建时间戳';
COMMENT ON COLUMN graph_canonical_entities.updated_at IS '最后更新时间戳';

-- 实体表
CREATE TABLE IF NOT EXISTS graph_entities
(
    id             TEXT PRIMARY KEY,
    user_id        TEXT,
    text           VARCHAR(500) NOT NULL,
    entity_type    VARCHAR(30)  NOT NULL,
    canonical_id   TEXT,
    canonical_name TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_graph_entities_user_text_type
        UNIQUE (user_id, text, entity_type),
    CONSTRAINT fk_graph_entities_canonical_id_graph_canonical_entities_id
        FOREIGN KEY (canonical_id) REFERENCES graph_canonical_entities (id)
);

COMMENT ON TABLE graph_entities IS '实体表';
COMMENT ON COLUMN graph_entities.id IS '主键';
COMMENT ON COLUMN graph_entities.user_id IS '用户标识';
COMMENT ON COLUMN graph_entities.text IS '实体原始提及文本';
COMMENT ON COLUMN graph_entities.entity_type IS '实体类型';
COMMENT ON COLUMN graph_entities.canonical_id IS '规范化实体标识';
COMMENT ON COLUMN graph_entities.canonical_name IS '规范化实体名称';
COMMENT ON COLUMN graph_entities.created_at IS '创建时间戳';
COMMENT ON COLUMN graph_entities.updated_at IS '最后更新时间戳';

-- 事实-实体关联表
CREATE TABLE IF NOT EXISTS graph_fact_entities
(
    id               TEXT PRIMARY KEY,
    user_id          TEXT,
    fact_id          TEXT,
    entity_id        TEXT,
    canonical_id     TEXT,
    relation_to_user TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_graph_fact_entities_user_fact_entity
        UNIQUE (user_id, fact_id, entity_id),
    CONSTRAINT fk_graph_fact_entities_graph_fact_id_facts_id
        FOREIGN KEY (fact_id) REFERENCES graph_facts (id),
    CONSTRAINT fk_graph_fact_entities_entity_id_entities_id
        FOREIGN KEY (entity_id) REFERENCES graph_entities (id),
    CONSTRAINT fk_graph_fact_entities_canonical_id_graph_canonical_entities_id
        FOREIGN KEY (canonical_id) REFERENCES graph_canonical_entities (id)
);

COMMENT ON TABLE graph_fact_entities IS '事实与实体的关联表';
COMMENT ON COLUMN graph_fact_entities.id IS '主键';
COMMENT ON COLUMN graph_fact_entities.user_id IS '用户标识';
COMMENT ON COLUMN graph_fact_entities.fact_id IS '事实标识';
COMMENT ON COLUMN graph_fact_entities.entity_id IS '实体标识';
COMMENT ON COLUMN graph_fact_entities.canonical_id IS '规范化实体标识';
COMMENT ON COLUMN graph_fact_entities.relation_to_user IS '实体与用户的关系描述';
COMMENT ON COLUMN graph_fact_entities.created_at IS '创建时间戳';
COMMENT ON COLUMN graph_fact_entities.updated_at IS '最后更新时间戳';

-- 实体-实体关联表
CREATE TABLE IF NOT EXISTS graph_entity_relations
(
    id                  TEXT PRIMARY KEY,
    user_id             TEXT,
    source_canonical_id TEXT,
    target_canonical_id TEXT,
    edge_relation       VARCHAR(50),
    relation_evidence   TEXT,
    infer_source        VARCHAR(50),
    confidence          DOUBLE PRECISION,
    fact_ids            JSONB     DEFAULT '[]',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_graph_entity_relations_no_self
        CHECK (source_canonical_id <> target_canonical_id),
    CONSTRAINT uq_graph_entity_relations_user_src_dst_type
        UNIQUE (user_id, source_canonical_id, target_canonical_id, edge_relation),
    CONSTRAINT fk_graph_entity_relations_sc_id_graph_canonical_entities_id
        FOREIGN KEY (source_canonical_id) REFERENCES graph_canonical_entities (id),
    CONSTRAINT fk_graph_entity_relations_tc_id_graph_canonical_entities_id
        FOREIGN KEY (target_canonical_id) REFERENCES graph_canonical_entities (id)
);

COMMENT ON TABLE graph_entity_relations IS '实体与实体之间的关联表';
COMMENT ON COLUMN graph_entity_relations.id IS '主键';
COMMENT ON COLUMN graph_entity_relations.user_id IS '用户标识';
COMMENT ON COLUMN graph_entity_relations.source_canonical_id IS '源规范化实体标识';
COMMENT ON COLUMN graph_entity_relations.target_canonical_id IS '目标规范化实体标识';
COMMENT ON COLUMN graph_entity_relations.edge_relation IS '边关系';
COMMENT ON COLUMN graph_entity_relations.relation_evidence IS '形成关系的证据';
COMMENT ON COLUMN graph_entity_relations.infer_source IS '关系推断来源（rule/llm/fallback）';
COMMENT ON COLUMN graph_entity_relations.confidence IS '关系推断置信度';
COMMENT ON COLUMN graph_entity_relations.fact_ids IS '与该实体关系相关的事实标识列表';
COMMENT ON COLUMN graph_entity_relations.created_at IS '创建时间戳';
COMMENT ON COLUMN graph_entity_relations.updated_at IS '最后更新时间戳';

-- 基础数据插入

INSERT INTO segment (current_segment)
VALUES ((SELECT COALESCE(MAX(segment), 0) FROM memories))
ON CONFLICT DO NOTHING;

-- 索引

CREATE INDEX IF NOT EXISTS idx_memories_sector ON memories (primary_sector);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories (last_seen_at);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_memories_qa_pair_id ON memories (qa_pair_id);
CREATE INDEX IF NOT EXISTS idx_memories_fact_joined ON memories (fact_joined) WHERE fact_joined = 0;
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_vectors_user ON vectors (user_id);
CREATE INDEX IF NOT EXISTS idx_vectors_v ON vectors USING hnsw (v vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_waypoints_src ON waypoints (src_id);
CREATE INDEX IF NOT EXISTS idx_waypoints_dst ON waypoints (dst_id);
CREATE INDEX IF NOT EXISTS idx_stats_ts ON stats (ts);
CREATE INDEX IF NOT EXISTS idx_embed_logs_user ON embed_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles (user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_active_updated ON user_profiles (user_id, is_active, updated_at);
CREATE UNIQUE INDEX IF NOT EXISTS uq_user_profiles_user_active_true
    ON user_profiles (user_id)
    WHERE is_active = TRUE AND user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_graph_topics_vector ON graph_topics USING hnsw (vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_graph_facts_created_at ON graph_facts (created_at);
CREATE INDEX IF NOT EXISTS idx_graph_facts_fact_kind ON graph_facts (fact_kind);
CREATE INDEX IF NOT EXISTS idx_graph_facts_occurred_start_end ON graph_facts (occurred_start, occurred_end);
CREATE INDEX IF NOT EXISTS idx_graph_facts_vector ON graph_facts USING hnsw (vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_graph_fact_entities_user_id ON graph_fact_entities (user_id);
CREATE INDEX IF NOT EXISTS idx_graph_fact_entities_fact_id ON graph_fact_entities (fact_id);
CREATE INDEX IF NOT EXISTS idx_graph_fact_entities_entity_id ON graph_fact_entities (entity_id);
CREATE INDEX IF NOT EXISTS idx_graph_fact_entities_canonical_id ON graph_fact_entities (canonical_id);
CREATE INDEX IF NOT EXISTS idx_graph_entities_user_id ON graph_entities (user_id);
CREATE INDEX IF NOT EXISTS idx_graph_entities_text ON graph_entities (text);
CREATE INDEX IF NOT EXISTS idx_graph_entities_entity_type ON graph_entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_graph_canonical_entities_name ON graph_canonical_entities (name);
CREATE INDEX IF NOT EXISTS idx_graph_canonical_entities_user_id ON graph_canonical_entities (user_id);
CREATE INDEX IF NOT EXISTS idx_graph_canonical_entities_entity_type ON graph_canonical_entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_graph_canonical_entities_entity_label ON graph_canonical_entities (entity_label);
CREATE INDEX IF NOT EXISTS idx_graph_canonical_entities_vector ON graph_canonical_entities USING hnsw (vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_graph_entity_relations_user_id ON graph_entity_relations (user_id);
CREATE INDEX IF NOT EXISTS idx_graph_entity_relations_source_canonical_id ON graph_entity_relations (source_canonical_id);
CREATE INDEX IF NOT EXISTS idx_graph_entity_relations_target_canonical_id ON graph_entity_relations (target_canonical_id);
CREATE INDEX IF NOT EXISTS idx_graph_entity_relations_edge_relation ON graph_entity_relations (edge_relation);
CREATE INDEX IF NOT EXISTS idx_graph_entity_relations_infer_source ON graph_entity_relations (infer_source);
