from sqlalchemy import (
    BIGINT,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import BYTEA, JSONB
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import VECTOR
from urllib.parse import urlparse

from agile.utils import LogHelper

from shared.config.settings import env

Base = declarative_base()
logger = LogHelper.get_logger(title="[DB_SCHEMA]")


class Memories(Base):
    __tablename__ = "memories"
    __table_args__ = (
        CheckConstraint("qa_role IS NULL OR qa_role IN ('human', 'assistant')", name="chk_memories_qa_role"),
        Index("idx_memories_sector", "primary_sector"),
        Index("idx_memories_ts", "last_seen_at"),
        Index("idx_memories_user", "user_id"),
        Index("idx_memories_qa_pair_id", "qa_pair_id"),
        Index("idx_memories_fact_joined", "fact_joined", postgresql_where=text("fact_joined = false")),
        {"comment": "用户记忆及派生元数据"},
    )

    id = Column(String(64), primary_key=True, comment="记忆主标识")
    user_id = Column(String(64), nullable=False, server_default="anonymous", comment="用户标识")
    qa_role = Column(String(16), nullable=True, comment="问答角色（human/assistant）")
    qa_pair_id = Column(String(64), nullable=True, comment="问答对标识")
    content = Column(Text, nullable=True, comment="原始记忆内容")
    summary = Column(Text, nullable=True, comment="记忆摘要")
    primary_sector = Column(String(128), nullable=True, comment="主扇区/主题标签")
    sectors = Column(JSONB, nullable=True, server_default=text("'[]'::jsonb"), comment="次级扇区/主题标签")
    tags = Column(JSONB, nullable=True, server_default=text("'[]'::jsonb"), comment="用户或系统标签")
    compressed_vec = Column(BYTEA, nullable=True, comment="压缩嵌入向量字节")
    meta = Column(JSONB, nullable=True, server_default=text("'{}'::jsonb"), comment="元数据载荷")
    profile_joined = Column(Boolean, nullable=False, server_default=text("false"), comment="是否已参与用户画像处理")
    session_joined = Column(Boolean, nullable=False, server_default=text("false"), comment="是否已参与会话总结处理")
    fact_joined = Column(Boolean, nullable=False, server_default=text("false"), comment="是否已参与事实处理")
    segment = Column(Integer, nullable=False, server_default="0", comment="分段编号")
    created_at = Column(DateTime, nullable=True, comment="创建时间戳")
    updated_at = Column(DateTime, nullable=True, comment="最后更新时间戳")
    last_seen_at = Column(DateTime, nullable=True, comment="最后访问时间戳")
    salience = Column(Float, nullable=True, comment="显著性得分")
    decay_lambda = Column(Float, nullable=True, comment="衰减率（lambda）")
    version = Column(Integer, nullable=True, comment="版本标记")
    mean_dim = Column(Integer, nullable=True, comment="均值嵌入维度")
    mean_vec = Column(BYTEA, nullable=True, comment="均值嵌入向量字节")
    feedback_score = Column(Float, nullable=False, server_default="0", comment="用户反馈得分")


class Sessions(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_user", "user_id"),
        {"comment": "会话总结表"},
    )

    id = Column(String(64), primary_key=True, comment="主键")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    summary = Column(Text, nullable=True, comment="会话摘要")
    vector = Column(VECTOR(1536), nullable=True, comment="会话摘要的嵌入向量")
    dialogue_ids = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"), comment="相关对话标识列表")
    key_facts = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"), comment="会话中的关键事实列表")
    created_at = Column(DateTime, nullable=True, comment="创建时间戳")
    updated_at = Column(DateTime, nullable=True, comment="最后更新时间戳")


class Vectors(Base):
    __tablename__ = "vectors"
    __table_args__ = (
        Index("idx_vectors_user", "user_id"),
        Index("idx_vectors_v", "v", postgresql_using="hnsw", postgresql_ops={"v": "vector_cosine_ops"}),
        {"comment": "记忆的嵌入向量"},
    )

    id = Column(String(64), ForeignKey("memories.id"), primary_key=True, comment="记忆标识")
    sector = Column(String(128), primary_key=True, comment="扇区/主题标签")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    v = Column(VECTOR(1536), nullable=True, comment="嵌入向量")
    dim = Column(Integer, nullable=True, comment="向量维度")


class Users(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_key", "project_key", "user_key", name="uq_users_tenant_project_user"),
        {"comment": "用户级摘要与计数器"},
    )

    id = Column(String(64), primary_key=True, comment="用户主键")
    tenant_key = Column(String(128), nullable=True, comment="租户标识")
    project_key = Column(String(128), nullable=True, comment="项目标识")
    user_key = Column(String(128), nullable=True, comment="用户标识")
    encryption_key = Column(String(128), nullable=True, comment="用户加密密钥（AES-256-GCM）")
    summary = Column(Text, nullable=True, comment="用户摘要文本")
    reflection_count = Column(Integer, nullable=True, comment="反思计数")
    status = Column(Integer, nullable=False, server_default="1", comment="用户状态，0 = 禁用，1 = 启用")
    created_at = Column(DateTime, nullable=True, comment="创建时间戳")
    updated_at = Column(DateTime, nullable=True, comment="最后更新时间戳")


class UserProfiles(Base):
    __tablename__ = "user_profiles"
    __table_args__ = ({"comment": "用户画像表"},)

    id = Column(String(64), primary_key=True, comment="主键")
    user_id = Column(String(64), ForeignKey("users.id"), nullable=True, comment="用户标识")
    demographic = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"), comment="用户个性特征")
    preferences = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"), comment="用户偏好")
    attributes = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"), comment="用户属性")
    tags = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"), comment="用户标签列表")
    is_active = Column(Boolean, nullable=False, server_default=text("true"), comment="是否可用")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间戳")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), comment="最后更新时间戳")


class Stats(Base):
    __tablename__ = "stats"
    __table_args__ = (
        Index("idx_stats_ts", "ts"),
        {"comment": "运行指标快照"},
    )

    id = Column(Integer, primary_key=True, autoincrement=True, comment="统计记录标识")
    ts = Column(BIGINT, nullable=True, comment="Unix 时间戳（毫秒）")
    metrics = Column(JSONB, nullable=True, comment="指标载荷")


class EmbedLogs(Base):
    __tablename__ = "embed_logs"
    __table_args__ = (
        Index("idx_embed_logs_user", "user_id"),
        {"comment": "嵌入生成日志"},
    )

    id = Column(String(64), primary_key=True, comment="主键")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    memory_id = Column(String(64), ForeignKey("memories.id"), nullable=True, comment="记忆标识")
    model = Column(String(128), nullable=True, comment="嵌入模型名称")
    status = Column(String(32), nullable=True, comment="嵌入任务状态")
    ts = Column(BIGINT, nullable=True, comment="Unix 时间戳（毫秒）")
    err = Column(Text, nullable=True, comment="失败时错误信息")


class Waypoints(Base):
    __tablename__ = "waypoints"
    __table_args__ = (
        Index("idx_waypoints_src", "src_id"),
        Index("idx_waypoints_dst", "dst_id"),
        {"comment": "相关记忆之间的边"},
    )

    src_id = Column(String(64), ForeignKey("memories.id"), primary_key=True, comment="源记忆标识")
    dst_id = Column(String(64), ForeignKey("memories.id"), primary_key=True, comment="目标记忆标识")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    weight = Column(Float, nullable=True, comment="边权重")
    created_at = Column(DateTime, nullable=True, comment="创建时间戳")
    updated_at = Column(DateTime, nullable=True, comment="最后更新时间戳")


class Segment(Base):
    __tablename__ = "segment"
    __table_args__ = ({"comment": "全局分段跟踪"},)

    id = Column(Integer, primary_key=True, autoincrement=True, comment="分段记录标识")
    current_segment = Column(BIGINT, nullable=False, server_default="0", comment="当前分段值")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间戳")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), comment="最后更新时间戳")


class GraphTopics(Base):
    __tablename__ = "graph_topics"
    __table_args__ = (
        Index("idx_graph_topics_vector", "vector", postgresql_using="hnsw", postgresql_ops={"vector": "vector_cosine_ops"}),
        {"comment": "主题表"},
    )

    id = Column(String(64), primary_key=True, comment="主题标识")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    name = Column(String(255), nullable=False, comment="主题名称")
    summary = Column(Text, nullable=True, comment="主题摘要")
    vector = Column(VECTOR(1536), nullable=True, comment="主题摘要的嵌入向量")
    keywords = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"), comment="主题相关关键词列表")
    dialogue_ids = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"), comment="相关对话标识列表")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间戳")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), comment="最后更新时间戳")


class GraphFacts(Base):
    __tablename__ = "graph_facts"
    __table_args__ = (
        Index("idx_graph_facts_created_at", "created_at"),
        Index("idx_graph_facts_fact_kind", "fact_kind"),
        Index("idx_graph_facts_occurred_start_end", "occurred_start", "occurred_end"),
        Index("idx_graph_facts_vector", "vector", postgresql_using="hnsw", postgresql_ops={"vector": "vector_cosine_ops"}),
        {"comment": "事实表"},
    )

    id = Column(String(64), primary_key=True, comment="事实标识")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    topic_id = Column(String(64), ForeignKey("graph_topics.id"), nullable=True, comment="主题 ID")
    what = Column(Text, nullable=False, comment="事实内容")
    when_ = Column(String(255), nullable=True, comment="事实发生的时间描述")
    where_ = Column(String(255), nullable=True, comment="事实发生的地点描述")
    who = Column(String(255), nullable=True, comment="事实相关的主体描述")
    why = Column(String(500), nullable=True, comment="事实发生的原因描述")
    vector = Column(VECTOR(1536), nullable=True, comment="事实的语义向量")
    confidence = Column(Float, nullable=False, server_default="0", comment="事实的置信度评分")
    fact_kind = Column(String(20), nullable=False, server_default="conversation", comment="事实类型")
    occurred_start = Column(DateTime, nullable=True, comment="事实发生的开始时间")
    occurred_end = Column(DateTime, nullable=True, comment="事实发生的结束时间")
    created_at = Column(DateTime, nullable=True, comment="创建时间戳")
    updated_at = Column(DateTime, nullable=True, comment="最后更新时间戳")
    processed_at = Column(DateTime, nullable=True, comment="事实被图谱化的时间戳")


class GraphCanonicalEntities(Base):
    __tablename__ = "graph_canonical_entities"
    __table_args__ = (
        UniqueConstraint("user_id", "name", "entity_type", name="uq_graph_canonical_entities_user_name_type"),
        Index("idx_graph_canonical_entities_name", "name"),
        Index("idx_graph_canonical_entities_user_id", "user_id"),
        Index("idx_graph_canonical_entities_entity_type", "entity_type"),
        Index("idx_graph_canonical_entities_entity_label", "entity_label"),
        Index(
            "idx_graph_canonical_entities_vector",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
        {"comment": "规范化实体表"},
    )

    id = Column(String(64), primary_key=True, comment="实体标识")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    name = Column(String(500), nullable=False, comment="实体名称")
    entity_type = Column(String(30), nullable=False, comment="实体类型")
    entity_label = Column(String(50), nullable=True, comment="实体标签")
    vector = Column(VECTOR(1536), nullable=True, comment="实体的嵌入向量")
    occurrence_count = Column(Integer, nullable=False, server_default="1", comment="实体出现次数")
    first_seen_at = Column(DateTime, nullable=False, server_default=func.now(), comment="实体首次出现时间")
    last_seen_at = Column(DateTime, nullable=False, server_default=func.now(), comment="实体最后出现时间")
    is_active = Column(Boolean, nullable=False, server_default=text("true"), comment="实体是否可用")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间戳")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), comment="最后更新时间戳")


class GraphEntities(Base):
    __tablename__ = "graph_entities"
    __table_args__ = (
        UniqueConstraint("user_id", "text", "entity_type", name="uq_graph_entities_user_text_type"),
        Index("idx_graph_entities_user_id", "user_id"),
        Index("idx_graph_entities_text", "text"),
        Index("idx_graph_entities_entity_type", "entity_type"),
        {"comment": "实体表"},
    )

    id = Column(String(64), primary_key=True, comment="主键")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    text = Column(String(500), nullable=False, comment="实体原始提及文本")
    entity_type = Column(String(30), nullable=False, comment="实体类型")
    canonical_id = Column(String(64), ForeignKey("graph_canonical_entities.id"), nullable=True, comment="规范化实体标识")
    canonical_name = Column(String(500), nullable=True, comment="规范化实体名称")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间戳")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), comment="最后更新时间戳")


class GraphFactEntities(Base):
    __tablename__ = "graph_fact_entities"
    __table_args__ = (
        UniqueConstraint("user_id", "fact_id", "entity_id", name="uq_graph_fact_entities_user_fact_entity"),
        Index("idx_graph_fact_entities_user_id", "user_id"),
        Index("idx_graph_fact_entities_fact_id", "fact_id"),
        Index("idx_graph_fact_entities_entity_id", "entity_id"),
        Index("idx_graph_fact_entities_canonical_id", "canonical_id"),
        {"comment": "事实与实体的关联表"},
    )

    id = Column(String(64), primary_key=True, comment="主键")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    fact_id = Column(String(64), ForeignKey("graph_facts.id"), nullable=True, comment="事实标识")
    entity_id = Column(String(64), ForeignKey("graph_entities.id"), nullable=True, comment="实体标识")
    canonical_id = Column(String(64), ForeignKey("graph_canonical_entities.id"), nullable=True, comment="规范化实体标识")
    relation_to_user = Column(String(255), nullable=True, comment="实体与用户的关系描述")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间戳")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), comment="最后更新时间戳")


class GraphEntityRelations(Base):
    __tablename__ = "graph_entity_relations"
    __table_args__ = (
        CheckConstraint("source_canonical_id <> target_canonical_id", name="chk_graph_entity_relations_no_self"),
        UniqueConstraint(
            "user_id",
            "source_canonical_id",
            "target_canonical_id",
            "edge_relation",
            name="uq_graph_entity_relations_user_src_dst_type",
        ),
        Index("idx_graph_entity_relations_user_id", "user_id"),
        Index("idx_graph_entity_relations_source_canonical_id", "source_canonical_id"),
        Index("idx_graph_entity_relations_target_canonical_id", "target_canonical_id"),
        Index("idx_graph_entity_relations_edge_relation", "edge_relation"),
        Index("idx_graph_entity_relations_infer_source", "infer_source"),
        {"comment": "实体与实体之间的关联表"},
    )

    id = Column(String(64), primary_key=True, comment="主键")
    user_id = Column(String(64), nullable=True, comment="用户标识")
    source_canonical_id = Column(
        String(64), ForeignKey("graph_canonical_entities.id"), nullable=True, comment="源规范化实体标识"
    )
    target_canonical_id = Column(
        String(64), ForeignKey("graph_canonical_entities.id"), nullable=True, comment="目标规范化实体标识"
    )
    edge_relation = Column(String(50), nullable=True, comment="边关系")
    relation_evidence = Column(Text, nullable=True, comment="形成关系的证据")
    infer_source = Column(String(50), nullable=True, comment="关系推断来源")
    confidence = Column(Float, nullable=True, comment="关系推断置信度")
    fact_ids = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"), comment="关联事实 ID 列表")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间戳")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), comment="最后更新时间戳")


def normalize_sync_postgres_url(db_url: str) -> tuple[str, str]:
    """将 async PostgreSQL URL 转换为适用于 DDL 的同步 SQLAlchemy URL。"""
    parsed = urlparse(db_url)
    scheme = parsed.scheme

    if "+asyncpg" in scheme:
        sync_scheme = scheme.replace("+asyncpg", "+psycopg")
    elif "+aiopg" in scheme:
        sync_scheme = scheme.replace("+aiopg", "+psycopg")
    elif scheme.startswith("postgresql") and "psycopg" not in scheme:
        sync_scheme = "postgresql+psycopg"
    else:
        sync_scheme = scheme

    connect_uri = f"{sync_scheme}://{parsed.netloc}"
    database = parsed.path.lstrip("/")
    return connect_uri, database


def _compile_server_default(column, engine) -> str:
    if column.server_default is None:
        return ""

    default_arg = column.server_default.arg
    if hasattr(default_arg, "compile"):
        compiled = str(default_arg.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True}))
    else:
        compiled = str(default_arg)

    return f" DEFAULT {compiled}"


def _sync_missing_columns(engine):
    """
    检测并补充 ORM 模型中定义但数据库表中缺失的列，仅新增列。
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table.name)}

        with engine.begin() as conn:
            for column in table.columns.values():
                if column.name in existing_columns:
                    continue

                col_type = column.type.compile(dialect=engine.dialect)
                nullable = "" if column.nullable else " NOT NULL"
                default = _compile_server_default(column, engine)
                alter_sql = (
                    f'ALTER TABLE "{table.name}" ADD COLUMN IF NOT EXISTS '
                    f'"{column.name}" {col_type}{default}{nullable}'
                )
                conn.execute(text(alter_sql))
                logger.info(f"[DB SYNC] Added missing column: {table.name}.{column.name}")


async def init_db_schema():
    """
    初始化数据库，创建数据库、启用 pgvector 扩展，并创建表结构。
    """
    connect_uri, database = normalize_sync_postgres_url(env.POSTGRES_DB_URL)
    if not database:
        raise ValueError("Database name is missing from POSTGRES_DB_URL")

    admin_engine = create_engine(f"{connect_uri}/postgres")
    try:
        with admin_engine.connect() as raw_conn:
            conn = raw_conn.execution_options(isolation_level="AUTOCOMMIT")
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db"),
                {"db": database},
            ).scalar()
            if not exists:
                safe_database = database.replace('"', '""')
                conn.execute(text(f'CREATE DATABASE "{safe_database}"'))
                logger.info(f"Database {database} created successfully.")
    finally:
        admin_engine.dispose()

    engine = create_engine(f"{connect_uri}/{database}")
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        Base.metadata.create_all(engine)
        _sync_missing_columns(engine)
        logger.info(f"Database {database} initialized successfully.")
    finally:
        engine.dispose()

    return None
