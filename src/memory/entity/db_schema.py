import uuid
from enum import IntEnum

from sqlalchemy import Column, Integer, DateTime, func, Boolean, String, Text, Float
from sqlalchemy.dialects.mssql import DOUBLE_PRECISION
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID, BYTEA
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class BaseEntity(Base):
    """
    基础模型类
    """

    # 标记为抽象类，不会生成实际数据库表
    __abstract__ = True

    # 属性
    extra = Column(JSONB, nullable=True, comment="扩展字段")
    is_deleted = Column(Boolean, nullable=False, default=False, index=True, comment="业务状态：0-正常，1-删除")
    created_at = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    created_by = Column(String(64), nullable=True, comment="创建人")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    updated_by = Column(String(64), nullable=True, comment="更新人")


"""
CREATE TABLE IF NOT EXISTS memories
(
    id             TEXT PRIMARY KEY,
    user_id        TEXT             DEFAULT 'anonymous',
    qa_role        TEXT,
    qa_pair_id     TEXT,
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
"""


class Memories(BaseEntity):
    __tablename__ = "memories"
    __table_args__ = (
        {"comment": "用户记忆及派生元数据"}
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="ID")
    user_id = Column(String(64), nullable=True, default="anonymous", server_default="anonymous", index=True, comment="用户标识")
    qa_role = Column(String(64), nullable=True, comment="问答角色（human/assistant）")
    qa_pair_id = Column(String(64), nullable=True, comment="问答对标识，用于关联同一轮问答的记忆")
    content = Column(Text, nullable=True, comment="原始记忆内容")
    summary = Column(Text, nullable=True, comment="记忆摘要")
    primary_sector = Column(String(128), nullable=True, comment="主扇区/主题标签")
    sectors = Column(String(512), nullable=True, comment="次级扇区/主题标签")
    tags = Column(String(512), nullable=True, comment="用户或系统标签")
    compressed_vec = Column(BYTEA, nullable=True, comment="压缩嵌入向量字节")
    meta = Column(Text, nullable=True, comment="元数据载荷（序列化）")
    profile_joined = Column(Boolean, nullable=False, default=False, comment="是否已参与用户画像处理，0 = 否，1 = 是")
    session_joined = Column(Boolean, nullable=False, default=False, comment="是否已参与会话总结处理，0 = 否，1 = 是")
    fact_joined = Column(Boolean, nullable=False, default=False, comment="是否已参与事实处理，0 = 否，1 = 是")
    segment = Column(Integer, nullable=False, default=0, comment="分段编号")
    last_seen_at = Column(DateTime, nullable=True, comment="最后访问时间戳")
    salience = Column(DOUBLE_PRECISION, nullable=True, comment="显著性得分")
    decay_lambda = Column(DOUBLE_PRECISION, nullable=True, comment="衰减率（lambda）")
    version = Column(Integer, nullable=True, comment="版本标记")
    mean_dim = Column(Integer, nullable=True, comment="均值嵌入维度")
    mean_vec = Column(BYTEA, nullable=True, comment="均值嵌入向量字节")
    feedback_score = Column(DOUBLE_PRECISION, nullable=True, default=0, comment="用户反馈得分")


class GraphEntityRelations(Base):
    pass


async def init_db_schema():
    """
    初始化数据库表结构
    """
    pass
