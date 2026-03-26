import datetime
import json
import uuid

from agile.utils import LogHelper

from src.core.db import get_db
from src.memory.graph.semantic_split import Topic
from src.memory.models.graph_models import Fact, Entity, EntityType
from src.memory.models.memory_models import IMemoryUser

logger = LogHelper.get_logger()

db = get_db()


async def add_topic(user: IMemoryUser, topic: Topic, conn=None):
    """
    添加主题
    :param user:
    :param topic:
    :param conn:
    :return:
    """
    now = datetime.datetime.now()
    topic_id = str(uuid.uuid4())
    topic.set_id(topic_id)
    db.execute(
        """
        INSERT INTO graph_topics(id, user_id, name, summary, keywords, dialogue_ids, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            topic_id,
            user.id,
            topic.name,
            topic.summary,
            json.dumps(topic.keywords or []),
            json.dumps(topic.dialogue_ids or []),
            now,
            now
        ),
        conn=conn
    )
    if conn is None:
        db.commit()
    return topic


async def add_fact(user: IMemoryUser, fact: Fact, topic: Topic, conn=None) -> Fact:
    """
    添加事实
    :param user:
    :param fact:
    :param topic:
    :param conn:
    :return:
    """
    now = datetime.datetime.now()
    fact_id = str(uuid.uuid4())
    fact.set_id(fact_id)
    db.execute(
        """
        INSERT INTO graph_facts (id, user_id, topic_id, what, when_, where_, who, why, status, fact_kind, occurred_start, occurred_end,
                                 created_at, updated_at, processed_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            fact_id,
            user.id,
            topic.id,
            fact.what,
            fact.when,
            fact.where,
            fact.who,
            fact.why,
            "pending",
            fact.fact_kind or "conversation",
            fact.occurred_start,
            fact.occurred_end,
            now,
            now,
            None
        ),
        conn=conn
    )
    if conn is None:
        db.commit()
    return fact


async def add_entity(entity: Entity, conn=None) -> Entity:
    """
    添加实体
    :param entity:
    :param conn:
    :return:
    """
    text = getattr(entity, "text", None)
    entity_type = getattr(entity.entity_type, 'name', EntityType.OTHER.name)

    # 如果存在则不添加实体（text 和 entity_type 相同，视为同一个实体）
    existing = db.fetchone(
        "SELECT id FROM graph_entities WHERE text = %s AND entity_type = %s",
        (text, entity_type),
        conn=conn
    )
    if existing:
        entity.set_id(existing["id"])
        entity.set_canonical_id(existing["canonical_id"])
        return entity

    now = datetime.datetime.now()
    entity_id = str(uuid.uuid4())
    entity.set_id(entity_id)
    db.execute(
        """
        INSERT INTO graph_entities (id, text, entity_type, canonical_id, canonical_text, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            entity_id,
            text,
            entity_type,
            getattr(entity, 'canonical_id', None),
            getattr(entity, 'canonical_text', None),
            now,
            now
        ),
        conn=conn
    )
    if conn is None:
        db.commit()
    return entity


async def link_fact_entities(fact: Fact, conn=None) -> None:
    """
    关联事实和实体关系
    :param fact:
    :param conn:
    :return:
    """
    now = datetime.datetime.now()
    for entity in fact.entities:
        if not entity.id:
            entity = await add_entity(entity=entity, conn=conn)
        db.execute(
            """
            INSERT INTO graph_fact_entities (fact_id, entity_id, relation_to_user, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                fact.id,
                entity.id,
                entity.relation_to_user,
                now,
                now
            ),
            conn=conn
        )
    if conn is None:
        db.commit()


async def mark_memoires_to_fact_joined(m_ids: list[str], conn=None) -> int:
    """
    将记忆标记为已参与事实处理
    :param m_ids:
    :param conn:
    :return:
    """
    if not m_ids:
        return 0
    format_strings = ','.join(['%s'] * len(m_ids))
    query = f"UPDATE memories SET fact_joined = 1 WHERE id IN ({format_strings})"
    affected_rows = db.execute(
        query,
        tuple(m_ids),
        conn=conn
    )
    if conn is None:
        db.commit()
    return affected_rows


async def increment_memoires_join_count(m_ids: list[str], discard_threshold: int = 2, conn=None) -> int:
    """
    将记忆的参与处理次数自增
    如果 joined_count >= discard_threshold，且未参与过事实处理，则标记为 -1，标识丢弃，后续不参加任何事实处理逻辑
    :param m_ids:
    :param discard_threshold:
    :param conn:
    :return:
    """
    if not m_ids:
        return 0
    format_strings = ','.join(['%s'] * len(m_ids))
    query = f"""
        UPDATE memories 
        SET 
            joined_count = joined_count + 1,
            fact_joined = CASE WHEN joined_count + 1 >= %s AND fact_joined = 0 THEN -1 ELSE fact_joined END
        WHERE id IN ({format_strings})
    """
    affected_rows = db.execute(
        query,
        (discard_threshold,) + tuple(m_ids),
        conn=conn
    )
    if conn is None:
        db.commit()
    return affected_rows
