import datetime
import uuid

from agile.utils import LogHelper

from src.core.db import get_db
from src.memory.graph.semantic_split import Topic
from src.memory.models.graph_models import Fact, Entity, EntityType
from src.memory.models.memory_models import IMemoryUserIdentity

logger = LogHelper.get_logger()

db = get_db()


async def add_topic(user_identity: IMemoryUserIdentity, topic: Topic):
    user_id = user_identity.user_id
    tenant_id = user_identity.tenant_id
    project_id = user_identity.project_id

    now = datetime.datetime.now()
    topic_id = str(uuid.uuid4())
    topic._id = topic_id

    db.execute(
        """
        INSERT INTO topics(id, tenant_id, project_id, user_id, name, summary, keywords, dialogue_ids, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            topic_id,
            tenant_id,
            project_id,
            user_id,
            topic.name,
            topic.summary,
            topic.keywords or [],
            topic.dialogue_ids or [],
            now,
            now
        )
    )
    db.commit()
    return topic


async def add_fact(user_identity: IMemoryUserIdentity, fact: Fact, topic: Topic) -> Fact:
    user_id = user_identity.user_id
    tenant_id = user_identity.tenant_id
    project_id = user_identity.project_id

    now = datetime.datetime.now()
    fact_id = str(uuid.uuid4())
    fact._id = fact_id

    db.execute(
        """
        INSERT INTO facts (id, tenant_id, project_id, user_id, topic_id, what, when_, where_, who, why, status, fact_kind, occurred_start, occurred_end,
                           created_at, updated_at, processed_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            fact_id,
            tenant_id,
            project_id,
            user_id,
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
        )
    )
    db.commit()
    return fact


async def add_entity(user_identity: IMemoryUserIdentity, entity: Entity) -> Entity:
    user_id = user_identity.user_id
    tenant_id = user_identity.tenant_id
    project_id = user_identity.project_id

    now = datetime.datetime.now()
    entity_id = str(uuid.uuid4())
    entity._id = entity_id
    entity_type = EntityType.from_value(entity.entity_type.lower())

    db.execute(
        """
        INSERT INTO entities (id, tenant_id, project_id, user_id, text, entity_type, canonical_id, canonical_text, occurrence_count,
                              first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            entity_id,
            tenant_id,
            project_id,
            user_id,
            entity.text,
            entity_type.name,
            entity.canonical_id,
            entity.canonical_text,
            1,
            now,
            now,
            now,
            now
        ))
    db.commit()
    return entity


async def link_fact_entities(user_identity: IMemoryUserIdentity, fact: Fact) -> None:
    now = datetime.datetime.now()
    for entity in fact.entities:
        if not entity.id:
            entity = await add_entity(user_identity=user_identity, entity=entity)
        db.execute(
            """
            INSERT INTO fact_entities (fact_id, entity_id, relation_to_user, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                fact.id,
                entity.id,
                entity.relation_to_user,
                now,
                now
            )
        )
