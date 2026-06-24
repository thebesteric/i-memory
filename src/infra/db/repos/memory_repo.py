import json
from typing import Optional, Any, Dict, List, Literal, cast

from agile.utils import LogHelper, singleton, timing
from sqlalchemy import and_, asc, delete, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from infra.db.engine import get_session_factory
from infra.db.orm_models import (
    EmbedLogs,
    GraphCanonicalEntities,
    GraphEntities,
    GraphEntityRelations,
    GraphFactEntities,
    GraphFacts,
    GraphTopics,
    Memories,
    Sessions,
    UserProfiles,
    Vectors,
    Waypoints,
)
from domain.memory.models import IMemoryUser
from infra.db.repos.user_repo import load_user_encryption_keys
from services.commons.encrypt_service import encryption_enabled, encrypt_if_necessary, decrypt_if_necessary
from shared.utils.json_utils import coerce_json_field

logger = LogHelper.get_logger(title="[MEM_OPS]")


@singleton
class MemOps:

    def __init__(self):
        self._session_factory = get_session_factory

    @staticmethod
    def _model_to_dict(model_obj) -> Dict[str, Any]:
        return {c.name: getattr(model_obj, c.name) for c in model_obj.__table__.columns}

    @staticmethod
    def _maybe_decrypt_memory_row(row: Dict[str, Any], user_encryption_key: dict[str, str]) -> Dict[str, Any]:
        if not encryption_enabled() or not row:
            return row

        user_id = row.get("user_id")
        mem_id = row.get("id")
        if not user_id or not mem_id:
            return row

        key_b64 = user_encryption_key.get(str(user_id))
        if not key_b64:
            return row

        aad = {"id": mem_id}
        for field in ("content", "summary"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                continue
            row[field] = decrypt_if_necessary(value, key_b64=key_b64, aad=aad)

        return row

    @staticmethod
    def _parse_order_item(order_item: str, default_model=Memories):
        clause = (order_item or "").strip()
        if not clause:
            return None
        parts = clause.split()
        col_name = parts[0]
        direction = parts[1].lower() if len(parts) > 1 else "asc"

        model = default_model
        field = col_name
        if "." in col_name:
            alias, field = col_name.split(".", 1)
            if alias == "v":
                model = Vectors
            else:
                model = Memories

        col = model.__table__.columns.get(field)
        if col is None:
            return None
        return desc(col) if direction == "desc" else asc(col)

    @staticmethod
    def _parse_condition(condition: str, params_iter, default_model=Memories):
        cond = (condition or "").strip()
        if not cond:
            return None

        for op in ("<=", ">=", "!=", "=", "<", ">"):
            if op in cond:
                left, right = [x.strip() for x in cond.split(op, 1)]
                break
        else:
            return None

        model = default_model
        field = left
        if "." in left:
            alias, field = left.split(".", 1)
            if alias == "v":
                model = Vectors
            else:
                model = Memories

        col = model.__table__.columns.get(field)
        if col is None:
            return None

        if right == "%s":
            value = next(params_iter)
        else:
            lowered = right.lower()
            if lowered in ("true", "false"):
                value = lowered == "true"
            elif right in ("0", "1") and field.endswith("_joined"):
                value = right == "1"
            else:
                try:
                    value = int(right)
                except ValueError:
                    value = right.strip("'\"")

        if op == "=":
            return col == value
        if op == "!=":
            return col != value
        if op == "<":
            return col < value
        if op == ">":
            return col > value
        if op == "<=":
            return col <= value
        if op == ">=":
            return col >= value
        return None

    @staticmethod
    def _build_filters(conditions: list[str], params: list[Any] | None, default_model=Memories) -> list[Any]:
        params_iter = iter(params or [])
        filters = []
        for cond in conditions:
            expr = MemOps._parse_condition(cond, params_iter, default_model=default_model)
            if expr is not None:
                filters.append(expr)
        return filters

    def ins_mem(self, **k) -> int:
        payload = {
            "id": k.get("id"),
            "user_id": k.get("user_id"),
            "segment": k.get("segment", 0),
            "content": k.get("content"),
            "summary": k.get("summary"),
            "primary_sector": k.get("primary_sector"),
            "sectors": coerce_json_field(k.get("sectors"), []),
            "tags": coerce_json_field(k.get("tags"), []),
            "meta": coerce_json_field(k.get("meta"), {}),
            "created_at": k.get("created_at"),
            "updated_at": k.get("updated_at"),
            "last_seen_at": k.get("last_seen_at"),
            "salience": k.get("salience", 1.0),
            "decay_lambda": k.get("decay_lambda", 0.02),
            "version": k.get("version", 1),
            "mean_dim": k.get("mean_dim"),
            "mean_vec": k.get("mean_vec"),
            "compressed_vec": k.get("compressed_vec"),
            "feedback_score": k.get("feedback_score", 0),
            "qa_role": k.get("qa_role"),
            "qa_pair_id": k.get("qa_pair_id"),
        }
        session_factory = self._session_factory()
        with session_factory() as session:
            user_id = payload.get("user_id")
            user_encryption_key = load_user_encryption_keys(session, [str(user_id)] if user_id else [])
            key_b64 = user_encryption_key.get(str(user_id)) if user_id else None
            if key_b64:
                payload["content"] = encrypt_if_necessary(payload.get("content"), key_b64=key_b64, aad={"id": payload.get("id")})
                payload["summary"] = encrypt_if_necessary(payload.get("summary"), key_b64=key_b64, aad={"id": payload.get("id")})

            stmt = pg_insert(Memories).values(**payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Memories.id],
                set_={field: getattr(stmt.excluded, field) for field in payload.keys() if field != "id"},
            )
            session.execute(stmt)
            session.commit()
        return 1

    def get_mem(self, mid: str) -> Dict[str, Any] | None:
        session_factory = self._session_factory()
        with session_factory() as session:
            mem = session.get(Memories, mid)
            if not mem:
                return None
            row = self._model_to_dict(mem)
            key_cache = load_user_encryption_keys(session, [str(row.get("user_id"))] if row.get("user_id") else [])
            return self._maybe_decrypt_memory_row(row, key_cache)

    def all_mem(self, limit=10, offset=0) -> List[Dict[str, Any]]:
        query = select(Memories).order_by(desc(Memories.created_at)).limit(limit).offset(offset)
        session_factory = self._session_factory()
        with session_factory() as session:
            rows = session.execute(query).scalars().all()
            result = [self._model_to_dict(r) for r in rows]
            user_ids = [str(r.get("user_id")) for r in result if r.get("user_id")]
            key_cache = load_user_encryption_keys(session, list(set(user_ids)))
            return [self._maybe_decrypt_memory_row(r, key_cache) for r in result]

    def ins_log(self, _id: str, user_id: str, mem_id: str, model: str, status: str, ts: int,
                err: Optional[str] = None) -> int:
        session_factory = self._session_factory()
        with session_factory() as session:
            session.add(EmbedLogs(id=_id, user_id=user_id, memory_id=mem_id, model=model, status=status, ts=ts, err=err))  # type: ignore[arg-type]
            session.commit()
        return 1

    def upd_log(self, _id: str, status: str, err: Optional[str] = None) -> int:
        stmt = update(EmbedLogs).where(EmbedLogs.id == _id).values(status=status, err=err)
        session_factory = self._session_factory()
        with session_factory() as session:
            ret = session.execute(stmt)
            session.commit()
            return cast(int, getattr(ret, "rowcount", 0) or 0)

    @timing
    def find_mem_by_ids(self, mids: list[str]) -> List[Dict[str, Any]]:
        if not mids:
            return []
        query = select(Memories).where(Memories.id.in_(mids))
        session_factory = self._session_factory()
        with session_factory() as session:
            rows = session.execute(query).scalars().all()
            result = [self._model_to_dict(r) for r in rows]
            user_ids = [str(r.get("user_id")) for r in result if r.get("user_id")]
            key_cache = load_user_encryption_keys(session, list(set(user_ids)))
            return [self._maybe_decrypt_memory_row(r, key_cache) for r in result]

    @timing
    def find_mem_by_user(self, user: IMemoryUser, order_by: List[str], limit=10, offset=0) -> List[Dict[str, Any]]:
        query = (
            select(Memories, Vectors.v, Vectors.sector, Vectors.dim)
            .join(Vectors, Memories.id == Vectors.id, isouter=True)
            .where(and_(Memories.user_id == user.id, Vectors.v.is_not(None)))
        )

        order_exprs = []
        for item in order_by or []:
            parsed = self._parse_order_item(item)
            if parsed is not None:
                order_exprs.append(parsed)
        if order_exprs:
            query = query.order_by(*order_exprs)

        query = query.limit(limit).offset(offset)
        session_factory = self._session_factory()
        with session_factory() as session:
            rows = session.execute(query).all()
            result = []
            user_ids = [str(mem.user_id) for mem, _, _, _ in rows if getattr(mem, "user_id", None)]
            user_encryption_key = load_user_encryption_keys(session, list(set(user_ids)))
            for mem, vec, sector, dim in rows:
                item = self._model_to_dict(mem)
                item = self._maybe_decrypt_memory_row(item, user_encryption_key)
                # Keep backward-compatible payload shape for callers that json.loads(user_memory["v"]).
                item["v"] = json.dumps(vec) if isinstance(vec, (list, tuple)) else vec
                item["sector"] = sector
                item["dim"] = dim
                result.append(item)
            return result

    def find_mem_by_conditions(self, *, conditions: list[str], order_by: List[str] | None = None,
                               params: list[Any] | None = None,
                               limit: int | None = None, offset: int = 0) -> List[Dict[str, Any]]:
        if not conditions:
            return []
        query = select(Memories)
        filters = self._build_filters(conditions, params, default_model=Memories)
        if filters:
            query = query.where(and_(*filters))

        order_exprs = []
        for item in order_by or []:
            parsed = self._parse_order_item(item)
            if parsed is not None:
                order_exprs.append(parsed)
        if order_exprs:
            query = query.order_by(*order_exprs)

        if limit is not None:
            query = query.limit(limit).offset(offset)

        session_factory = self._session_factory()
        with session_factory() as session:
            rows = session.execute(query).scalars().all()
            result = [self._model_to_dict(r) for r in rows]
            user_ids = [str(r.get("user_id")) for r in result if r.get("user_id")]
            key_cache = load_user_encryption_keys(session, list(set(user_ids)))
            return [self._maybe_decrypt_memory_row(r, key_cache) for r in result]

    def all_mem_by_user(self,
                        user: IMemoryUser,
                        limit=10,
                        offset=0,
                        sort_order: Literal["asc", "desc"] = "desc") -> List[Dict[str, Any]]:
        sort_order = sort_order.lower()
        order_expr = asc(Memories.created_at) if sort_order == "asc" else desc(Memories.created_at)
        query = select(Memories).where(Memories.user_id == user.id).order_by(order_expr).limit(limit).offset(offset)
        session_factory = self._session_factory()
        with session_factory() as session:
            rows = session.execute(query).scalars().all()
            result = [self._model_to_dict(r) for r in rows]
            key_cache = load_user_encryption_keys(session, [str(user.id)] if user.id else [])
            return [self._maybe_decrypt_memory_row(r, key_cache) for r in result]

    def count_mem_by_user(self, user: IMemoryUser, conditions: list[str] | None = None) -> int:
        query = select(func.count(Memories.id)).where(Memories.user_id == user.id)
        if conditions:
            filters = self._build_filters(conditions, [], default_model=Memories)
            if filters:
                query = query.where(and_(*filters))

        session_factory = self._session_factory()
        with session_factory() as session:
            cnt = session.execute(query).scalar_one()
            return int(cnt or 0)

    def get_waypoints_by_src(self, src_id: str) -> List[Dict[str, Any]]:
        query = select(Waypoints).where(Waypoints.src_id == src_id)
        session_factory = self._session_factory()
        with session_factory() as session:
            rows = session.execute(query).scalars().all()
            return [self._model_to_dict(r) for r in rows]

    def del_mem(self, mem_id: str) -> int:
        session_factory = self._session_factory()
        with session_factory() as session:
            session.execute(delete(Vectors).where(Vectors.id == mem_id))
            session.execute(delete(Waypoints).where((Waypoints.src_id == mem_id) | (Waypoints.dst_id == mem_id)))
            session.execute(delete(EmbedLogs).where(EmbedLogs.memory_id == mem_id))
            ret = session.execute(delete(Memories).where(Memories.id == mem_id))
            session.commit()
            return cast(int, getattr(ret, "rowcount", 0) or 0)

    def del_mem_by_user(self, user: IMemoryUser) -> int:
        user_id = user.id
        session_factory = self._session_factory()
        with session_factory() as session:
            session.execute(delete(GraphEntityRelations).where(GraphEntityRelations.user_id == user_id))
            session.execute(delete(GraphFactEntities).where(GraphFactEntities.user_id == user_id))
            session.execute(delete(GraphEntities).where(GraphEntities.user_id == user_id))
            session.execute(delete(GraphCanonicalEntities).where(GraphCanonicalEntities.user_id == user_id))
            session.execute(delete(GraphFacts).where(GraphFacts.user_id == user_id))
            session.execute(delete(GraphTopics).where(GraphTopics.user_id == user_id))

            session.execute(delete(Sessions).where(Sessions.user_id == user_id))
            session.execute(delete(UserProfiles).where(UserProfiles.user_id == user_id))

            session.execute(delete(Vectors).where(Vectors.user_id == user_id))
            session.execute(delete(Waypoints).where(Waypoints.user_id == user_id))
            session.execute(delete(EmbedLogs).where(EmbedLogs.user_id == user_id))

            ret = session.execute(delete(Memories).where(Memories.user_id == user_id))
            session.commit()
            return cast(int, getattr(ret, "rowcount", 0) or 0)


mem_ops = MemOps()
