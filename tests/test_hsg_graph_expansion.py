import datetime
from types import SimpleNamespace

from src.memory.graph_search import GraphExpansionCandidate
from src.memory.hsg import query_hsg_memories
from src.memory.memory_models import IMemoryFilters, IMemoryFiltersConfig, IMemoryGraphConfig, IMemoryUserIdentity
from src.memory.session.session_models import Sessions


class _MemoryCache:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


async def _noop_async(*args, **kwargs):
    return None


def test_query_hsg_memories_uses_graph_candidate_scores(monkeypatch):
    import src.memory.hsg as hsg

    now = datetime.datetime.now()
    fake_cache = _MemoryCache()
    fake_user = SimpleNamespace(id="u1")

    monkeypatch.setattr(hsg, "MEMORIES_CACHE", fake_cache)
    monkeypatch.setattr(hsg.user_ops, "get_user", lambda **kwargs: _return(fake_user))
    monkeypatch.setattr(
        hsg,
        "embed_query_for_all_sectors",
        lambda _query, sectors: _return({sector: [1.0] for sector in sectors}),
    )
    monkeypatch.setattr(hsg.sector_classifier, "classify", lambda *args, **kwargs: _return(SimpleNamespace(primary="semantic")))
    monkeypatch.setattr(hsg.vector_store, "search", lambda *args, **kwargs: _return([]))
    monkeypatch.setattr(hsg.waypoints, "expand_via_waypoints", lambda *args, **kwargs: _return([]))
    captured_graph_kwargs = {}

    def _fake_expand_candidate_ids_via_graph(*args, **kwargs):
        captured_graph_kwargs.update(kwargs)
        return [GraphExpansionCandidate(id="m3", score=0.9)]

    monkeypatch.setattr(hsg.graph_search, "expand_candidate_ids_via_graph", _fake_expand_candidate_ids_via_graph)
    monkeypatch.setattr(
        hsg.mem_ops,
        "find_mem_by_ids",
        lambda *args, **kwargs: [
            {
                "id": "m3",
                "content": "graph memory",
                "salience": 0.2,
                "primary_sector": "semantic",
                "last_seen_at": now,
                "created_at": now,
                "tags": [],
                "meta": "{}",
                "user_id": "u1",
            }
        ],
    )
    monkeypatch.setattr(hsg, "calc_multi_vec_fusion_score", lambda *args, **kwargs: _return(0.4))
    monkeypatch.setattr(hsg, "calc_cross_sector_resonance_score", lambda *args, **kwargs: _return(0.4))
    monkeypatch.setattr(hsg, "compute_tag_match_score", lambda *args, **kwargs: _return(0.0))
    monkeypatch.setattr(hsg.decay, "calc_decay", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(hsg.decay, "calc_recency_score_decay", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(hsg.asyncio, "create_task", lambda coro: coro.close())
    monkeypatch.setattr(hsg, "update_user_summary", lambda *args, **kwargs: _noop_async())
    monkeypatch.setattr(hsg.user_profile_ops, "get_user_profile", lambda *args, **kwargs: _noop_async())
    monkeypatch.setattr(hsg.session_ops, "session_search", lambda *args, **kwargs: _return(Sessions()))
    monkeypatch.setattr(hsg, "compute_hybrid_score", lambda **kwargs: 0.5)

    filters = IMemoryFilters(
        user_identity=IMemoryUserIdentity(tenant_key="t1", project_key="p1", user_key="u1"),
        config=IMemoryFiltersConfig(
            bm25_enable=False,
            graph=IMemoryGraphConfig(
                enable=True,
                type="custom",
                max_hops=2,
                hop_decay=0.7,
                per_hop_limit=123,
                min_walk_score=0.08,
                min_relation_confidence=0.66,
            ),
            user_profile_enable=False,
            session_summary_enable=False,
            session_dedup_enable=False,
            debug=False,
        ),
    )

    result = _run(query_hsg_memories("test query", top_k=3, filters=filters))

    assert result.memories
    assert result.memories[0].id == "m3"
    assert result.memories[0].metadata["from"] == "graph"
    assert result.memories[0].metadata["graph_score"] == 0.9
    assert result.memories[0].metadata["graph_bonus"] > 0.0
    assert captured_graph_kwargs["max_hops"] == 2
    assert captured_graph_kwargs["hop_decay"] == 0.7
    assert captured_graph_kwargs["per_hop_limit"] == 123
    assert captured_graph_kwargs["min_walk_score"] == 0.08
    assert captured_graph_kwargs["min_relation_confidence"] == 0.66


def test_query_hsg_memories_uses_recall_graph_config(monkeypatch):
    import src.memory.hsg as hsg

    now = datetime.datetime.now()
    fake_cache = _MemoryCache()
    fake_user = SimpleNamespace(id="u1")
    recall = IMemoryGraphConfig.recall_first()

    monkeypatch.setattr(hsg, "MEMORIES_CACHE", fake_cache)
    monkeypatch.setattr(hsg.user_ops, "get_user", lambda **kwargs: _return(fake_user))
    monkeypatch.setattr(
        hsg,
        "embed_query_for_all_sectors",
        lambda _query, sectors: _return({sector: [1.0] for sector in sectors}),
    )
    monkeypatch.setattr(hsg.sector_classifier, "classify", lambda *args, **kwargs: _return(SimpleNamespace(primary="semantic")))
    monkeypatch.setattr(hsg.vector_store, "search", lambda *args, **kwargs: _return([]))
    monkeypatch.setattr(hsg.waypoints, "expand_via_waypoints", lambda *args, **kwargs: _return([]))
    captured_graph_kwargs = {}

    def _fake_expand_candidate_ids_via_graph(*args, **kwargs):
        captured_graph_kwargs.update(kwargs)
        return [GraphExpansionCandidate(id="m3", score=0.9)]

    monkeypatch.setattr(hsg.graph_search, "expand_candidate_ids_via_graph", _fake_expand_candidate_ids_via_graph)
    monkeypatch.setattr(
        hsg.mem_ops,
        "find_mem_by_ids",
        lambda *args, **kwargs: [
            {
                "id": "m3",
                "content": "graph memory",
                "salience": 0.2,
                "primary_sector": "semantic",
                "last_seen_at": now,
                "created_at": now,
                "tags": [],
                "meta": "{}",
                "user_id": "u1",
            }
        ],
    )
    monkeypatch.setattr(hsg, "calc_multi_vec_fusion_score", lambda *args, **kwargs: _return(0.4))
    monkeypatch.setattr(hsg, "calc_cross_sector_resonance_score", lambda *args, **kwargs: _return(0.4))
    monkeypatch.setattr(hsg, "compute_tag_match_score", lambda *args, **kwargs: _return(0.0))
    monkeypatch.setattr(hsg.decay, "calc_decay", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(hsg.decay, "calc_recency_score_decay", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(hsg.asyncio, "create_task", lambda coro: coro.close())
    monkeypatch.setattr(hsg, "update_user_summary", lambda *args, **kwargs: _noop_async())
    monkeypatch.setattr(hsg.user_profile_ops, "get_user_profile", lambda *args, **kwargs: _noop_async())
    monkeypatch.setattr(hsg.session_ops, "session_search", lambda *args, **kwargs: _return(Sessions()))
    monkeypatch.setattr(hsg, "compute_hybrid_score", lambda **kwargs: 0.5)

    filters = IMemoryFilters(
        user_identity=IMemoryUserIdentity(tenant_key="t1", project_key="p1", user_key="u1"),
        config=IMemoryFiltersConfig(
            bm25_enable=False,
            graph=recall,
            user_profile_enable=False,
            session_summary_enable=False,
            session_dedup_enable=False,
            debug=False,
        ),
    )

    result = _run(query_hsg_memories("test query", top_k=3, filters=filters))

    assert result.memories
    assert captured_graph_kwargs["max_hops"] == recall.max_hops
    assert captured_graph_kwargs["hop_decay"] == recall.hop_decay
    assert captured_graph_kwargs["per_hop_limit"] == recall.per_hop_limit
    assert captured_graph_kwargs["min_walk_score"] == recall.min_walk_score
    assert captured_graph_kwargs["min_relation_confidence"] == recall.min_relation_confidence


def test_query_hsg_memories_uses_precision_default_graph_config(monkeypatch):
    import src.memory.hsg as hsg

    now = datetime.datetime.now()
    fake_cache = _MemoryCache()
    fake_user = SimpleNamespace(id="u1")
    precision = IMemoryGraphConfig.precision_first()

    monkeypatch.setattr(hsg, "MEMORIES_CACHE", fake_cache)
    monkeypatch.setattr(hsg.user_ops, "get_user", lambda **kwargs: _return(fake_user))
    monkeypatch.setattr(
        hsg,
        "embed_query_for_all_sectors",
        lambda _query, sectors: _return({sector: [1.0] for sector in sectors}),
    )
    monkeypatch.setattr(hsg.sector_classifier, "classify", lambda *args, **kwargs: _return(SimpleNamespace(primary="semantic")))
    monkeypatch.setattr(hsg.vector_store, "search", lambda *args, **kwargs: _return([]))
    monkeypatch.setattr(hsg.waypoints, "expand_via_waypoints", lambda *args, **kwargs: _return([]))
    captured_graph_kwargs = {}

    def _fake_expand_candidate_ids_via_graph(*args, **kwargs):
        captured_graph_kwargs.update(kwargs)
        return [GraphExpansionCandidate(id="m3", score=0.9)]

    monkeypatch.setattr(hsg.graph_search, "expand_candidate_ids_via_graph", _fake_expand_candidate_ids_via_graph)
    monkeypatch.setattr(
        hsg.mem_ops,
        "find_mem_by_ids",
        lambda *args, **kwargs: [
            {
                "id": "m3",
                "content": "graph memory",
                "salience": 0.2,
                "primary_sector": "semantic",
                "last_seen_at": now,
                "created_at": now,
                "tags": [],
                "meta": "{}",
                "user_id": "u1",
            }
        ],
    )
    monkeypatch.setattr(hsg, "calc_multi_vec_fusion_score", lambda *args, **kwargs: _return(0.4))
    monkeypatch.setattr(hsg, "calc_cross_sector_resonance_score", lambda *args, **kwargs: _return(0.4))
    monkeypatch.setattr(hsg, "compute_tag_match_score", lambda *args, **kwargs: _return(0.0))
    monkeypatch.setattr(hsg.decay, "calc_decay", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(hsg.decay, "calc_recency_score_decay", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(hsg.asyncio, "create_task", lambda coro: coro.close())
    monkeypatch.setattr(hsg, "update_user_summary", lambda *args, **kwargs: _noop_async())
    monkeypatch.setattr(hsg.user_profile_ops, "get_user_profile", lambda *args, **kwargs: _noop_async())
    monkeypatch.setattr(hsg.session_ops, "session_search", lambda *args, **kwargs: _return(Sessions()))
    monkeypatch.setattr(hsg, "compute_hybrid_score", lambda **kwargs: 0.5)

    filters = IMemoryFilters(
        user_identity=IMemoryUserIdentity(tenant_key="t1", project_key="p1", user_key="u1"),
        config=IMemoryFiltersConfig(
            bm25_enable=False,
            user_profile_enable=False,
            session_summary_enable=False,
            session_dedup_enable=False,
            debug=False,
        ),
    )

    result = _run(query_hsg_memories("test query", top_k=3, filters=filters))

    assert result.memories
    assert captured_graph_kwargs["max_hops"] == precision.max_hops
    assert captured_graph_kwargs["hop_decay"] == precision.hop_decay
    assert captured_graph_kwargs["per_hop_limit"] == precision.per_hop_limit
    assert captured_graph_kwargs["min_walk_score"] == precision.min_walk_score
    assert captured_graph_kwargs["min_relation_confidence"] == precision.min_relation_confidence


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner()

