from services.graph import graph_search


class _FakeLogger:
    def __init__(self):
        self.info_msgs = []
        self.debug_msgs = []
        self.warning_msgs = []

    def info(self, msg):
        self.info_msgs.append(msg)

    def debug(self, msg):
        self.debug_msgs.append(msg)

    def warning(self, msg):
        self.warning_msgs.append(msg)


def test_expand_candidate_ids_via_graph_happy_path_with_deterministic_ordering(monkeypatch):
    fake_logger = _FakeLogger()
    monkeypatch.setattr(graph_search, "logger", fake_logger)
    monkeypatch.setattr(graph_search, "_find_seed_topic_ids", lambda *args, **kwargs: ["t1"])
    monkeypatch.setattr(graph_search, "_find_seed_fact_ids", lambda *args, **kwargs: ["f1"])
    monkeypatch.setattr(graph_search, "_find_seed_canonical_ids", lambda *args, **kwargs: ["c1"])
    monkeypatch.setattr(
        graph_search,
        "_walk_related_canonical_scores",
        lambda *args, **kwargs: {"c2": 0.82, "c3": 0.61},
    )
    monkeypatch.setattr(
        graph_search,
        "_fetch_related_fact_scores",
        lambda *args, **kwargs: {"f2": 0.82, "f3": 0.61},
    )
    monkeypatch.setattr(
        graph_search,
        "_fetch_related_topic_scores",
        lambda *args, **kwargs: {"t2": 0.82, "t3": 0.61},
    )
    monkeypatch.setattr(
        graph_search,
        "_collect_related_memory_scores",
        lambda *args, **kwargs: {"m3": 0.82, "m4": 0.61, "m1": 0.95},
    )

    result = graph_search.expand_candidate_ids_via_graph(
        user_id="u1",
        seed_memory_ids={"m1", "m2"},
        limit=10,
        min_relation_confidence=0.6,
    )

    assert [c.id for c in result] == ["m3", "m4"]
    assert result[0].score >= result[1].score
    assert any("Expanded memory candidates by graph" in msg for msg in fake_logger.info_msgs)


def test_expand_candidate_ids_via_graph_clamps_invalid_confidence(monkeypatch):
    monkeypatch.setattr(graph_search, "logger", _FakeLogger())
    monkeypatch.setattr(graph_search, "_find_seed_topic_ids", lambda *args, **kwargs: ["t1"])
    monkeypatch.setattr(graph_search, "_find_seed_fact_ids", lambda *args, **kwargs: ["f1"])
    monkeypatch.setattr(graph_search, "_find_seed_canonical_ids", lambda *args, **kwargs: ["c1"])
    captured = {}

    def _fake_walk(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(graph_search, "_walk_related_canonical_scores", _fake_walk)
    monkeypatch.setattr(graph_search, "_fetch_related_fact_scores", lambda *args, **kwargs: {})
    monkeypatch.setattr(graph_search, "_fetch_related_topic_scores", lambda *args, **kwargs: {})
    monkeypatch.setattr(graph_search, "_collect_related_memory_scores", lambda *args, **kwargs: {})

    result = graph_search.expand_candidate_ids_via_graph(
        user_id="u1",
        seed_memory_ids={"m1"},
        limit=5,
        min_relation_confidence=9.9,
    )

    assert captured["min_relation_confidence"] == 1.0
    assert result == []


def test_expand_candidate_ids_via_graph_clamps_invalid_max_hops(monkeypatch):
    monkeypatch.setattr(graph_search, "logger", _FakeLogger())
    monkeypatch.setattr(graph_search, "_find_seed_topic_ids", lambda *args, **kwargs: ["t1"])
    monkeypatch.setattr(graph_search, "_find_seed_fact_ids", lambda *args, **kwargs: ["f1"])
    monkeypatch.setattr(graph_search, "_find_seed_canonical_ids", lambda *args, **kwargs: ["c1"])
    captured = {}

    def _fake_walk(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(graph_search, "_walk_related_canonical_scores", _fake_walk)
    monkeypatch.setattr(graph_search, "_fetch_related_fact_scores", lambda *args, **kwargs: {})
    monkeypatch.setattr(graph_search, "_fetch_related_topic_scores", lambda *args, **kwargs: {})
    monkeypatch.setattr(graph_search, "_collect_related_memory_scores", lambda *args, **kwargs: {})

    graph_search.expand_candidate_ids_via_graph(
        user_id="u1",
        seed_memory_ids={"m1"},
        limit=5,
        max_hops=99,
    )

    assert captured["max_hops"] == 4


def test_expand_candidate_ids_via_graph_empty_user_id_short_circuit(monkeypatch):
    fake_logger = _FakeLogger()
    monkeypatch.setattr(graph_search, "logger", fake_logger)

    result = graph_search.expand_candidate_ids_via_graph(
        user_id="",
        seed_memory_ids={"m1"},
        limit=1,
    )

    assert result == []
    assert any("empty user_id" in msg for msg in fake_logger.warning_msgs)

