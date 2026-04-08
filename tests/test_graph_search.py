from src.memory import graph_search


class _FakeDB:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def fetchall(self, sql, params):
        self.calls.append((sql, params))
        if not self._responses:
            return []
        return self._responses.pop(0)


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
    fake_db = _FakeDB([
        [
            {"id": "m3", "graph_score": 0.82},
            {"id": "m4", "graph_score": 0.61},
        ],
    ])
    fake_logger = _FakeLogger()
    monkeypatch.setattr(graph_search, "db", fake_db)
    monkeypatch.setattr(graph_search, "logger", fake_logger)

    result = graph_search.expand_candidate_ids_via_graph(
        user_id="u1",
        seed_memory_ids={"m1", "m2"},
        limit=10,
        min_relation_confidence=0.6,
    )

    assert [c.id for c in result] == ["m3", "m4"]
    assert result[0].score >= result[1].score
    assert len(fake_db.calls) == 1
    assert "WITH RECURSIVE seed_memory AS" in fake_db.calls[0][0]
    assert "relation_weights AS" in fake_db.calls[0][0]
    assert "walk AS" in fake_db.calls[0][0]
    assert any("Expanded memory candidates by graph" in msg for msg in fake_logger.info_msgs)


def test_expand_candidate_ids_via_graph_clamps_invalid_confidence(monkeypatch):
    fake_db = _FakeDB([
        [],
    ])
    monkeypatch.setattr(graph_search, "db", fake_db)
    monkeypatch.setattr(graph_search, "logger", _FakeLogger())

    result = graph_search.expand_candidate_ids_via_graph(
        user_id="u1",
        seed_memory_ids={"m1"},
        limit=5,
        min_relation_confidence=9.9,
    )

    relation_query_params = fake_db.calls[0][1]
    assert relation_query_params[11] == 1.0
    assert result == []


def test_expand_candidate_ids_via_graph_clamps_invalid_max_hops(monkeypatch):
    fake_db = _FakeDB([
        [],
    ])
    monkeypatch.setattr(graph_search, "db", fake_db)
    monkeypatch.setattr(graph_search, "logger", _FakeLogger())

    graph_search.expand_candidate_ids_via_graph(
        user_id="u1",
        seed_memory_ids={"m1"},
        limit=5,
        max_hops=99,
    )

    relation_query_params = fake_db.calls[0][1]
    assert relation_query_params[10] == 4


def test_expand_candidate_ids_via_graph_empty_user_id_short_circuit(monkeypatch):
    fake_db = _FakeDB([])
    fake_logger = _FakeLogger()
    monkeypatch.setattr(graph_search, "db", fake_db)
    monkeypatch.setattr(graph_search, "logger", fake_logger)

    result = graph_search.expand_candidate_ids_via_graph(
        user_id="",
        seed_memory_ids={"m1"},
        limit=1,
    )

    assert result == []
    assert fake_db.calls == []
    assert any("empty user_id" in msg for msg in fake_logger.warning_msgs)

