import datetime

import pytest

from services.memory.hsg import _promote_qa_assistant_answer, _resolve_auto_qa_linking
from domain.memory.models import IMemoryItemInfo, IMemoryUser


class _FakeScalarResult:
    def __init__(self, values):
        self._values = list(values)

    def all(self):
        return list(self._values)


class _FakeExecuteResult:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return _FakeScalarResult(self._values)


class _FakeSession:
    def __init__(self, execute_results):
        self._execute_results = list(execute_results)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, _query):
        return _FakeExecuteResult(self._execute_results.pop(0))


def test_resolve_auto_qa_linking_reuses_latest_unpaired_human(monkeypatch):
    import services.memory.hsg as hsg

    fake_user = IMemoryUser(id="u1")
    fake_session = _FakeSession([
        ["pair-older"],
        ["pair-latest", "pair-older"],
    ])
    monkeypatch.setattr(hsg, "session_factory", lambda: fake_session)

    pair_id = _resolve_auto_qa_linking(fake_user, "assistant")

    assert pair_id == "pair-latest"


def test_resolve_auto_qa_linking_generates_new_pair_when_no_candidate(monkeypatch):
    import services.memory.hsg as hsg

    fake_user = IMemoryUser(id="u1")
    fake_session = _FakeSession([
        ["pair-1"],
        ["pair-1"],
    ])
    monkeypatch.setattr(hsg, "session_factory", lambda: fake_session)
    monkeypatch.setattr(hsg.uuid, "uuid4", lambda: "generated-pair-id")

    pair_id = _resolve_auto_qa_linking(fake_user, "assistant")

    assert pair_id == "generated-pair-id"


def test_promote_qa_assistant_answer_appends_missing_assistant(monkeypatch):
    import services.memory.hsg as hsg

    now = datetime.datetime.now()
    human = IMemoryItemInfo(
        id="h1",
        content="human question",
        score=0.7,
        primary_sector="semantic",
        path=["h1"],
        salience=0.3,
        last_seen_at=now,
        created_at=now,
        tags=["q"],
        role="human",
        pair_id="pair-123",
        metadata={},
    )
    monkeypatch.setattr(
        hsg,
        "_find_latest_assistant_by_pair_id",
        lambda _pair_id: {
            "id": "a1",
            "content": "assistant answer",
            "primary_sector": "semantic",
            "salience": 0.5,
            "last_seen_at": now,
            "tags": "[\"a\"]",
            "role": "assistant",
            "pair_id": "pair-123",
            "meta": "{}",
        },
    )

    items = _promote_qa_assistant_answer([human], "semantic")

    assert len(items) == 2
    appended = next(item for item in items if item.id == "a1")
    assert appended.score == pytest.approx(0.9)
    assert appended.role == "assistant"
    assert appended.tags == ["a"]

