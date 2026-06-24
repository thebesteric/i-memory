import asyncio
from types import SimpleNamespace

from infra.scheduler import jobs
from interfaces.api.routes import backend_router


def test_scheduler_job_definitions_include_manual_trigger_targets():
    job_ids = {job.id for job in jobs.get_job_definitions()}
    assert {"memory_decay", "graph_build", "force_graph_build", "session_build", "user_profile"}.issubset(job_ids)


def test_run_job_dispatches_to_underlying_callable(monkeypatch):
    called: list[str] = []

    async def _fake_session_build():
        called.append("session_build")

    monkeypatch.setattr(jobs.session_builder, "session_build", _fake_session_build)

    result = asyncio.run(jobs.run_job("session_build"))

    assert result is None
    assert called == ["session_build"]


def test_trigger_scheduler_job_returns_success(monkeypatch):
    fake_job = SimpleNamespace(
        id="memory_decay",
        name="Periodic memory decay",
        trigger_type="interval",
        trigger_args={"seconds": 3600},
        enable=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    triggered: list[str] = []

    async def _fake_run_job(job_id: str):
        triggered.append(job_id)

    monkeypatch.setattr(backend_router, "get_job_definitions", lambda: [fake_job])
    monkeypatch.setattr(backend_router, "run_job", _fake_run_job)

    response = asyncio.run(backend_router.trigger_scheduler_job("memory_decay"))
    payload = response.model_dump()

    assert payload["code"] == 200
    assert payload["message"] == "已触发任务: Periodic memory decay"
    assert payload["data"] == {"job_id": "memory_decay", "job_name": "Periodic memory decay"}
    assert triggered == ["memory_decay"]


def test_trigger_scheduler_job_returns_404_for_unknown_job(monkeypatch):
    monkeypatch.setattr(backend_router, "get_job_definitions", lambda: [])

    response = asyncio.run(backend_router.trigger_scheduler_job("unknown_job"))
    payload = response.model_dump()

    assert payload["code"] == 404
    assert payload["message"] == "任务不存在: unknown_job"

