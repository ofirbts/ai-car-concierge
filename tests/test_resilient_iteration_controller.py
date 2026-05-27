from pathlib import Path

from backend.resilient_iteration_controller import ResilientIterationController


def test_resilient_iteration_replay_skips_completed_steps(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    controller = ResilientIterationController(journal)
    calls: list[str] = []

    def step_a(state):
        calls.append("a")
        return {"a": 1}

    def step_b(state):
        calls.append("b")
        return {"b": state.get("a", 0) + 1}

    first = controller.run("run-1", {}, [("step_a", step_a), ("step_b", step_b)])
    assert first.last_event == "RUN_COMPLETED"
    assert first.state["a"] == 1
    assert first.state["b"] == 2
    assert calls == ["a", "b"]

    calls.clear()
    second = controller.run("run-1", {}, [("step_a", step_a), ("step_b", step_b)])
    assert second.last_event == "RUN_COMPLETED"
    assert calls == []
    assert second.completed_steps == ["step_a", "step_b"]


def test_resilient_iteration_stops_on_failure(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    controller = ResilientIterationController(journal)

    def ok_step(_state):
        return {"ok": True}

    def failing_step(_state):
        raise RuntimeError("boom")

    result = controller.run("run-2", {}, [("ok", ok_step), ("fail", failing_step)])
    assert result.last_event == "STEP_FAILED"
    assert result.completed_steps == ["ok"]
