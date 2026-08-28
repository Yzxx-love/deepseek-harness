"""运行时层单元测试：预算、追踪、报告、安全策略。

不依赖模型、工具、记忆实现，因此可独立、快速地验证运行时约束。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agi_lab.config import Config
from agi_lab.errors import BudgetExceeded
from agi_lab.runtime.budget import BudgetTracker
from agi_lab.runtime.safety import PolicyGuard
from agi_lab.runtime.tracing import (
    Tracer,
    read_events,
    render_timeline,
    write_report,
)
from agi_lab.types import (
    Metrics,
    Observation,
    Plan,
    PlanStep,
    RunResult,
    RunStatus,
    ToolResult,
    ToolSpec,
    Usage,
)


def _config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "provider": "mock",
        "runs_dir": Path("runs"),
        "state_dir": Path("state"),
        "max_seconds": 600.0,
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


class FakeClock:
    """可推进的单调时钟。"""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BudgetTrackerTest(unittest.TestCase):
    def test_records_metrics(self) -> None:
        clock = FakeClock()
        budget = BudgetTracker(_config(), clock=clock)
        budget.record_llm(Usage(prompt_tokens=10, completion_tokens=5))
        budget.record_llm(Usage(prompt_tokens=20, completion_tokens=3))
        budget.record_tool(ok=True)
        budget.record_tool(ok=False)
        budget.record_step()
        budget.record_replan()
        budget.record_retry()
        budget.record_lesson()
        budget.record_format_repair()
        budget.record_safety_block()
        clock.advance(2.5)
        metrics = budget.finish()
        self.assertEqual(metrics.llm_calls, 2)
        self.assertEqual(metrics.prompt_tokens, 30)
        self.assertEqual(metrics.completion_tokens, 8)
        self.assertEqual(metrics.total_tokens, 38)
        self.assertEqual(metrics.tool_calls, 2)
        self.assertEqual(metrics.tool_failures, 1)
        self.assertEqual(metrics.steps, 1)
        self.assertEqual(metrics.replans, 1)
        self.assertEqual(metrics.retries, 1)
        self.assertEqual(metrics.lessons, 1)
        self.assertEqual(metrics.format_repairs, 1)
        self.assertEqual(metrics.safety_blocks, 1)
        self.assertAlmostEqual(metrics.seconds, 2.5)

    def test_step_limit(self) -> None:
        budget = BudgetTracker(_config(max_steps=2))
        budget.record_step()
        budget.check()
        budget.record_step()
        with self.assertRaises(BudgetExceeded) as caught:
            budget.check()
        self.assertEqual(caught.exception.limit_name, "max_steps")

    def test_llm_call_limit(self) -> None:
        budget = BudgetTracker(_config(max_llm_calls=1))
        budget.record_llm(Usage())
        with self.assertRaises(BudgetExceeded) as caught:
            budget.check()
        self.assertEqual(caught.exception.limit_name, "max_llm_calls")

    def test_token_limit(self) -> None:
        budget = BudgetTracker(_config(max_tokens=100))
        budget.record_llm(Usage(prompt_tokens=60, completion_tokens=60))
        with self.assertRaises(BudgetExceeded) as caught:
            budget.check()
        self.assertEqual(caught.exception.limit_name, "max_tokens")

    def test_wall_clock_limit(self) -> None:
        clock = FakeClock()
        budget = BudgetTracker(_config(max_seconds=10.0), clock=clock)
        clock.advance(11.0)
        with self.assertRaises(BudgetExceeded) as caught:
            budget.check()
        self.assertEqual(caught.exception.limit_name, "max_seconds")

    def test_budget_error_carries_numbers(self) -> None:
        budget = BudgetTracker(_config(max_steps=1))
        budget.record_step()
        with self.assertRaises(BudgetExceeded) as caught:
            budget.check()
        self.assertIn("max_steps", str(caught.exception))
        self.assertEqual(caught.exception.limit, 1.0)


class TracerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run-1"

    def test_writes_jsonl_and_reads_back(self) -> None:
        tracer = Tracer(self.run_dir, run_id="run-1")
        tracer.event("run_start", task={"id": "t1"})
        tracer.event("observation", observation=Observation(source="calculator", content="978"))
        tracer.close()
        events = read_events(tracer.path)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["run_id"], "run-1")
        self.assertEqual(events[0]["kind"], "run_start")
        self.assertEqual(events[1]["payload"]["observation"]["source"], "calculator")
        self.assertIn("ts", events[0])

    def test_events_serializes_plan(self) -> None:
        tracer = Tracer(self.run_dir, run_id="run-1")
        plan = Plan(goal="g", steps=[PlanStep(id="s1", description="a")])
        tracer.event("plan", plan=plan)
        tracer.close()
        payload = read_events(tracer.path)[0]["payload"]["plan"]
        self.assertEqual(payload["steps"][0]["id"], "s1")
        self.assertEqual(payload["steps"][0]["status"], "pending")

    def test_bad_line_skipped(self) -> None:
        tracer = Tracer(self.run_dir, run_id="run-1")
        tracer.event("run_start")
        tracer.close()
        with tracer.path.open("a", encoding="utf-8") as handle:
            handle.write("{坏行\n")
        tracer2 = Tracer(self.run_dir, run_id="run-1")
        tracer2.event("run_end")
        tracer2.close()
        kinds = [event["kind"] for event in read_events(tracer.path)]
        self.assertEqual(kinds, ["run_start", "run_end"])

    def test_render_timeline(self) -> None:
        tracer = Tracer(self.run_dir, run_id="run-1")
        tracer.event("plan", version=1)
        tracer.event("run_end", status="success")
        tracer.close()
        text = render_timeline(tracer.path)
        self.assertIn("plan", text)
        self.assertIn("run_end", text)
        self.assertIn("共 2 个事件", text)

    def test_missing_trace_renders_note(self) -> None:
        self.assertIn("没有事件", render_timeline(self.run_dir / "nope.jsonl"))


class ReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run-1"

    def _result(self) -> RunResult:
        metrics = Metrics(steps=2, llm_calls=8, prompt_tokens=900, completion_tokens=120, seconds=1.5)
        return RunResult(
            run_id="run-1",
            task_id="demo",
            status=RunStatus.SUCCESS,
            success=False,
            answer="答案是 978",
            metrics=metrics,
            trace_path=str(self.run_dir / "trace.jsonl"),
            plan=Plan(goal="g", steps=[PlanStep(id="s1", description="a")]),
        )

    def test_report_contains_sections(self) -> None:
        tracer = Tracer(self.run_dir, run_id="run-1")
        tracer.event("safety_event", tool="read_file", error="路径越界")
        tracer.close()
        path = write_report(self.run_dir, self._result())
        text = path.read_text(encoding="utf-8")
        for section in ("run 报告", "## 指标", "## 最终答案", "## 计划终态", "## 安全事件", "## 时间线"):
            self.assertIn(section, text)
        self.assertIn("978", text)
        self.assertIn("路径越界", text)
        self.assertIn("| steps | 2 |", text)

    def test_report_without_plan(self) -> None:
        result = self._result()
        result.plan = None
        path = write_report(self.run_dir, result)
        self.assertNotIn("## 计划终态", path.read_text(encoding="utf-8"))

    def test_report_written_even_with_no_events(self) -> None:
        path = write_report(self.run_dir, self._result(), events=[])
        self.assertTrue(path.exists())


class PolicyGuardTest(unittest.TestCase):
    def test_redacts_api_key(self) -> None:
        guard = PolicyGuard(_config(api_key="sk-topsecret-1234567890"))
        self.assertNotIn("sk-topsecret-1234567890", guard.redact("leak sk-topsecret-1234567890 here"))
        self.assertIn("***", guard.redact("leak sk-topsecret-1234567890 here"))

    def test_redacts_sk_pattern_without_key(self) -> None:
        guard = PolicyGuard(_config())
        self.assertNotIn("sk-abcdefghijklmnop", guard.redact("token sk-abcdefghijklmnop"))
        self.assertEqual(guard.redact(""), "")

    def test_approval_none_for_deny_and_allow(self) -> None:
        self.assertIsNone(PolicyGuard(_config(approval_mode="deny")).approval())
        self.assertIsNone(PolicyGuard(_config(approval_mode="allow")).approval())

    def test_prompt_non_interactive_denies(self) -> None:
        guard = PolicyGuard(_config(approval_mode="prompt"), interactive=False)
        callback = guard.approval()
        self.assertIsNotNone(callback)
        spec = ToolSpec(name="delete_file", description="删除", requires_approval=True)
        self.assertFalse(callback(spec, {"path": "a"}))

    def test_prompt_accepts_yes(self) -> None:
        answers = iter(["y"])
        guard = PolicyGuard(_config(approval_mode="prompt"), input_fn=lambda _: next(answers), interactive=True)
        callback = guard.approval()
        spec = ToolSpec(name="delete_file", description="删除", requires_approval=True)
        self.assertTrue(callback(spec, {"path": "a"}))

    def test_prompt_rejects_no(self) -> None:
        guard = PolicyGuard(_config(approval_mode="prompt"), input_fn=lambda _: "n", interactive=True)
        callback = guard.approval()
        spec = ToolSpec(name="delete_file", description="删除", requires_approval=True)
        self.assertFalse(callback(spec, {"path": "a"}))

    def test_is_safety_block(self) -> None:
        self.assertTrue(PolicyGuard.is_safety_block(ToolResult(ok=False, error="拒绝", metadata={"safety": True})))
        self.assertFalse(PolicyGuard.is_safety_block(ToolResult(ok=False, error="普通失败")))


if __name__ == "__main__":
    unittest.main()
