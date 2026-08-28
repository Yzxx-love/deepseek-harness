"""端到端集成测试：Agent + mock provider + 真实工具/记忆实现。

每个用例用临时目录，不联网、不写仓库。覆盖成功路径、失败恢复、安全拦截、
预算终止、诚实失败、ASK 阻塞与人工应答、fixture 越界。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agi_lab.config import Config
from agi_lab.demo import demo_tasks, find_demo_task
from agi_lab.llm.mock import MockLLM
from agi_lab.memory.hub import MemoryHub
from agi_lab.runtime.agent import Agent
from agi_lab.runtime.tracing import read_events
from agi_lab.tools import build_default_tools
from agi_lab.types import RunStatus, TaskSpec


class AgentIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = Config(
            provider="mock",
            runs_dir=self.root / "runs",
            state_dir=self.root / "state",
            max_seconds=60.0,
            max_steps=12,
        ).resolve()

    def _run(
        self,
        task: TaskSpec,
        *,
        config: Config | None = None,
        memory: MemoryHub | None = None,
        answer_fn: object = None,
        label: str = "run",
    ):
        cfg = config or self.config
        workspace = self.root / label / "workspace"
        memory = memory or MemoryHub(root=None, persist=False)
        tools = build_default_tools(workspace=workspace, memory=memory)
        agent = Agent(
            config=cfg,
            llm=MockLLM(task.mock_script),
            tools=tools,
            memory=memory,
            workspace=workspace,
            run_dir=self.root / label,
            answer_fn=answer_fn,  # type: ignore[arg-type]
        )
        return agent.run(task), memory, workspace

    def _task(self, task_id: str) -> TaskSpec:
        task = find_demo_task(task_id)
        self.assertIsNotNone(task)
        return task  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 成功路径
    # ------------------------------------------------------------------

    def test_arithmetic_success_and_artifacts(self) -> None:
        result, _, _ = self._run(self._task("arithmetic"))
        self.assertIs(result.status, RunStatus.SUCCESS)
        self.assertIn("978", result.answer)
        self.assertEqual(result.metrics.tool_calls, 1)
        self.assertEqual(result.metrics.llm_calls, 5)
        trace_path = Path(result.trace_path)
        self.assertTrue(trace_path.exists())
        kinds = [event["kind"] for event in read_events(trace_path)]
        for expected in ("run_start", "plan", "step_start", "action", "observation", "critic_step", "critic_final", "run_end"):
            self.assertIn(expected, kinds)
        self.assertTrue(Path(result.report_path).exists())

    def test_success_flag_is_not_self_assessed(self) -> None:
        result, _, _ = self._run(self._task("arithmetic"))
        self.assertIs(result.status, RunStatus.SUCCESS)
        self.assertFalse(result.success)  # 通过与否由评测层独立 grader 决定

    def test_file_report_writes_artifact(self) -> None:
        result, _, workspace = self._run(self._task("file-report"))
        self.assertIs(result.status, RunStatus.SUCCESS)
        report = workspace / "report.md"
        self.assertTrue(report.exists())
        self.assertIn("70", report.read_text(encoding="utf-8"))
        self.assertEqual(result.metrics.steps, 2)

    def test_self_heal_exercises_reflection_and_persistence(self) -> None:
        memory_root = self.root / "memory"
        memory = MemoryHub(root=memory_root, persist=True)
        result, memory, _ = self._run(self._task("self-heal"), memory=memory)
        self.assertIs(result.status, RunStatus.SUCCESS)
        self.assertEqual(result.metrics.retries, 1)
        self.assertEqual(result.metrics.replans, 1)
        self.assertEqual(result.metrics.lessons, 1)
        self.assertEqual(len(memory.reflection), 1)
        kinds = [event["kind"] for event in read_events(Path(result.trace_path))]
        self.assertIn("reflection", kinds)
        self.assertIn("plan_update", kinds)
        self.assertIn("retry", kinds)
        # 持久化：新实例从同一目录能读回教训
        reloaded = MemoryHub(root=memory_root, persist=True)
        self.assertGreaterEqual(len(reloaded.reflection), 1)

    def test_arithmetic_tool_failure_recovered_within_step(self) -> None:
        goal = "调用一个不存在的工具，然后改用 calculator 计算 6*7。"
        script = {
            "planner": [{"goal": goal, "steps": [{"id": "s1", "description": "计算 6*7", "success_criteria": "得到 42"}]}],
            "actor": [
                {"action": {"type": "tool", "tool": "nonexistent_tool", "arguments": {}}},
                {"action": {"type": "tool", "tool": "calculator", "arguments": {"expression": "6*7"}}},
                {"action": {"type": "final", "answer": "6*7 = 42"}},
            ],
            "critic_step": [{"ok": True, "missing": [], "reason": "观测含 42"}],
            "critic_final": [{"complete": True, "confidence": 0.9, "answer": "6*7 = 42", "missing": []}],
        }
        task = TaskSpec(id="tool-error", goal=goal, mock_script=script)
        result, _, _ = self._run(task, label="tool-error")
        self.assertIs(result.status, RunStatus.SUCCESS)
        self.assertEqual(result.metrics.tool_failures, 1)
        self.assertEqual(result.metrics.tool_calls, 2)

    # ------------------------------------------------------------------
    # 安全与越界
    # ------------------------------------------------------------------

    def test_safety_block_recorded(self) -> None:
        result, _, workspace = self._run(self._task("safety"))
        self.assertIs(result.status, RunStatus.SUCCESS)
        self.assertEqual(result.metrics.safety_blocks, 1)
        kinds = [event["kind"] for event in read_events(Path(result.trace_path))]
        self.assertIn("safety_event", kinds)
        self.assertFalse((workspace.parent / "secret.txt").exists())

    def test_fixture_path_escape_rejected(self) -> None:
        task = TaskSpec(id="escape", goal="越界", initial_files={"../evil.txt": "x"})
        result, _, _ = self._run(task, label="escape")
        self.assertIs(result.status, RunStatus.ERROR)
        self.assertIn("SafetyViolation", result.error or "")

    # ------------------------------------------------------------------
    # 预算与终止语义
    # ------------------------------------------------------------------

    def test_budget_exhausted(self) -> None:
        config = Config(
            provider="mock",
            runs_dir=self.root / "runs",
            state_dir=self.root / "state",
            max_llm_calls=2,
            max_seconds=60.0,
        ).resolve()
        result, _, _ = self._run(self._task("arithmetic"), config=config, label="budget")
        self.assertIs(result.status, RunStatus.BUDGET_EXHAUSTED)
        self.assertIn("max_llm_calls", result.stop_reason)
        kinds = [event["kind"] for event in read_events(Path(result.trace_path))]
        self.assertIn("budget_exhausted", kinds)

    def test_max_steps_stops_run(self) -> None:
        config = Config(
            provider="mock",
            runs_dir=self.root / "runs",
            state_dir=self.root / "state",
            max_steps=1,
            max_seconds=60.0,
        ).resolve()
        result, _, _ = self._run(self._task("file-report"), config=config, label="max-steps")
        self.assertIs(result.status, RunStatus.BUDGET_EXHAUSTED)
        self.assertIn("max_steps", result.stop_reason)

    def test_default_mock_fails_honestly(self) -> None:
        task = TaskSpec(id="no-script", goal="随便做点什么")
        result, _, _ = self._run(task, label="no-script")
        self.assertIs(result.status, RunStatus.FAILED)
        self.assertIn("终审未通过", result.stop_reason)

    # ------------------------------------------------------------------
    # 提问与循环检测
    # ------------------------------------------------------------------

    def test_ask_without_human_blocks(self) -> None:
        goal = "需要人类澄清的任务"
        script = {
            "planner": [{"goal": goal, "steps": [{"id": "s1", "description": "提问"}]}],
            "actor": [{"action": {"type": "ask", "question": "用哪个文件？"}}],
        }
        task = TaskSpec(id="ask-blocked", goal=goal, mock_script=script)
        result, _, _ = self._run(task, label="ask-blocked")
        self.assertIs(result.status, RunStatus.BLOCKED)
        self.assertIn("无人应答", result.stop_reason)

    def test_ask_with_human_continues(self) -> None:
        goal = "需要人类澄清的任务"
        script = {
            "planner": [{"goal": goal, "steps": [{"id": "s1", "description": "读取 notes.txt"}]}],
            "actor": [
                {"action": {"type": "ask", "question": "用哪个文件？"}},
                {"action": {"type": "tool", "tool": "read_file", "arguments": {"path": "notes.txt"}}},
                {"action": {"type": "final", "answer": "要点：运行测试"}},
            ],
            "critic_step": [{"ok": True, "missing": [], "reason": "读到内容"}],
            "critic_final": [{"complete": True, "confidence": 0.9, "answer": "要点：运行测试", "missing": []}],
        }
        task = TaskSpec(
            id="ask-answered",
            goal=goal,
            initial_files={"notes.txt": "运行测试\n"},
            mock_script=script,
        )
        result, memory, _ = self._run(task, answer_fn=lambda question: "用 notes.txt", label="ask-answered")
        self.assertIs(result.status, RunStatus.SUCCESS)
        kinds = [event["kind"] for event in read_events(Path(result.trace_path))]
        self.assertIn("human_answer", kinds)
        self.assertEqual(len(memory.semantic), 1)

    def test_loop_detection(self) -> None:
        goal = "陷入重复调用"
        script = {
            "planner": [{"goal": goal, "steps": [{"id": "s1", "description": "计算"}]}],
            "actor": [
                {"action": {"type": "tool", "tool": "calculator", "arguments": {"expression": "1+1"}}},
                {"action": {"type": "tool", "tool": "calculator", "arguments": {"expression": "1+1"}}},
                {"action": {"type": "tool", "tool": "calculator", "arguments": {"expression": "1+1"}}},
                {"action": {"type": "tool", "tool": "calculator", "arguments": {"expression": "1+1"}}},
            ],
            "critic_step": [{"ok": False, "missing": ["没有完成"], "reason": "重复调用"}],
            "reflector": [{"failure_type": "wrong_plan", "diagnosis": "重复调用", "lesson": "重复调用时改变策略", "plan_patch": {"remove": [], "add": []}}],
            "critic_final": [{"complete": False, "confidence": 0.0, "answer": "", "missing": ["未完成"]}],
        }
        config = Config(
            provider="mock",
            runs_dir=self.root / "runs",
            state_dir=self.root / "state",
            max_attempts=1,
            max_replans=0,
            max_seconds=60.0,
        ).resolve()
        task = TaskSpec(id="loop", goal=goal, mock_script=script)
        result, _, _ = self._run(task, config=config, label="loop")
        kinds = [event["kind"] for event in read_events(Path(result.trace_path))]
        self.assertIn("loop_detected", kinds)
        self.assertIs(result.status, RunStatus.FAILED)

    # ------------------------------------------------------------------
    # 报告与演示集合
    # ------------------------------------------------------------------

    def test_all_demo_tasks_meet_expectations(self) -> None:
        for task in demo_tasks():
            with self.subTest(task=task.id):
                result, _, _ = self._run(task, label=f"demo-{task.id}")
                expected = str(task.metadata.get("expected_status", "success"))
                self.assertEqual(result.status.value, expected)
                minimum = task.metadata.get("expect_replans_min")
                if isinstance(minimum, int):
                    self.assertGreaterEqual(result.metrics.replans, minimum)
                safety_minimum = task.metadata.get("expect_safety_min")
                if isinstance(safety_minimum, int):
                    self.assertGreaterEqual(result.metrics.safety_blocks, safety_minimum)


if __name__ == "__main__":
    unittest.main()
