"""CLI 端到端测试：真实子进程调用 ``python -m agi_lab``。

验证命令可见行为与退出码契约（0 成功 / 1 未达标 / 2 参数或配置错误），
覆盖配置脱敏、工具列表、离线演示、单任务演示、任意目标拒绝 mock、trace 回放、
机制套件评测。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _clean_env(tmp: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AGI_LAB_") and key not in {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"}
    }
    env.update(
        {
            "AGI_LAB_RUNS_DIR": str(tmp / "runs"),
            "AGI_LAB_STATE_DIR": str(tmp / "state"),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return env


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _run(self, *args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "agi_lab", *args],
            cwd=str(PROJECT_ROOT),
            env=_clean_env(self.tmp),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

    def test_version(self) -> None:
        result = self._run("--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn("agi-lab", result.stdout)

    def test_help(self) -> None:
        result = self._run("--help")
        self.assertEqual(result.returncode, 0)
        for command in ("run", "demo", "eval", "tools", "replay", "config"):
            self.assertIn(command, result.stdout)

    def test_config_redacts_key_and_shows_defaults(self) -> None:
        result = self._run("config")
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["provider"], "mock")
        self.assertIsNone(payload["api_key"])
        self.assertIn("max_steps", payload)

    def test_tools_lists_calculator_and_switches(self) -> None:
        result = self._run("tools")
        self.assertEqual(result.returncode, 0)
        self.assertIn("calculator", result.stdout)
        self.assertIn("read_file", result.stdout)
        self.assertIn("安全开关", result.stdout)
        self.assertIn("shell=off", result.stdout)
        self.assertIn("network=off", result.stdout)

    def test_tools_json(self) -> None:
        result = self._run("tools", "--json")
        self.assertEqual(result.returncode, 0)
        specs = json.loads(result.stdout)
        names = {spec["name"] for spec in specs}
        self.assertIn("calculator", names)
        self.assertNotIn("shell", names)  # 默认关闭
        calculator = next(spec for spec in specs if spec["name"] == "calculator")
        self.assertIn("expression", calculator["parameters"]["required"])

    def test_demo_all_pass(self) -> None:
        result = self._run("demo")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.count("PASS"), 4)
        self.assertIn("0 失败", result.stdout)
        trace_files = list((self.tmp / "runs").glob("demo-*/**/trace.jsonl"))
        self.assertGreaterEqual(len(trace_files), 4)

    def test_run_demo_task(self) -> None:
        result = self._run("run", "--demo-task", "arithmetic")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("978", result.stdout)
        self.assertIn("报告", result.stdout)

    def test_run_arbitrary_goal_with_mock_rejected(self) -> None:
        result = self._run("run", "帮我算 1+1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("provider=mock", result.stderr)
        self.assertIn("demo", result.stderr)

    def test_replay_trace(self) -> None:
        run_result = self._run("run", "--demo-task", "arithmetic")
        self.assertEqual(run_result.returncode, 0, msg=run_result.stderr)
        traces = list((self.tmp / "runs").glob("run-*/trace.jsonl"))
        self.assertEqual(len(traces), 1)
        result = self._run("replay", str(traces[0]))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("plan", result.stdout)
        self.assertIn("critic_final", result.stdout)

    def test_replay_missing_trace(self) -> None:
        result = self._run("replay", str(self.tmp / "nope.jsonl"))
        self.assertEqual(result.returncode, 2)

    def test_eval_mechanism_suite(self) -> None:
        result = self._run("eval", "--suite", "mechanism", "--provider", "mock")
        self.assertEqual(result.returncode, 0, msg=result.stderr[-2000:])
        self.assertIn("mechanism", result.stdout)


if __name__ == "__main__":
    unittest.main()
