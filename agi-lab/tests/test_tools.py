"""工具层测试：计算器安全、路径围栏、文件工具、注册表审批、子进程、记忆工具。

全部使用标准库 unittest 与 ``tempfile.TemporaryDirectory``，不访问网络。Windows 上
创建符号链接可能因权限失败，对应用例使用 ``skipTest`` 而不是让套件变红。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agi_lab.errors import SafetyViolation, ToolInvocationError
from agi_lab.tools import (
    BaseTool,
    CalculatorTool,
    PathGuard,
    ToolRegistry,
    build_default_tools,
)
from agi_lab.types import ToolCall, ToolResult, ToolSpec


class _WorkspaceCase(unittest.TestCase):
    """每个用例一个临时工作区。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def registry(self, **kwargs: object) -> ToolRegistry:
        kwargs.setdefault("workspace", self.workspace)
        return build_default_tools(**kwargs)  # type: ignore[arg-type]

    def call(self, registry: ToolRegistry, tool: str, **arguments: object) -> ToolResult:
        return registry.execute(ToolCall(tool, dict(arguments)))


class CalculatorToolTests(_WorkspaceCase):
    def test_basic_arithmetic(self) -> None:
        result = self.call(self.registry(), "calculator", expression="2 + 3 * 4")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output, "14")
        self.assertEqual(result.metadata["expression"], "2 + 3 * 4")
        self.assertEqual(result.metadata["value"], 14)

    def test_operator_set(self) -> None:
        cases = {
            "(2 + 3) * 4": "20",
            "7 / 2": "3.5",
            "10 // 3": "3",
            "10 % 4": "2",
            "2 ** 10": "1024",
            "-5 + 2": "-3",
            "2 ** 0.5": "1.4142135623730951",
        }
        registry = self.registry()
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = self.call(registry, "calculator", expression=expression)
                self.assertTrue(result.ok, result.error)
                self.assertEqual(result.output, expected)

    def test_division_by_zero(self) -> None:
        result = self.call(self.registry(), "calculator", expression="1 / 0")
        self.assertFalse(result.ok)
        self.assertIn("除零", result.error or "")

    def test_rejects_import_call(self) -> None:
        result = self.call(self.registry(), "calculator", expression='__import__("os")')
        self.assertFalse(result.ok)
        self.assertIn("Call", result.error or "")

    def test_rejects_attribute_access(self) -> None:
        result = self.call(self.registry(), "calculator", expression="(1).__class__")
        self.assertFalse(result.ok)
        self.assertIn("Attribute", result.error or "")

    def test_rejects_huge_exponent(self) -> None:
        result = self.call(self.registry(), "calculator", expression="2 ** 1000")
        self.assertFalse(result.ok)
        self.assertIn("指数", result.error or "")

    def test_rejects_overlong_expression(self) -> None:
        result = self.call(self.registry(), "calculator", expression="1+" * 300 + "1")
        self.assertFalse(result.ok)
        self.assertIn("过长", result.error or "")

    def test_rejects_name(self) -> None:
        result = self.call(self.registry(), "calculator", expression="x + 1")
        self.assertFalse(result.ok)
        self.assertIn("Name", result.error or "")

    def test_rejects_non_numeric_constant(self) -> None:
        result = self.call(self.registry(), "calculator", expression='"a" * 3')
        self.assertFalse(result.ok)
        self.assertIn("数字常量", result.error or "")

    def test_rejects_other_nodes(self) -> None:
        result = self.call(self.registry(), "calculator", expression="[1, 2]")
        self.assertFalse(result.ok)
        self.assertIn("List", result.error or "")

    def test_rejects_infinite_float(self) -> None:
        result = self.call(self.registry(), "calculator", expression="1e308 * 1e308")
        self.assertFalse(result.ok)
        self.assertIn("有限数值", result.error or "")

    def test_rejects_empty_expression(self) -> None:
        result = self.call(self.registry(), "calculator", expression="   ")
        self.assertFalse(result.ok)
        self.assertIn("不能为空", result.error or "")

    def test_tool_available_in_default_factory(self) -> None:
        self.assertIn("calculator", self.registry().names())


class PathGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "workspace"
        self.root.mkdir()
        self.guard = PathGuard(self.root)

    def test_rejects_parent_escape(self) -> None:
        with self.assertRaises(SafetyViolation):
            self.guard.resolve("../x.txt")

    def test_rejects_nested_escape(self) -> None:
        with self.assertRaises(SafetyViolation):
            self.guard.resolve("a/../../x.txt")

    def test_rejects_absolute_outside(self) -> None:
        outside = "C:\\Windows\\x" if os.name == "nt" else "/etc/passwd"
        with self.assertRaises(SafetyViolation):
            self.guard.resolve(outside)

    def test_allows_relative_inside(self) -> None:
        resolved = self.guard.resolve("a/b.txt")
        self.assertEqual(resolved, self.root / "a" / "b.txt")
        self.assertTrue(resolved.is_relative_to(self.guard.root))

    def test_allows_absolute_inside(self) -> None:
        resolved = self.guard.resolve(str(self.root / "c.txt"))
        self.assertEqual(resolved, self.root / "c.txt")

    def test_root_itself_is_allowed(self) -> None:
        self.assertEqual(self.guard.resolve("."), self.guard.root)

    def test_must_exist(self) -> None:
        with self.assertRaises(ToolInvocationError):
            self.guard.resolve("missing.txt", must_exist=True)

    def test_exists_and_relative(self) -> None:
        target = self.root / "a.txt"
        target.write_text("x", encoding="utf-8")
        self.assertTrue(self.guard.exists("a.txt"))
        self.assertFalse(self.guard.exists("b.txt"))
        self.assertEqual(self.guard.relative(target), "a.txt")
        self.assertEqual(self.guard.relative(self.root), ".")

    def test_symlink_escape_rejected(self) -> None:
        outside = Path(self._tmp.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.root / "link.txt"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("当前平台不允许创建符号链接（Windows 需开发者模式）")
        with self.assertRaises(SafetyViolation):
            self.guard.resolve("link.txt")
        # 即使打开 allow_symlinks，解析后越界仍必须拒绝。
        with self.assertRaises(SafetyViolation):
            PathGuard(self.root, allow_symlinks=True).resolve("link.txt")

    def test_internal_symlink_rejected_by_default(self) -> None:
        target = self.root / "a.txt"
        target.write_text("x", encoding="utf-8")
        link = self.root / "inner-link.txt"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            self.skipTest("当前平台不允许创建符号链接（Windows 需开发者模式）")
        with self.assertRaises(SafetyViolation):
            self.guard.resolve("inner-link.txt")
        self.assertEqual(PathGuard(self.root, allow_symlinks=True).resolve("inner-link.txt"), target)


class FileToolTests(_WorkspaceCase):
    def test_write_read_list_round_trip(self) -> None:
        registry = self.registry()
        content = "hello 世界\nsecond line"
        written = self.call(registry, "write_file", path="notes/a.txt", content=content)
        self.assertTrue(written.ok, written.error)
        self.assertEqual(written.metadata["bytes"], len(content.encode("utf-8")))
        self.assertEqual(written.metadata["path"], "notes/a.txt")
        self.assertTrue(written.metadata["created"])
        read = self.call(registry, "read_file", path="notes/a.txt")
        self.assertTrue(read.ok, read.error)
        self.assertEqual(read.output, content)
        self.assertFalse(read.metadata["truncated"])
        self.assertEqual(read.metadata["path"], "notes/a.txt")
        self.assertEqual(read.metadata["bytes"], len(content.encode("utf-8")))
        listing = self.call(registry, "list_dir", path="notes")
        self.assertTrue(listing.ok, listing.error)
        self.assertEqual(listing.output, "a.txt")
        self.assertEqual(listing.metadata["entries"], 1)
        root_listing = self.call(registry, "list_dir")
        self.assertTrue(root_listing.ok, root_listing.error)
        self.assertEqual(root_listing.output, "notes/")

    def test_append_mode(self) -> None:
        registry = self.registry()
        self.assertTrue(self.call(registry, "write_file", path="log.txt", content="a").ok)
        second = self.call(registry, "write_file", path="log.txt", content="b", mode="append")
        self.assertTrue(second.ok, second.error)
        self.assertEqual(second.metadata["mode"], "append")
        self.assertFalse(second.metadata["created"])
        self.assertEqual(self.call(registry, "read_file", path="log.txt").output, "ab")

    def test_invalid_mode(self) -> None:
        result = self.call(self.registry(), "write_file", path="a.txt", content="x", mode="rewrite")
        self.assertFalse(result.ok)
        self.assertIn("mode", result.error or "")

    def test_read_missing_file_is_friendly(self) -> None:
        result = self.call(self.registry(), "read_file", path="missing.txt")
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.error or "")
        self.assertNotIn("Traceback", result.error or "")

    def test_read_truncates(self) -> None:
        registry = self.registry()
        self.assertTrue(self.call(registry, "write_file", path="big.txt", content="x" * 50).ok)
        result = self.call(registry, "read_file", path="big.txt", max_chars=10)
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.output.startswith("x" * 10))
        self.assertIn("截断", result.output)
        self.assertTrue(result.metadata["truncated"])

    def test_path_escape_rejected_for_file_tools(self) -> None:
        registry = self.registry()
        # delete_file 需审批，默认 deny 时会在路径校验前被拒绝；这里放行审批以测到围栏。
        approval_registry = self.registry(approval_mode="allow")
        cases = (
            (registry, "read_file", {"path": "../secret.txt"}),
            (registry, "write_file", {"path": "../secret.txt", "content": "x"}),
            (registry, "list_dir", {"path": "../"}),
            (approval_registry, "delete_file", {"path": "../secret.txt"}),
        )
        for active, tool, arguments in cases:
            with self.subTest(tool=tool):
                result = self.call(active, tool, **arguments)
                self.assertFalse(result.ok)
                self.assertTrue(result.metadata.get("safety"))
                self.assertIn("越界", result.error or "")
        self.assertFalse((self.workspace.parent / "secret.txt").exists())

    def test_absolute_escape_rejected(self) -> None:
        outside = "C:\\Windows\\win.ini" if os.name == "nt" else "/etc/passwd"
        result = self.call(self.registry(), "read_file", path=outside)
        self.assertFalse(result.ok)
        self.assertTrue(result.metadata.get("safety"))

    def test_delete_file_with_approval_allow(self) -> None:
        registry = self.registry(approval_mode="allow")
        self.assertTrue(self.call(registry, "write_file", path="gone.txt", content="x").ok)
        result = self.call(registry, "delete_file", path="gone.txt")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metadata["path"], "gone.txt")
        self.assertFalse((self.workspace / "gone.txt").exists())
        again = self.call(registry, "delete_file", path="gone.txt")
        self.assertFalse(again.ok)
        self.assertIn("不存在", again.error or "")

    def test_delete_file_rejects_directory(self) -> None:
        registry = self.registry(approval_mode="allow")
        (self.workspace / "sub").mkdir()
        result = self.call(registry, "delete_file", path="sub")
        self.assertFalse(result.ok)
        self.assertIn("目录", result.error or "")
        self.assertTrue((self.workspace / "sub").is_dir())

    def test_list_dir_marks_directories_and_sorts(self) -> None:
        registry = self.registry()
        (self.workspace / "b-dir").mkdir()
        self.assertTrue(self.call(registry, "write_file", path="a.txt", content="x").ok)
        result = self.call(registry, "list_dir")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output.splitlines(), ["b-dir/", "a.txt"])

    def test_list_dir_missing_path(self) -> None:
        result = self.call(self.registry(), "list_dir", path="nope")
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.error or "")


class _BoomTool(BaseTool):
    """总是抛指定异常的工具，用于验证注册表归类。"""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.spec = ToolSpec(
            name="boom",
            description="总是失败的测试工具",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def _execute(self, arguments: object) -> ToolResult:
        raise self._error


class _BadReturnTool(BaseTool):
    """返回值不是 ToolResult 的工具。"""

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="bad_return",
            description="返回值非法的测试工具",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def _execute(self, arguments: object) -> ToolResult:
        return "not a result"  # type: ignore[return-value]


class ToolRegistryTests(_WorkspaceCase):
    def test_unknown_tool(self) -> None:
        result = self.call(self.registry(), "nope", x=1)
        self.assertFalse(result.ok)
        self.assertIn("未知工具", result.error or "")
        self.assertTrue(result.metadata.get("unknown_tool"))

    def test_missing_required_argument(self) -> None:
        result = self.call(self.registry(), "calculator")
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "缺少参数: expression")
        self.assertTrue(result.metadata.get("invalid_arguments"))

    def test_argument_type_mismatch(self) -> None:
        result = self.call(self.registry(), "calculator", expression=123)
        self.assertFalse(result.ok)
        self.assertIn("string", result.error or "")
        self.assertTrue(result.metadata.get("invalid_arguments"))

    def test_approval_denied_by_default(self) -> None:
        registry = self.registry()
        (self.workspace / "keep.txt").write_text("x", encoding="utf-8")
        result = self.call(registry, "delete_file", path="keep.txt")
        self.assertFalse(result.ok)
        self.assertTrue(result.metadata.get("safety"))
        self.assertEqual(result.metadata.get("denied"), "approval_required")
        self.assertTrue((self.workspace / "keep.txt").exists())

    def test_approval_precedes_path_validation(self) -> None:
        """需审批工具在默认 deny 下先被审批拒绝，不执行工具内的路径围栏。"""
        result = self.call(self.registry(), "delete_file", path="../secret.txt")
        self.assertFalse(result.ok)
        self.assertTrue(result.metadata.get("safety"))
        self.assertEqual(result.metadata.get("denied"), "approval_required")
        self.assertNotIn("越界", result.error or "")

    def test_approval_allow_mode(self) -> None:
        registry = self.registry(approval_mode="allow")
        (self.workspace / "keep.txt").write_text("x", encoding="utf-8")
        result = self.call(registry, "delete_file", path="keep.txt")
        self.assertTrue(result.ok, result.error)
        self.assertFalse((self.workspace / "keep.txt").exists())

    def test_approval_callback_denied(self) -> None:
        seen: list[tuple[str, dict[str, object]]] = []

        def deny(spec: ToolSpec, arguments: object) -> bool:
            seen.append((spec.name, dict(arguments)))  # type: ignore[arg-type]
            return False

        registry = self.registry(approval=deny, approval_mode="prompt")
        (self.workspace / "keep.txt").write_text("x", encoding="utf-8")
        result = self.call(registry, "delete_file", path="keep.txt")
        self.assertFalse(result.ok)
        self.assertTrue(result.metadata.get("safety"))
        self.assertEqual(seen[0][0], "delete_file")
        self.assertTrue((self.workspace / "keep.txt").exists())

    def test_approval_callback_approved(self) -> None:
        registry = self.registry(approval=lambda spec, arguments: True, approval_mode="prompt")
        (self.workspace / "keep.txt").write_text("x", encoding="utf-8")
        result = self.call(registry, "delete_file", path="keep.txt")
        self.assertTrue(result.ok, result.error)

    def test_approval_callback_failure_fails_closed(self) -> None:
        def explode(spec: ToolSpec, arguments: object) -> bool:
            raise RuntimeError("审批器故障")

        registry = self.registry(approval=explode, approval_mode="prompt")
        (self.workspace / "keep.txt").write_text("x", encoding="utf-8")
        result = self.call(registry, "delete_file", path="keep.txt")
        self.assertFalse(result.ok)
        self.assertTrue(result.metadata.get("safety"))
        self.assertTrue((self.workspace / "keep.txt").exists())

    def test_safety_violation_from_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(_BoomTool(SafetyViolation("路径越界")))
        result = registry.execute(ToolCall("boom"))
        self.assertFalse(result.ok)
        self.assertIn("越界", result.error or "")
        self.assertTrue(result.metadata.get("safety"))
        self.assertIsNone(result.metadata.get("tool_error"))

    def test_tool_exception_becomes_result(self) -> None:
        registry = ToolRegistry()
        registry.register(_BoomTool(ValueError("炸了")))
        result = registry.execute(ToolCall("boom"))
        self.assertFalse(result.ok)
        self.assertIn("ValueError", result.error or "")
        self.assertIn("炸了", result.error or "")
        self.assertTrue(result.metadata.get("tool_error"))

    def test_bad_return_value(self) -> None:
        registry = ToolRegistry()
        registry.register(_BadReturnTool())
        result = registry.execute(ToolCall("bad_return"))
        self.assertFalse(result.ok)
        self.assertIn("ToolResult", result.error or "")

    def test_execute_never_raises_for_bad_arguments(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = registry.execute(ToolCall("calculator", []))  # type: ignore[arg-type]
        self.assertFalse(result.ok)
        self.assertTrue(result.metadata.get("invalid_arguments"))

    def test_register_duplicate_and_lookup(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        with self.assertRaises(ValueError):
            registry.register(CalculatorTool())
        self.assertIsInstance(registry.get("calculator"), CalculatorTool)
        registry.unregister("calculator")
        with self.assertRaises(KeyError):
            registry.get("calculator")
        self.assertEqual(registry.names(), [])

    def test_call_tracking(self) -> None:
        registry = self.registry()
        self.assertEqual(registry.call_count, 0)
        registry.execute(ToolCall("calculator", {"expression": "1 + 1"}))
        self.assertEqual(registry.call_count, 1)
        self.assertEqual(registry.last_tool_name, "calculator")
        self.assertGreaterEqual(registry.last_duration, 0.0)


class ShellToolTests(_WorkspaceCase):
    def test_not_registered_by_default(self) -> None:
        self.assertNotIn("shell", self.registry().names())

    def test_echo_with_allow_and_approval(self) -> None:
        registry = self.registry(allow_shell=True, approval_mode="allow")
        result = self.call(registry, "shell", command="echo hi")
        self.assertTrue(result.ok, result.error)
        self.assertIn("hi", result.output)
        self.assertEqual(result.metadata["returncode"], 0)

    def test_secret_environment_is_filtered(self) -> None:
        registry = self.registry(allow_shell=True, approval_mode="allow")
        command = (
            f'"{sys.executable}" -c "'
            "import os;print(os.environ.get('AGI_LAB_TEST_SECRET_TOKEN', 'MISSING'))"
            '"'
        )
        with mock.patch.dict(os.environ, {"AGI_LAB_TEST_SECRET_TOKEN": "leak-me"}):
            result = self.call(registry, "shell", command=command)
        self.assertTrue(result.ok, result.error)
        self.assertNotIn("leak-me", result.output)
        self.assertIn("MISSING", result.output)

    def test_timeout_reports_failure(self) -> None:
        registry = self.registry(allow_shell=True, approval_mode="allow")
        command = f'"{sys.executable}" -c "import time;time.sleep(2)"'
        started = time.perf_counter()
        result = self.call(registry, "shell", command=command, timeout=0.2)
        elapsed = time.perf_counter() - started
        self.assertFalse(result.ok)
        self.assertIn("超时", result.error or "")
        self.assertTrue(result.metadata.get("timed_out"))
        self.assertLess(elapsed, 15.0)

    def test_nonzero_exit_keeps_output(self) -> None:
        registry = self.registry(allow_shell=True, approval_mode="allow")
        command = f'"{sys.executable}" -c "import sys;print(\'bad\');sys.exit(3)"'
        result = self.call(registry, "shell", command=command)
        self.assertFalse(result.ok)
        self.assertIn("退出码 3", result.error or "")
        self.assertIn("bad", result.output)
        self.assertEqual(result.metadata["returncode"], 3)


class PythonExecToolTests(_WorkspaceCase):
    def test_print_expression(self) -> None:
        result = self.call(self.registry(), "python_exec", code="print(1 + 1)")
        self.assertTrue(result.ok, result.error)
        self.assertIn("2", result.output)

    def test_runs_in_workspace(self) -> None:
        result = self.call(self.registry(), "python_exec", code="import os;print(os.getcwd())")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(Path(result.output.strip()).resolve(), self.workspace.resolve())

    def test_timeout_kills_process(self) -> None:
        started = time.perf_counter()
        result = self.call(self.registry(), "python_exec", code="import time\ntime.sleep(5)", timeout=0.2)
        elapsed = time.perf_counter() - started
        self.assertFalse(result.ok)
        self.assertIn("超时", result.error or "")
        self.assertTrue(result.metadata.get("timed_out"))
        self.assertLess(elapsed, 10.0)

    def test_nonzero_exit(self) -> None:
        result = self.call(self.registry(), "python_exec", code="raise SystemExit(3)")
        self.assertFalse(result.ok)
        self.assertIn("退出码 3", result.error or "")

    def test_disabled_by_factory(self) -> None:
        registry = self.registry(allow_python_exec=False)
        self.assertNotIn("python_exec", registry.names())


class _FakeMemory:
    """duck-typed 记忆门面：只实现工具要求的两个方法。"""

    def __init__(self) -> None:
        self.facts: list[tuple[str, list[str]]] = []
        self.queries: list[tuple[str, int]] = []

    def search(self, *, text: str, k: int = 5) -> list[SimpleNamespace]:
        self.queries.append((text, k))
        hit = SimpleNamespace(item=SimpleNamespace(kind="semantic", text=f"已知事实：{text}"), score=0.875)
        return [hit][:k]

    def add_fact(self, *, text: str, tags: list[str] | None = None) -> str:
        self.facts.append((text, list(tags or [])))
        return "mem-42"


class _EmptyMemory:
    def search(self, *, text: str, k: int = 5) -> list[object]:
        return []


class MemoryToolTests(_WorkspaceCase):
    def setUp(self) -> None:
        super().setUp()
        self.memory = _FakeMemory()
        self.registry_with_memory = self.registry(memory=self.memory)

    def test_search_renders_hits(self) -> None:
        result = self.call(self.registry_with_memory, "memory_search", query="python")
        self.assertTrue(result.ok, result.error)
        self.assertIn("semantic", result.output)
        self.assertIn("0.875", result.output)
        self.assertIn("已知事实：python", result.output)
        self.assertEqual(result.metadata["count"], 1)
        self.assertEqual(self.memory.queries, [("python", 5)])

    def test_search_without_hits(self) -> None:
        registry = self.registry(memory=_EmptyMemory())
        result = self.call(registry, "memory_search", query="nothing")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output, "无相关记忆")
        self.assertEqual(result.metadata["count"], 0)

    def test_remember_writes_fact(self) -> None:
        result = self.call(self.registry_with_memory, "memory_remember", text="天是蓝的", tags=["color"])
        self.assertTrue(result.ok, result.error)
        self.assertIn("mem-42", result.output)
        self.assertEqual(result.metadata["id"], "mem-42")
        self.assertEqual(self.memory.facts, [("天是蓝的", ["color"])])

    def test_missing_methods_are_reported(self) -> None:
        registry = self.registry(memory=object())
        search = self.call(registry, "memory_search", query="x")
        self.assertFalse(search.ok)
        self.assertIn("缺少", search.error or "")
        remember = self.call(registry, "memory_remember", text="x")
        self.assertFalse(remember.ok)
        self.assertIn("缺少", remember.error or "")

    def test_not_registered_without_memory(self) -> None:
        names = self.registry().names()
        self.assertNotIn("memory_search", names)
        self.assertNotIn("memory_remember", names)


class FactoryAndDescribeTests(_WorkspaceCase):
    def test_default_registration(self) -> None:
        names = set(self.registry().names())
        self.assertEqual(
            names,
            {"calculator", "read_file", "write_file", "list_dir", "delete_file", "python_exec"},
        )

    def test_optional_tool_switches(self) -> None:
        registry = self.registry(allow_shell=True, allow_network=True)
        self.assertIn("shell", registry.names())
        self.assertIn("http_get", registry.names())
        self.assertNotIn("python_exec", self.registry(allow_python_exec=False).names())

    def test_http_requires_approval(self) -> None:
        registry = self.registry(allow_network=True)
        result = self.call(registry, "http_get", url="http://example.com")
        self.assertFalse(result.ok)
        self.assertTrue(result.metadata.get("safety"))
        self.assertEqual(result.metadata.get("denied"), "approval_required")

    def test_http_rejects_non_http_scheme(self) -> None:
        registry = self.registry(allow_network=True, approval_mode="allow")
        result = self.call(registry, "http_get", url="file:///C:/Windows/win.ini")
        self.assertFalse(result.ok)
        self.assertIn("http/https", result.error or "")

    def test_describe_renders_tools_and_required_markers(self) -> None:
        text = self.registry().describe()
        self.assertIn("- calculator(expression: string)", text)
        self.assertIn("read_file(path: string, max_chars?: integer)", text)
        self.assertIn("- delete_file(path: string)", text)
        self.assertIn("需审批", text)
        self.assertIn("有副作用", text)
        self.assertIn("- python_exec(code: string, timeout?: number)", text)

    def test_describe_truncates(self) -> None:
        text = self.registry().describe(max_chars=20)
        self.assertIn("截断", text)
        self.assertLess(len(text), 60)

    def test_empty_registry_describe(self) -> None:
        self.assertEqual(ToolRegistry().describe(), "（无可用工具）")

    def test_specs_follow_names_order(self) -> None:
        registry = self.registry()
        self.assertEqual([spec.name for spec in registry.specs()], registry.names())


if __name__ == "__main__":
    unittest.main()
