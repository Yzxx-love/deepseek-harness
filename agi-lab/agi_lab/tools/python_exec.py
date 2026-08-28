"""python_exec 工具：在隔离模式子进程中执行一段 Python 代码。

默认注册，可由 ``build_default_tools(allow_python_exec=False)`` 关闭。子进程带 ``-I``
（隔离模式，忽略用户 site 与 PYTHONPATH），cwd 固定为工作区；超时时显式 ``kill()``。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from ..errors import ToolInvocationError
from ..types import JSONObject, ToolResult, ToolSpec
from .base import DEFAULT_TRUNCATE_CHARS, BaseTool, combine_output, sanitized_environment, truncate_text

DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 60.0
_MAX_KILL_WAIT_SECONDS = 5.0


class PythonExecTool(BaseTool):
    """运行一段 Python 代码，返回合并后的 stdout/stderr。"""

    def __init__(self, workspace: Path) -> None:
        self._workspace = Path(workspace).resolve()
        self.spec = ToolSpec(
            name="python_exec",
            description="在工作区目录内运行一段 Python 代码并返回其 stdout/stderr",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 源码"},
                    "timeout": {
                        "type": "number",
                        "description": f"超时秒数，默认 {DEFAULT_TIMEOUT_SECONDS:g}，上限 {MAX_TIMEOUT_SECONDS:g}",
                    },
                },
                "required": ["code"],
            },
            side_effects=True,
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        code = self._require_str(arguments, "code", allow_empty=True)
        timeout = self._require_optional_float(
            arguments, "timeout", DEFAULT_TIMEOUT_SECONDS, minimum=0.05, maximum=MAX_TIMEOUT_SECONDS
        )
        if not self._workspace.is_dir():
            raise ToolInvocationError(f"工作区目录不存在: {self._workspace}")
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", code],
                cwd=str(self._workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=sanitized_environment(),
            )
        except OSError as exc:
            raise ToolInvocationError(f"无法启动 Python 子进程: {exc}") from exc
        started = time.perf_counter()
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.communicate(timeout=_MAX_KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass  # 孙进程仍占用管道时不再等待，超时结论已经确定
            return ToolResult(
                ok=False,
                error=f"python_exec 超时（{timeout:g} 秒），子进程已 kill",
                metadata={"timed_out": True, "timeout": timeout},
            )
        elapsed = time.perf_counter() - started
        output = truncate_text(combine_output(stdout, stderr), DEFAULT_TRUNCATE_CHARS)
        metadata: JSONObject = {
            "returncode": process.returncode,
            "timed_out": False,
            "seconds": round(elapsed, 3),
        }
        if process.returncode != 0:
            return ToolResult(
                ok=False,
                output=output,
                error=f"Python 进程退出码 {process.returncode}",
                metadata={**metadata, "tool_error": True},
            )
        return ToolResult(ok=True, output=output or "（无输出）", metadata=metadata)
