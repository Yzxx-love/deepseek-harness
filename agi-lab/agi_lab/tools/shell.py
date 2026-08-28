"""shell 工具：在工作区目录内执行命令。

默认不注册，必须由 ``build_default_tools(allow_shell=True)`` 显式开启；并且
``requires_approval=True``，由注册表按 ``approval_mode`` 审批。子进程环境会剔除
变量名含 ``KEY`` / ``TOKEN`` / ``SECRET`` 的条目，避免模型通过命令读取宿主密钥。
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from ..errors import ToolInvocationError
from ..types import JSONObject, ToolResult, ToolSpec
from .base import DEFAULT_TRUNCATE_CHARS, BaseTool, combine_output, sanitized_environment, truncate_text

DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 60.0


class ShellTool(BaseTool):
    """执行 shell 命令，返回合并后的 stdout/stderr；超时即终止。"""

    def __init__(self, workspace: Path) -> None:
        self._workspace = Path(workspace).resolve()
        self.spec = ToolSpec(
            name="shell",
            description="在工作区目录内执行 shell 命令（有副作用，需审批）",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令字符串"},
                    "timeout": {
                        "type": "number",
                        "description": f"超时秒数，默认 {DEFAULT_TIMEOUT_SECONDS:g}，上限 {MAX_TIMEOUT_SECONDS:g}",
                    },
                },
                "required": ["command"],
            },
            side_effects=True,
            requires_approval=True,
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        command = self._require_str(arguments, "command")
        timeout = self._require_optional_float(
            arguments, "timeout", DEFAULT_TIMEOUT_SECONDS, minimum=0.05, maximum=MAX_TIMEOUT_SECONDS
        )
        if not self._workspace.is_dir():
            raise ToolInvocationError(f"工作区目录不存在: {self._workspace}")
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self._workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=sanitized_environment(),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                error=f"命令超时（{timeout:g} 秒），已终止",
                metadata={"timed_out": True, "timeout": timeout},
            )
        except OSError as exc:
            raise ToolInvocationError(f"命令无法执行: {exc}") from exc
        elapsed = time.perf_counter() - started
        output = truncate_text(combine_output(completed.stdout, completed.stderr), DEFAULT_TRUNCATE_CHARS)
        metadata: JSONObject = {
            "returncode": completed.returncode,
            "timed_out": False,
            "seconds": round(elapsed, 3),
        }
        if completed.returncode != 0:
            return ToolResult(
                ok=False,
                output=output,
                error=f"命令退出码 {completed.returncode}",
                metadata={**metadata, "tool_error": True},
            )
        return ToolResult(ok=True, output=output or "（无输出）", metadata=metadata)
