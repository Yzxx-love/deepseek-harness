"""文件工具：读、写、列目录、删文件，全部经由 :class:`PathGuard` 强制工作区围栏。

写与删的 ``side_effects=True`` 供提示词与运行时识别；删除另需审批（由注册表执行）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..errors import ToolInvocationError
from ..types import JSONObject, ToolResult, ToolSpec
from .base import BaseTool, truncate_text
from .pathguard import PathGuard

DEFAULT_READ_MAX_CHARS = 8000
MAX_READ_CHARS = 200_000
MAX_LIST_ENTRIES = 200


class _GuardTool(BaseTool):
    """持有 :class:`PathGuard` 的文件工具基类。"""

    def __init__(self, guard: PathGuard) -> None:
        self._guard = guard

    def _resolve(self, path_str: str, *, must_exist: bool = False) -> Path:
        return self._guard.resolve(path_str, must_exist=must_exist)

    def _relative(self, path: Path) -> str:
        return self._guard.relative(path)


class ReadFileTool(_GuardTool):
    """读取工作区内一个 UTF-8 文本文件。"""

    def __init__(self, guard: PathGuard) -> None:
        super().__init__(guard)
        self.spec = ToolSpec(
            name="read_file",
            description="读取工作区内的 UTF-8 文本文件，超长内容截断",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对工作区的文件路径"},
                    "max_chars": {
                        "type": "integer",
                        "description": f"最多返回的字符数，默认 {DEFAULT_READ_MAX_CHARS}",
                    },
                },
                "required": ["path"],
            },
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        path_str = self._require_str(arguments, "path")
        max_chars = self._require_optional_int(
            arguments, "max_chars", DEFAULT_READ_MAX_CHARS, minimum=1, maximum=MAX_READ_CHARS
        )
        target = self._resolve(path_str, must_exist=True)
        if target.is_dir():
            raise ToolInvocationError(f"路径是目录，read_file 只能读文件: {path_str}")
        try:
            raw = target.read_bytes()
        except OSError as exc:
            raise ToolInvocationError(f"读取文件失败: {path_str}（{exc}）") from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolInvocationError(f"文件不是有效 UTF-8 文本: {path_str}（{exc.reason}）") from None
        metadata: JSONObject = {
            "path": self._relative(target),
            "chars": len(text),
            "bytes": len(raw),
            "truncated": len(text) > max_chars,
        }
        return ToolResult(ok=True, output=truncate_text(text, max_chars), metadata=metadata)


class WriteFileTool(_GuardTool):
    """写入或追加文件；自动创建父目录。"""

    def __init__(self, guard: PathGuard) -> None:
        super().__init__(guard)
        self.spec = ToolSpec(
            name="write_file",
            description="把文本写入工作区文件（自动创建父目录，mode=overwrite 覆盖 / append 追加）",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对工作区的文件路径"},
                    "content": {"type": "string", "description": "要写入的 UTF-8 文本"},
                    "mode": {"type": "string", "description": "overwrite（默认）或 append"},
                },
                "required": ["path", "content"],
            },
            side_effects=True,
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        path_str = self._require_str(arguments, "path")
        content = self._require_str(arguments, "content", allow_empty=True)
        raw_mode = arguments.get("mode", "overwrite")
        if raw_mode is None:
            raw_mode = "overwrite"
        if raw_mode not in ("overwrite", "append"):
            raise ToolInvocationError(f"参数 mode 必须是 overwrite 或 append，收到 {raw_mode!r}")
        mode = str(raw_mode)
        target = self._resolve(path_str)
        if target.is_dir():
            raise ToolInvocationError(f"路径是目录，write_file 只能写文件: {path_str}")
        existed = target.exists()
        payload = content.encode("utf-8")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a" if mode == "append" else "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
        except OSError as exc:
            raise ToolInvocationError(f"写入文件失败: {path_str}（{exc}）") from exc
        relative = self._relative(target)
        verb = "追加" if mode == "append" else "写入"
        metadata: JSONObject = {
            "path": relative,
            "bytes": len(payload),
            "mode": mode,
            "created": not existed,
        }
        return ToolResult(ok=True, output=f"已{verb} {len(payload)} 字节: {relative}", metadata=metadata)


class ListDirTool(_GuardTool):
    """列出工作区内的目录项。"""

    def __init__(self, guard: PathGuard) -> None:
        super().__init__(guard)
        self.spec = ToolSpec(
            name="list_dir",
            description="列出工作区目录内容，目录名带 / 后缀，最多 200 项",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对工作区的目录路径，默认 ."},
                },
                "required": [],
            },
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raw_path = arguments.get("path") or "."
        if not isinstance(raw_path, str):
            raise ToolInvocationError("参数 path 必须是字符串")
        target = self._resolve(raw_path, must_exist=True)
        if not target.is_dir():
            raise ToolInvocationError(f"路径不是目录: {raw_path}")
        try:
            entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError as exc:
            raise ToolInvocationError(f"读取目录失败: {raw_path}（{exc}）") from exc
        shown = entries[:MAX_LIST_ENTRIES]
        lines = [item.name + ("/" if item.is_dir() else "") for item in shown]
        if len(entries) > len(shown):
            lines.append(f"…[截断，共 {len(entries)} 项，仅显示前 {len(shown)} 项]")
        output = "\n".join(lines) if lines else "（空目录）"
        metadata: JSONObject = {
            "path": self._relative(target),
            "entries": len(entries),
            "shown": len(shown),
            "truncated": len(entries) > len(shown),
        }
        return ToolResult(ok=True, output=output, metadata=metadata)


class DeleteFileTool(_GuardTool):
    """删除工作区内的单个文件；需要审批，且拒绝删除目录。"""

    def __init__(self, guard: PathGuard) -> None:
        super().__init__(guard)
        self.spec = ToolSpec(
            name="delete_file",
            description="删除工作区内的单个文件（仅文件，不删目录），需要审批",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对工作区的文件路径"},
                },
                "required": ["path"],
            },
            side_effects=True,
            requires_approval=True,
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        path_str = self._require_str(arguments, "path")
        target = self._resolve(path_str, must_exist=True)
        if target.is_dir():
            raise ToolInvocationError(f"目标是目录，delete_file 只删除文件: {path_str}")
        try:
            size = target.stat().st_size
            target.unlink()
        except OSError as exc:
            raise ToolInvocationError(f"删除文件失败: {path_str}（{exc}）") from exc
        relative = self._relative(target)
        metadata: JSONObject = {"path": relative, "bytes": size}
        return ToolResult(ok=True, output=f"已删除文件: {relative}", metadata=metadata)
