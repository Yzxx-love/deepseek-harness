"""工具层对外入口：默认工具工厂与常用类导出。

工厂是工具白名单的唯一装配点：只读与计算工具恒有，``python_exec`` / ``shell`` /
``http_get`` 分别受 ``allow_python_exec`` / ``allow_shell`` / ``allow_network`` 控制；
记忆工具仅当传入 duck-typed ``memory`` 对象时注册。本模块不导入 runtime 与 memory 的
具体实现，依赖方向保持在工具层之上。
"""

from __future__ import annotations

from pathlib import Path

from ..types import ApprovalFn
from .base import BaseTool, ToolRegistry
from .calculator import CalculatorTool
from .files import DeleteFileTool, ListDirTool, ReadFileTool, WriteFileTool
from .http import HttpFetchTool
from .memory_tools import MemoryRememberTool, MemorySearchTool
from .pathguard import PathGuard
from .python_exec import PythonExecTool
from .shell import ShellTool

__all__ = [
    "BaseTool",
    "CalculatorTool",
    "DeleteFileTool",
    "HttpFetchTool",
    "ListDirTool",
    "MemoryRememberTool",
    "MemorySearchTool",
    "PathGuard",
    "PythonExecTool",
    "ReadFileTool",
    "ShellTool",
    "ToolRegistry",
    "WriteFileTool",
    "build_default_tools",
]


def build_default_tools(
    *,
    workspace: Path,
    allow_shell: bool = False,
    allow_network: bool = False,
    allow_python_exec: bool = True,
    approval: ApprovalFn | None = None,
    approval_mode: str = "deny",
    memory: object | None = None,
) -> ToolRegistry:
    """装配默认工具注册表。

    参数：
        workspace: 工作区根目录；文件工具与子进程都以它为路径围栏 / cwd。
        allow_shell: 是否注册 ``shell``（默认关闭，有副作用且需审批）。
        allow_network: 是否注册 ``http_get``（默认关闭，需审批）。
        allow_python_exec: 是否注册 ``python_exec``（默认开启）。
        approval: 审批回调，见 :data:`agi_lab.types.ApprovalFn`。
        approval_mode: ``"allow"`` 直接放行需审批工具，``"deny"`` 一律拒绝，其余取值
            在提供 ``approval`` 时询问、否则拒绝（fail closed）。
        memory: duck-typed 记忆门面；不为 ``None`` 时注册 ``memory_search`` 与
            ``memory_remember``。

    返回：
        注册好工具、绑定工作区围栏与审批策略的 :class:`ToolRegistry`。

    异常：
        ValueError: 工具重名（正常装配不会发生）。
    """
    guard = PathGuard(Path(workspace))
    registry = ToolRegistry(approval=approval, approval_mode=approval_mode)
    registry.register(CalculatorTool())
    registry.register(ReadFileTool(guard))
    registry.register(WriteFileTool(guard))
    registry.register(ListDirTool(guard))
    registry.register(DeleteFileTool(guard))
    if allow_python_exec:
        registry.register(PythonExecTool(workspace))
    if allow_shell:
        registry.register(ShellTool(workspace))
    if allow_network:
        registry.register(HttpFetchTool())
    if memory is not None:
        registry.register(MemorySearchTool(memory))
        registry.register(MemoryRememberTool(memory))
    return registry
