"""运行时安全策略：交互审批、密钥脱敏与安全事件归类。

路径围栏由工具层的 ``PathGuard`` 强制，本模块负责另外两件事：

1. ``prompt`` 审批模式下的交互式许可（非交互环境一律 fail closed）；
2. 进入 trace 与提示词之前的内容脱敏，避免密钥随工具输出泄漏。
"""

from __future__ import annotations

import re
import sys
from typing import Callable, Mapping

from ..config import Config
from ..types import ApprovalFn, ToolResult, ToolSpec

_SK_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,})")


class PolicyGuard:
    """安全策略的运行期助手（不实现路径围栏，围栏在工具层）。

    参数：
        config: 提供 ``api_key`` 与 ``approval_mode``。
        input_fn: 审批提问的输入函数，``None`` 时用 ``input``（仅在交互终端）。
        output_fn: 审批提示的输出函数，默认 ``print``。
        interactive: 是否允许交互提问；``None`` 时按 ``sys.stdin.isatty()`` 判断。
    """

    def __init__(
        self,
        config: Config,
        *,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] = print,
        interactive: bool | None = None,
    ) -> None:
        self.config = config
        self._input_fn = input_fn
        self._output_fn = output_fn
        self.interactive = sys.stdin.isatty() if interactive is None else interactive

    def approval(self) -> ApprovalFn | None:
        """返回供 ``ToolRegistry`` 使用的审批回调。

        ``deny`` / ``allow`` 模式由注册表自身处理，返回 ``None``；
        ``prompt`` 模式返回交互回调，非交互环境下该回调恒拒绝（fail closed）。
        """
        if self.config.approval_mode != "prompt":
            return None
        if not self.interactive and self._input_fn is None:
            def _deny(spec: ToolSpec, arguments: Mapping[str, object]) -> bool:
                self._output_fn(f"[安全] 非交互环境拒绝需审批工具：{spec.name}")
                return False

            return _deny
        return self._ask

    def _ask(self, spec: ToolSpec, arguments: Mapping[str, object]) -> bool:
        self._output_fn(f"[审批] 工具 {spec.name} 请求执行，参数：{dict(arguments)}")
        prompt = f"允许执行 {spec.name}？[y/N] "
        if self._input_fn is not None:
            answer = self._input_fn(prompt)
        else:
            answer = input(prompt)
        return answer.strip().lower() in ("y", "yes", "是")

    def redact(self, text: str) -> str:
        """脱敏文本中的密钥与 ``sk-`` 形态 token。

        参数：
            text: 原始文本。

        返回：
            脱敏后的文本；未配置密钥时只做 ``sk-`` 形态替换。
        """
        if not text:
            return text
        if self.config.api_key:
            text = text.replace(self.config.api_key, "***")
        return _SK_PATTERN.sub("***", text)

    @staticmethod
    def is_safety_block(result: ToolResult) -> bool:
        """判断工具结果是否来自安全策略拒绝。"""
        return bool(result.metadata.get("safety"))
