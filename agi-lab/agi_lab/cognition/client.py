"""结构化模型调用封装。

所有认知动作都是"一次补全 + JSON 解析 + 可选校验 + 至多一次修复重试"。
把这段逻辑集中在一处，避免 planner / actor / critic / reflector 各写一遍。

与运行时统计的接线通过回调完成：``on_usage`` 累加 token，``on_repair`` 累加格式
修复次数。这样认知层不依赖 ``Metrics``，可独立测试。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypeVar

from ..errors import ActionFormatError, LLMResponseFormatError, PlanFormatError
from ..types import LLMProvider, Message, Usage
from .prompts import repair_messages

T = TypeVar("T")

_VALIDATION_ERRORS = (LLMResponseFormatError, ActionFormatError, PlanFormatError)


class StructuredClient:
    """模型补全 + JSON 提取 + 一次修复重试。

    参数：
        llm: 任意 :class:`agi_lab.types.LLMProvider` 实现。
        on_usage: 每次成功补全后收到 :class:`Usage` 的回调。
        on_repair: 每次触发修复重试时调用。
        max_repairs: 修复重试次数上限，默认 1。
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        on_usage: Callable[[Usage], None] | None = None,
        on_repair: Callable[[], None] | None = None,
        max_repairs: int = 1,
    ) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs 必须 >= 0")
        self.llm = llm
        self._on_usage = on_usage
        self._on_repair = on_repair
        self.max_repairs = max_repairs
        self.calls = 0
        self.repairs = 0

    def call(
        self,
        messages: Sequence[Message],
        *,
        role: str,
        validator: Callable[[dict[str, Any]], T] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> T | dict[str, Any]:
        """执行一次结构化调用。

        参数：
            messages: 提示消息序列。
            role: 角色名，仅用于错误信息与日志。
            validator: 把解析出的 JSON 对象转成目标类型的可调用对象；
                校验失败时抛 ``AgiLabError`` 子类会触发修复重试。
            max_tokens: 单次输出上限，``None`` 用 provider 默认值。
            temperature: 采样温度，``None`` 用 provider 默认值。

        返回：
            有 ``validator`` 时返回其返回值，否则返回解析出的 JSON 对象。

        异常：
            LLMResponseFormatError: ``max_repairs`` 次重试后仍无法得到可解析 JSON。
            LLMTransportError: 网络/HTTP 失败（由 provider 处理重试后仍失败）。
        """
        from ..jsonutil import extract_json_object

        conversation = list(messages)
        last_error: Exception | None = None
        last_text = ""
        for attempt in range(self.max_repairs + 1):
            response = self.llm.complete(
                conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            self.calls += 1
            if self._on_usage is not None:
                self._on_usage(response.usage)
            last_text = response.text
            try:
                obj = extract_json_object(response.text)
                if validator is None:
                    return obj
                return validator(obj)
            except _VALIDATION_ERRORS as exc:
                last_error = exc
                if attempt >= self.max_repairs:
                    break
                self.repairs += 1
                if self._on_repair is not None:
                    self._on_repair()
                conversation = repair_messages(messages, str(exc), last_text)
        if isinstance(last_error, LLMResponseFormatError):
            raise last_error
        raise LLMResponseFormatError(
            f"{role} 输出无法解析：{last_error}",
            excerpt=last_text[:200],
        )
