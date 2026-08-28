"""provider 基类与传输层重试策略。

``BaseLLM`` 是 :class:`agi_lab.types.LLMProvider` 的模板实现：重试、用量累计与
调用计数集中在 ``complete``，具体提供方只实现一次请求的 ``_complete_once``。

重试只覆盖 ``retryable=True`` 的 :class:`agi_lab.errors.LLMTransportError`。
:class:`agi_lab.errors.LLMResponseFormatError` 不在此层重试：格式错误的修复重试
由认知层按结构化输出协议决定，语义与网络抖动不同。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

from ..errors import LLMTransportError
from ..types import LLMResponse, Message, Usage


@dataclass(frozen=True)
class RetryPolicy:
    """传输层重试策略（指数退避）。

    参数：
        max_attempts: 单次 ``complete`` 的最大尝试次数（含首次），必须 >= 1。
        base_delay: 第 1 次尝试失败后的等待秒数，之后每次翻倍，必须 >= 0。
        max_delay: 单次等待秒数上限，必须 >= 0。
        retry_statuses: 视为可重试的 HTTP 状态码。5xx 另有区间判断，不在此列也重试。
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_statuses: tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts 必须 >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("RetryPolicy 的退避秒数必须 >= 0")

    def delay_for(self, attempt: int) -> float:
        """返回第 ``attempt`` 次尝试失败后的等待秒数。

        参数：
            attempt: 已失败的尝试序号，从 1 开始。

        返回：
            ``min(base_delay * 2 ** (attempt - 1), max_delay)``，不含 jitter。

        异常：
            ValueError: ``attempt`` 小于 1。
        """
        if attempt < 1:
            raise ValueError("attempt 从 1 开始")
        return min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)


class BaseLLM(ABC):
    """模型补全的模板基类，实现 :class:`agi_lab.types.LLMProvider` 的 ``complete``。

    参数：
        name: provider 名称，写入运行时指标与 trace。
        retry: 重试策略；``None`` 用 :class:`RetryPolicy` 默认值。
        sleep: 等待函数，测试注入以避免真实等待。
        jitter: 乘性抖动因子，通常传 ``random.random``；``None`` 表示等待时间不抖动，
            非 ``None`` 时实际等待为 ``退避秒数 * jitter()``。

    属性：
        name: provider 名称。
        retry: 生效的重试策略。
        total_usage: 所有成功补全的 :class:`agi_lab.types.Usage` 之和。
        call_count: 成功补全的次数，重试不重复计数。
    """

    name: str

    def __init__(
        self,
        name: str,
        *,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        self.name = name
        self.retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep
        self._jitter = jitter
        self._total_usage = Usage()
        self._call_count = 0

    @property
    def total_usage(self) -> Usage:
        """所有成功补全的 token 用量之和。"""
        return self._total_usage

    @property
    def call_count(self) -> int:
        """成功补全的次数（重试只算最终成功的一次）。"""
        return self._call_count

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        """执行一次补全，传输层失败按 :class:`RetryPolicy` 重试。

        参数：
            messages: 对话消息序列。
            temperature: 采样温度，``None`` 用 provider 默认值。
            max_tokens: 输出上限，``None`` 用 provider 默认值。
            json_mode: 是否要求提供方返回严格 JSON。
            timeout: 本次请求超时秒数，``None`` 用 provider 默认值。

        返回：
            提供方解析后的 :class:`LLMResponse`；成功一次即累计用量与调用计数。

        异常：
            LLMTransportError: 传输层失败且不可重试或重试耗尽，保留原始 ``retryable`` 与 ``status``。
            LLMResponseFormatError: 提供方返回体不可解析，原样向上抛，不在本层重试。
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._complete_once(
                    messages, temperature, max_tokens, json_mode, timeout
                )
            except LLMTransportError as exc:
                if not exc.retryable or attempt >= self.retry.max_attempts:
                    raise
                self._sleep(self._delay_for(attempt))
                continue
            self._total_usage = self._total_usage + response.usage
            self._call_count += 1
            return response

    def _delay_for(self, attempt: int) -> float:
        delay = self.retry.delay_for(attempt)
        if self._jitter is not None:
            delay *= self._jitter()
        return delay

    @abstractmethod
    def _complete_once(
        self,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        timeout: float | None,
    ) -> LLMResponse:
        """执行一次模型请求，不含重试。

        参数：
            messages: 对话消息序列。
            temperature: 采样温度，``None`` 用实现默认值。
            max_tokens: 输出上限，``None`` 用实现默认值。
            json_mode: 是否要求提供方返回严格 JSON。
            timeout: 本次请求超时秒数，``None`` 用实现默认值。

        返回：
            解析后的 :class:`LLMResponse`。

        异常：
            LLMTransportError: 网络、超时或 HTTP 状态失败；``retryable`` 标明是否可重试。
            LLMResponseFormatError: 返回体结构不可解析。
        """
