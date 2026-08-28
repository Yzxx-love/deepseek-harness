"""资源预算：所有循环的硬上限。

预算不是提示词里的建议，而是每次循环前强制检查的运行时约束。任何一维耗尽都
停止运行，并在 :class:`agi_lab.errors.BudgetExceeded` 中带上维度名、上限与已用量，
使报告能明确说明"停在哪一步、为什么停"。

时间用 ``time.monotonic`` 累计，避免系统时钟回拨导致预算被绕过。
"""

from __future__ import annotations

import time
from typing import Callable

from ..config import Config
from ..errors import BudgetExceeded
from ..types import Metrics, Usage


class BudgetTracker:
    """累计资源用量并在超限时抛异常。

    参数：
        config: 提供各维度上限。
        clock: 单调时钟，测试可注入假时钟。
        started_at: 起始时刻，``None`` 表示构造时开始计时。
    """

    def __init__(
        self,
        config: Config,
        *,
        clock: Callable[[], float] = time.monotonic,
        started_at: float | None = None,
    ) -> None:
        self.config = config
        self.metrics = Metrics()
        self._clock = clock
        self._started = clock() if started_at is None else started_at

    def elapsed(self) -> float:
        """返回自开始以来的秒数（单调时钟）。"""
        return self._clock() - self._started

    def check(self) -> None:
        """检查全部维度，任一超限抛 :class:`BudgetExceeded`。

        异常：
            BudgetExceeded: 步数、模型调用数、token 数或墙钟时间达到上限。
        """
        self.metrics.seconds = self.elapsed()
        checks = (
            ("max_steps", float(self.config.max_steps), float(self.metrics.steps)),
            ("max_llm_calls", float(self.config.max_llm_calls), float(self.metrics.llm_calls)),
            ("max_tokens", float(self.config.max_tokens), float(self.metrics.total_tokens)),
            ("max_seconds", float(self.config.max_seconds), self.metrics.seconds),
        )
        for name, limit, used in checks:
            if used >= limit:
                raise BudgetExceeded(name, limit, used)

    def record_llm(self, usage: Usage) -> None:
        """记录一次成功的模型补全及其 token 用量。"""
        self.metrics.llm_calls += 1
        self.metrics.prompt_tokens += usage.prompt_tokens
        self.metrics.completion_tokens += usage.completion_tokens

    def record_tool(self, ok: bool) -> None:
        """记录一次工具调用。"""
        self.metrics.tool_calls += 1
        if not ok:
            self.metrics.tool_failures += 1

    def record_step(self) -> None:
        """记录一次步骤执行（重试也计入，重试消耗预算）。"""
        self.metrics.steps += 1

    def record_replan(self) -> None:
        """记录一次重规划。"""
        self.metrics.replans += 1

    def record_retry(self) -> None:
        """记录一次步骤重试。"""
        self.metrics.retries += 1

    def record_lesson(self) -> None:
        """记录一条写入反思记忆的教训。"""
        self.metrics.lessons += 1

    def record_format_repair(self) -> None:
        """记录一次模型输出格式修复重试。"""
        self.metrics.format_repairs += 1

    def record_safety_block(self) -> None:
        """记录一次被安全策略拒绝的工具调用。"""
        self.metrics.safety_blocks += 1

    def finish(self) -> Metrics:
        """结束计时并返回累计指标。"""
        self.metrics.seconds = self.elapsed()
        return self.metrics
