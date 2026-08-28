"""LLM 层：provider 基类、OpenAI 兼容实现与配置驱动工厂。

跨模块只依赖 :class:`agi_lab.types.LLMProvider`；本包导出具体实现与
:func:`build_llm`，由运行时按 :class:`agi_lab.config.Config` 选择 mock 或真实 provider。
"""

from __future__ import annotations

from ..config import Config
from ..errors import ConfigError
from ..types import LLMProvider
from .base import BaseLLM, RetryPolicy
from .openai_compat import OpenAICompatibleLLM

__all__ = ["BaseLLM", "RetryPolicy", "OpenAICompatibleLLM", "build_llm"]


def build_llm(config: Config) -> LLMProvider:
    """按 ``config.provider`` 构造模型 provider。

    参数：
        config: 已加载并校验的运行配置。

    返回：
        ``provider="mock"`` 时返回脚本化 :class:`MockLLM`（``script=None``，离线运行）；
        ``provider="deepseek"`` 时返回 :class:`OpenAICompatibleLLM`。``MockLLM`` 在函数内
        延迟导入，避免 ``llm`` 包在 mock 模块导入期形成循环依赖。

    异常：
        ConfigError: provider 未知，或 ``provider="deepseek"`` 缺少 api_key。
    """
    if config.provider == "mock":
        from .mock import MockLLM

        return MockLLM(script=None)
    if config.provider == "deepseek":
        if not config.api_key:
            raise ConfigError("provider=deepseek 需要 DEEPSEEK_API_KEY（或 --api-key）")
        return OpenAICompatibleLLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
            default_temperature=config.temperature,
            default_max_tokens=config.max_output_tokens,
        )
    raise ConfigError(f"未知 provider：{config.provider!r}")
