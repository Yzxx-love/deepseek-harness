"""脚本化离线模型提供者（MockLLM）。

存在理由：在没有 API key、不能联网、且要求确定性的环境里，仍能验证**运行时机制**
（计划-执行-评审-反思-重规划-记忆写入-安全拦截-预算终止）是否正确。

它**不衡量能力**：脚本是预先写好的动作序列，不是模型推理。任何用 mock 得到的
"成功率"只能说明运行时跑通了，不能说明智能体变强了。真实能力评测必须用
``--provider deepseek``，见 ``docs/04-evaluation.md``。

脚本格式：``{"planner": [...], "actor": [...], "critic_step": [...],
"critic_final": [...], "reflector": [...]}``。每轮按角色名从对应队列弹出下一项；
队列耗尽后使用该角色的保守默认响应。
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from ..types import LLMResponse, Message, Usage

ROLE_MARKER = "[agi-lab:role="


def detect_role(messages: Sequence[Message]) -> str:
    """从系统消息的角色标记中提取角色名。

    参数：
        messages: 对话消息序列。

    返回：
        角色名；找不到标记时返回 ``"any"``。
    """
    for message in messages:
        if message.role != "system":
            continue
        marker = message.content.find(ROLE_MARKER)
        if marker < 0:
            continue
        start = marker + len(ROLE_MARKER)
        end = message.content.find("]", start)
        if end > start:
            return message.content[start:end].strip()
    return "any"


def _default_response(role: str) -> dict[str, Any]:
    if role == "planner":
        return {
            "goal": "（离线 mock 默认目标）",
            "steps": [
                {
                    "id": "s1",
                    "description": "直接执行目标",
                    "success_criteria": "给出最终答案",
                    "depends_on": [],
                }
            ],
            "notes": "离线 mock 默认单步计划",
        }
    if role == "actor":
        return {
            "thought": "离线 mock 没有为该任务提供脚本",
            "action": {"type": "final", "answer": "离线 mock 未提供该任务的脚本，无法继续执行。"},
        }
    if role == "critic_step":
        return {"ok": True, "missing": [], "reason": "离线 mock 默认通过子目标评审"}
    if role == "critic_final":
        return {
            "complete": False,
            "confidence": 0.0,
            "answer": "",
            "missing": ["离线 mock 没有目标达成的证据"],
            "reason": "离线 mock 默认不判定目标完成",
        }
    if role == "reflector":
        return {
            "failure_type": "unknown",
            "diagnosis": "离线 mock 默认诊断",
            "lesson": "模型输出无法解析时，先确认只输出单个 JSON 对象",
            "plan_patch": {"remove": [], "add": []},
        }
    return {"ok": True}


class MockLLM:
    """按角色队列回放脚本的确定性 provider。

    参数：
        script: 角色名 → 响应对象队列；``None`` 表示全部使用默认响应。
        name: provider 名称，默认 ``"mock"``。
    """

    name = "mock"

    def __init__(self, script: Mapping[str, Sequence[Mapping[str, Any]]] | None = None, *, name: str = "mock") -> None:
        self.name = name
        self._queues: dict[str, list[dict[str, Any]]] = {
            role: [copy.deepcopy(dict(item)) for item in items] for role, items in (script or {}).items()
        }
        self.calls: list[dict[str, Any]] = []

    def remaining(self) -> dict[str, int]:
        """返回各角色队列剩余条目数，用于测试断言脚本被完整消费。"""
        return {role: len(items) for role, items in self._queues.items()}

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        """按角色弹出下一条脚本响应，构造确定性的 :class:`LLMResponse`。

        参数同 :class:`agi_lab.types.LLMProvider.complete`；``json_mode`` 与
        ``timeout`` 被忽略（mock 永远返回合法 JSON 且瞬时完成）。
        """
        role = detect_role(messages)
        self.calls.append({"role": role, "messages": len(messages)})
        payload = self._next(role)
        text = json.dumps(payload, ensure_ascii=False)
        prompt_chars = sum(len(message.content) for message in messages)
        return LLMResponse(
            text=text,
            model=self.name,
            usage=Usage(prompt_tokens=max(1, prompt_chars // 3), completion_tokens=max(1, len(text) // 3)),
            finish_reason="stop",
        )

    def _next(self, role: str) -> dict[str, Any]:
        queue = self._queues.get(role)
        if queue:
            return queue.pop(0)
        fallback = self._queues.get("any")
        if fallback:
            return fallback.pop(0)
        return _default_response(role)
