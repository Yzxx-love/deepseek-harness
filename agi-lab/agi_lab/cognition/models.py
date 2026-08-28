"""认知层的结构化判定对象。

三类对象对应三种模型评审输出：单步评审、目标终审、失败反思。
解析严格（字段缺失或类型不符抛 ``LLMResponseFormatError``），坏输出由
:class:`agi_lab.cognition.client.StructuredClient` 触发一次修复重试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..errors import LLMResponseFormatError
from ..types import JSONObject

FAILURE_TYPES = (
    "tool_error",
    "wrong_plan",
    "insufficient_info",
    "format",
    "safety",
    "budget",
    "unknown",
)


def _as_str_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    raise LLMResponseFormatError(f"期望字符串数组，收到 {type(value).__name__}")


@dataclass(frozen=True)
class StepVerdict:
    """单步评审结论：该子目标是否达成。"""

    ok: bool
    missing: tuple[str, ...] = ()
    reason: str = ""

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "StepVerdict":
        """从 critic_step 角色的 JSON 输出解析。

        参数：
            obj: 含 ``ok``（必需）、``missing``、``reason`` 的对象。

        返回：
            :class:`StepVerdict`。

        异常：
            LLMResponseFormatError: ``ok`` 缺失或不是布尔值。
        """
        if not isinstance(obj, Mapping):
            raise LLMResponseFormatError("critic_step 输出不是 JSON 对象")
        raw_ok = obj.get("ok")
        if not isinstance(raw_ok, bool):
            raise LLMResponseFormatError("critic_step 输出缺少布尔字段 ok")
        return cls(ok=raw_ok, missing=_as_str_list(obj.get("missing")), reason=str(obj.get("reason") or "").strip())


@dataclass(frozen=True)
class FinalVerdict:
    """目标终审结论：整个任务是否完成，以及最终答案。"""

    complete: bool
    confidence: float = 0.5
    answer: str = ""
    missing: tuple[str, ...] = ()
    reason: str = ""

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "FinalVerdict":
        """从 critic_final 角色的 JSON 输出解析。

        参数：
            obj: 含 ``complete``（必需）、``confidence``、``answer``、``missing`` 的对象。

        返回：
            :class:`FinalVerdict`；``confidence`` 会被裁剪到 [0, 1]。

        异常：
            LLMResponseFormatError: ``complete`` 缺失或不是布尔值。
        """
        if not isinstance(obj, Mapping):
            raise LLMResponseFormatError("critic_final 输出不是 JSON 对象")
        raw_complete = obj.get("complete")
        if not isinstance(raw_complete, bool):
            raise LLMResponseFormatError("critic_final 输出缺少布尔字段 complete")
        confidence = 0.5
        raw_confidence = obj.get("confidence")
        if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
            confidence = min(1.0, max(0.0, float(raw_confidence)))
        return cls(
            complete=raw_complete,
            confidence=confidence,
            answer=str(obj.get("answer") or "").strip(),
            missing=_as_str_list(obj.get("missing")),
            reason=str(obj.get("reason") or "").strip(),
        )


@dataclass(frozen=True)
class Reflection:
    """失败反思结论：失败分类、诊断、可迁移教训与计划补丁。

    ``lesson`` 是跨任务记忆的写入对象，要求是通用经验而非任务细节复述。
    ``plan_patch`` 形如 ``{"remove": ["s2"], "add": [{"id": "s3", ...}]}``，
    由 :func:`agi_lab.cognition.planner.apply_patch` 应用。
    """

    failure_type: str
    diagnosis: str
    lesson: str
    plan_patch: JSONObject = field(default_factory=lambda: {"remove": [], "add": []})

    @property
    def has_patch(self) -> bool:
        """补丁是否包含任何增删。"""
        remove = self.plan_patch.get("remove") or []
        add = self.plan_patch.get("add") or []
        return bool(remove) or bool(add)

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Reflection":
        """从 reflector 角色的 JSON 输出解析。

        参数：
            obj: 含 ``failure_type``、``diagnosis``、``lesson``、``plan_patch`` 的对象。

        返回：
            :class:`Reflection`；未知 ``failure_type`` 归一为 ``unknown``，
            缺失 ``plan_patch`` 归一为空补丁。

        异常：
            LLMResponseFormatError: 缺少 ``diagnosis`` 或 ``lesson``。
        """
        if not isinstance(obj, Mapping):
            raise LLMResponseFormatError("reflector 输出不是 JSON 对象")
        diagnosis = str(obj.get("diagnosis") or "").strip()
        lesson = str(obj.get("lesson") or "").strip()
        if not diagnosis:
            raise LLMResponseFormatError("reflector 输出缺少 diagnosis")
        failure_type = str(obj.get("failure_type") or "unknown").strip()
        if failure_type not in FAILURE_TYPES:
            failure_type = "unknown"
        raw_patch = obj.get("plan_patch")
        patch: JSONObject = {"remove": [], "add": []}
        if isinstance(raw_patch, Mapping):
            remove = [str(item) for item in (raw_patch.get("remove") or [])]
            add = [dict(item) for item in (raw_patch.get("add") or []) if isinstance(item, Mapping)]
            patch = {"remove": remove, "add": add}
        return cls(failure_type=failure_type, diagnosis=diagnosis, lesson=lesson, plan_patch=patch)
