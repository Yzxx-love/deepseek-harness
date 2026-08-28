"""计划生成与补丁应用。

规划器只产出计划；重规划不是"整份重写"，而是对现有计划应用反思给出的最小补丁
（删除失败步骤、追加替代步骤）。这样计划漂移可追踪，也避免模型在重写时丢掉已完成信息。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from ..errors import PlanFormatError
from ..types import MemoryHit, Plan, PlanStep, StepStatus, ToolSpec
from .client import StructuredClient
from .models import Reflection
from .prompts import planner_messages, reflector_messages

DEFAULT_MAX_PLAN_STEPS = 6


class Planner:
    """把目标分解为步骤计划，并对失败计划应用补丁。

    参数：
        client: 结构化模型调用封装。
        max_steps: 单份计划的步骤上限，同时是 ``plan`` 的默认上限。
    """

    def __init__(self, client: StructuredClient, *, max_steps: int = DEFAULT_MAX_PLAN_STEPS) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        self.client = client
        self.max_steps = max_steps

    def plan(
        self,
        goal: str,
        *,
        tools: Sequence[ToolSpec],
        lessons: Sequence[MemoryHit] = (),
        memory_context: str = "",
        max_steps: int | None = None,
    ) -> Plan:
        """生成初始计划。

        参数：
            goal: 目标原文。
            tools: 可用工具。
            lessons: 跨任务教训。
            memory_context: 记忆上下文。
            max_steps: 本次允许的步骤上限，``None`` 用构造参数。

        返回：
            校验通过的 :class:`Plan`；步骤数超限时截断并在 ``notes`` 说明。

        异常：
            PlanFormatError: 模型计划无法解析（空步骤、重复 id、悬挂依赖）。
            LLMResponseFormatError: 重试后仍无合法 JSON。
        """
        limit = min(max_steps or self.max_steps, self.max_steps)
        plan = self.client.call(
            planner_messages(goal=goal, tools=tools, lessons=lessons, memory_context=memory_context, max_steps=limit),
            role="planner",
            validator=Plan.from_json,
        )
        return _fit_steps(plan, limit)


class Reflector:
    """把失败转化为诊断、可迁移教训与计划补丁。

    参数：
        client: 结构化模型调用封装。
        max_lesson_chars: 教训长度上限，超出截断（教训要短才能被高效复用）。
    """

    def __init__(self, client: StructuredClient, *, max_lesson_chars: int = 160) -> None:
        self.client = client
        self.max_lesson_chars = max_lesson_chars

    def reflect(
        self,
        *,
        goal: str,
        plan: Plan,
        step: PlanStep | None,
        failure: str,
        attempts: int,
        observations: Sequence[Any] = (),
    ) -> Reflection:
        """分析一次失败。

        参数：
            goal: 目标原文。
            plan: 当前计划。
            step: 失败步骤；``None`` 表示计划级失败。
            failure: 失败原因文本。
            attempts: 已尝试次数。
            observations: 相关观测。

        返回：
            :class:`Reflection`。模型输出无法解析时返回保守回退（空补丁 +
            由失败原因合成的教训），避免一次坏输出直接终结整个 run。
        """
        from ..errors import LLMResponseFormatError

        try:
            reflection = self.client.call(
                reflector_messages(
                    goal=goal, plan=plan, step=step, failure=failure, attempts=attempts, observations=observations
                ),
                role="reflector",
                validator=Reflection.from_json,
            )
            return self._sanitize(reflection, failure)
        except LLMResponseFormatError:
            return Reflection(
                failure_type="unknown",
                diagnosis=f"反思输出无法解析，保留原始失败信息：{failure[:200]}",
                lesson=_synthesize_lesson(failure),
                plan_patch={"remove": [], "add": []},
            )

    def _sanitize(self, reflection: Reflection, failure: str) -> Reflection:
        lesson = reflection.lesson.strip()
        if len(lesson) > self.max_lesson_chars:
            lesson = lesson[: self.max_lesson_chars] + "…"
        if not lesson:
            lesson = _synthesize_lesson(failure)
        if not reflection.diagnosis:
            diagnosis = failure[:300]
        else:
            diagnosis = reflection.diagnosis
        return Reflection(
            failure_type=reflection.failure_type,
            diagnosis=diagnosis,
            lesson=lesson,
            plan_patch=reflection.plan_patch,
        )


def _synthesize_lesson(failure: str) -> str:
    excerpt = " ".join(failure.split())[:80]
    return f"出现同类失败时先核对工具参数与环境：{excerpt}"


class Critic:
    """步骤评审与目标终审。

    评审不可用（输出无法解析）时返回保守结论而非抛异常：一次坏的评审输出不应
    直接终止 run，而应记为失败证据交给反思环节。
    """

    def __init__(self, client: StructuredClient) -> None:
        self.client = client

    def judge_step(self, *, goal: str, step: PlanStep, observations: Sequence[Any]) -> Any:
        """判断子目标是否达成。"""
        from ..errors import LLMResponseFormatError
        from .models import StepVerdict
        from .prompts import critic_step_messages

        try:
            return self.client.call(
                critic_step_messages(goal=goal, step=step, observations=observations),
                role="critic_step",
                validator=StepVerdict.from_json,
            )
        except LLMResponseFormatError as exc:
            return StepVerdict(ok=False, missing=("评审输出无法解析",), reason=str(exc))

    def judge_final(self, *, goal: str, plan: Plan, answer: str, observations: Sequence[Any]) -> Any:
        """判断目标是否完成。"""
        from ..errors import LLMResponseFormatError
        from .models import FinalVerdict
        from .prompts import critic_final_messages

        try:
            verdict = self.client.call(
                critic_final_messages(goal=goal, plan=plan, answer=answer, observations=observations),
                role="critic_final",
                validator=FinalVerdict.from_json,
            )
        except LLMResponseFormatError as exc:
            return FinalVerdict(complete=False, confidence=0.0, missing=("评审输出无法解析",), reason=str(exc))
        if verdict.answer:
            return verdict
        if answer:
            return FinalVerdict(
                complete=verdict.complete,
                confidence=verdict.confidence,
                answer=answer,
                missing=verdict.missing,
                reason=verdict.reason,
            )
        return verdict


def _fit_steps(plan: Plan, limit: int) -> Plan:
    """把计划步骤数裁剪到 ``limit``，并检查裁剪后依赖仍完整。"""
    if len(plan.steps) <= limit:
        return plan
    kept = plan.steps[:limit]
    kept_ids = {step.id for step in kept}
    for step in kept:
        for dep in step.depends_on:
            if dep not in kept_ids:
                raise PlanFormatError(f"截断后步骤 {step.id} 依赖被丢弃的步骤 {dep}")
    notes = (plan.notes + f"（原计划 {len(plan.steps)} 步，按上限截断为 {limit} 步）").strip()
    return Plan(goal=plan.goal, steps=kept, version=plan.version, notes=notes)


def apply_patch(plan: Plan, patch: Mapping[str, Any], *, max_steps: int | None = None) -> Plan:
    """把反思补丁应用到计划，返回新计划（原计划不被修改）。

    语义：

    - ``remove`` 中的步骤被删除，依赖其的步骤级联删除（否则永远无法就绪）。
    - ``add`` 中的步骤追加到末尾，依赖只能引用存活步骤；id 冲突或依赖悬空抛错。
    - 存活步骤中状态为 ``FAILED`` 的重置为 ``PENDING`` 并清零尝试次数。
    - 删除后计划为空抛 :class:`PlanFormatError`（无效重规划）。

    参数：
        plan: 当前计划。
        patch: ``{"remove": [id], "add": [{...}]}``，缺键视为空。
        max_steps: 重规划后的步骤上限，``None`` 不限制。

    返回：
        版本号加一的新计划。

    异常：
        PlanFormatError: 补丁非法或结果计划为空。
    """
    raw_remove = patch.get("remove") or []
    raw_add = patch.get("add") or []
    if not isinstance(raw_remove, list) or not isinstance(raw_add, list):
        raise PlanFormatError("plan_patch 的 remove/add 必须是数组")
    remove: set[str] = {str(item) for item in raw_remove}
    # 级联删除依赖被删步骤的步骤
    changed = True
    existing = {step.id for step in plan.steps}
    while changed:
        changed = False
        for step in plan.steps:
            if step.id in remove:
                continue
            if any(dep in remove for dep in step.depends_on):
                remove.add(step.id)
                changed = True
    remove &= existing  # 忽略模型编造的 id
    surviving: list[PlanStep] = []
    for step in plan.steps:
        if step.id in remove:
            continue
        status = StepStatus.PENDING if step.status is StepStatus.FAILED else step.status
        attempts = 0 if step.status is StepStatus.FAILED else step.attempts
        surviving.append(
            PlanStep(
                id=step.id,
                description=step.description,
                success_criteria=step.success_criteria,
                depends_on=step.depends_on,
                status=status,
                attempts=attempts,
                result=step.result,
            )
        )
    live_ids = {step.id for step in surviving}
    added: list[PlanStep] = []
    for index, raw in enumerate(raw_add):
        if not isinstance(raw, Mapping):
            raise PlanFormatError("plan_patch.add 的每一项必须是对象")
        step_id = str(raw.get("id") or f"r{plan.version + 1}_{index + 1}").strip()
        if not step_id:
            raise PlanFormatError("新增步骤缺少 id")
        if step_id in live_ids:
            raise PlanFormatError(f"新增步骤 id 与现有步骤冲突：{step_id}")
        description = str(raw.get("description") or "").strip()
        if not description:
            raise PlanFormatError(f"新增步骤 {step_id} 缺少 description")
        depends_raw = raw.get("depends_on") or []
        if isinstance(depends_raw, str):
            depends_raw = [depends_raw]
        if not isinstance(depends_raw, list):
            raise PlanFormatError(f"新增步骤 {step_id} 的 depends_on 必须是数组")
        depends = tuple(str(item) for item in depends_raw)
        for dep in depends:
            if dep not in live_ids:
                raise PlanFormatError(f"新增步骤 {step_id} 依赖不存在的步骤 {dep}")
        added.append(
            PlanStep(
                id=step_id,
                description=description,
                success_criteria=str(raw.get("success_criteria") or "").strip(),
                depends_on=depends,
            )
        )
        live_ids.add(step_id)
    final_steps = surviving + added
    if not final_steps:
        raise PlanFormatError("重规划后计划为空")
    if max_steps is not None and len(final_steps) > max_steps:
        raise PlanFormatError(f"重规划后步骤数 {len(final_steps)} 超过上限 {max_steps}")
    notes = plan.notes
    if remove:
        notes = (notes + f" 移除步骤 {sorted(remove)}").strip()
    if added:
        notes = (notes + f" 新增步骤 {[step.id for step in added]}").strip()
    return Plan(goal=plan.goal, steps=final_steps, version=plan.version + 1, notes=notes)
