"""Agent 运行时：双层循环（计划执行 + 内层 ReAct）。

一次 ``run`` 的完整流程：

1. 写入任务 fixture，召回跨任务教训，生成计划；
2. 外层按依赖执行计划步骤，内层每轮做一个动作（工具/终答/重规划/提问）；
3. 每步结束用评审器判定；未过则重试，重试耗尽后反思 → 计划补丁 → 重规划；
4. 全部步骤收敛后做目标终审，按结果给出 ``SUCCESS`` / ``FAILED`` 等状态；
5. 无论成功失败都写 trace 与报告，并 flush 跨任务记忆。

预算检查在每个循环入口执行，超限抛 :class:`agi_lab.errors.BudgetExceeded`，
run 以 ``BUDGET_EXHAUSTED`` 结束并保留部分结果。未知异常只终止本次 run 并记录
traceback，不向上抛——批量评测不应被单次 run 的异常打断。
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..cognition.client import StructuredClient
from ..cognition.models import FinalVerdict, StepVerdict
from ..cognition.planner import Critic, Planner, Reflector, apply_patch
from ..cognition.prompts import actor_messages
from ..config import Config
from ..errors import (
    AgiLabError,
    BudgetExceeded,
    LLMError,
    LLMResponseFormatError,
    PlanFormatError,
)
from ..memory.hub import MemoryHub
from ..tools.base import ToolRegistry
from ..tools.pathguard import PathGuard
from ..types import (
    Action,
    ActionType,
    AnswerFn,
    MemoryKind,
    Observation,
    Plan,
    PlanStep,
    RunResult,
    RunStatus,
    StepStatus,
    TaskSpec,
)
from .budget import BudgetTracker
from .safety import PolicyGuard
from .tracing import Tracer, write_report

_MAX_OBSERVATION_RENDER = 1600
_DEFAULT_MAX_PLAN_STEPS = 6


@dataclass
class StepOutcome:
    """单步执行结果，供外层循环路由。"""

    kind: str  # done | failed | replan | blocked
    answer: str = ""
    failure: str = ""
    missing: tuple[str, ...] = ()
    summary: str = ""
    observations: tuple[Observation, ...] = field(default_factory=tuple)

    @classmethod
    def done(cls, *, answer: str, summary: str, observations: Sequence[Observation]) -> "StepOutcome":
        """构造"步骤完成"结果。"""
        return cls(kind="done", answer=answer, summary=summary, observations=tuple(observations))

    @classmethod
    def failed(
        cls,
        *,
        failure: str,
        missing: Sequence[str] = (),
        answer: str = "",
        observations: Sequence[Observation] = (),
    ) -> "StepOutcome":
        """构造"步骤失败"结果。"""
        return cls(
            kind="failed",
            failure=failure,
            missing=tuple(missing),
            answer=answer,
            observations=tuple(observations),
        )


class Agent:
    """通用智能体运行时。

    参数：
        config: 运行配置（预算、审批模式、密钥脱敏等）。
        llm: 模型提供者，离线 mock 与真实 provider 同一接口。
        tools: 已构建的工具注册表（注册内容决定能力面与安全面）。
        memory: 记忆门面（跨 run 共享实例即为持续学习）。
        workspace: 工具可访问的目录根，fixture 也写在这里。
        run_dir: 本次 run 的产物目录（trace.jsonl、report.md）。
        answer_fn: 人类应答回调，用于处理 ``ASK`` 动作；``None`` 时遇到提问即
            ``BLOCKED``。
        tracer: 已构造的追踪器；``None`` 时在 ``run`` 中创建并负责关闭。
        run_id: 指定 run 标识（测试用），``None`` 时自动生成。
        clock: 单调时钟，测试可注入。
    """

    def __init__(
        self,
        *,
        config: Config,
        llm: Any,
        tools: ToolRegistry,
        memory: MemoryHub,
        workspace: Path,
        run_dir: Path,
        answer_fn: AnswerFn | None = None,
        tracer: Tracer | None = None,
        run_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.workspace = Path(workspace)
        self.run_dir = Path(run_dir)
        self.answer_fn = answer_fn
        self.tracer = tracer
        self.run_id = run_id
        self.clock = clock
        self.policy = PolicyGuard(config)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, task: TaskSpec) -> RunResult:
        """执行一个任务并返回结果（永不向调用方抛异常）。

        参数：
            task: 任务定义。

        返回：
            :class:`RunResult`；``success`` 字段保持 ``False``，通过与否由评测层的
            独立 grader 判定（见 ``docs/04-evaluation.md``）。
        """
        run_id = self.run_id or _new_run_id()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        tracer = self.tracer or Tracer(self.run_dir, run_id=run_id, echo=self.config.verbose)
        owns_tracer = self.tracer is None
        budget = BudgetTracker(self.config, clock=self.clock)
        client = StructuredClient(
            self.llm,
            on_usage=budget.record_llm,
            on_repair=budget.record_format_repair,
        )
        plan_limit = min(self.config.max_steps, _DEFAULT_MAX_PLAN_STEPS)
        planner = Planner(client, max_steps=plan_limit)
        reflector = Reflector(client)
        critic = Critic(client)

        plan: Plan | None = None
        answer = ""
        stop_reason = ""
        status = RunStatus.ERROR
        error: str | None = None
        try:
            self._prepare_fixtures(task, tracer)
            self.memory.working.clear()
            self.memory.set_goal(task.goal)
            tracer.event(
                "run_start",
                task=task.to_json(),
                provider=self.llm.name,
                config=self.config.redacted(),
            )
            lessons = self.memory.search(
                task.goal,
                k=3,
                kinds=(MemoryKind.REFLECTION, MemoryKind.SEMANTIC),
            )
            tracer.event(
                "memory_recall",
                lessons=[{"kind": hit.item.kind.value, "text": hit.item.text, "score": round(hit.score, 4)} for hit in lessons],
            )
            plan = planner.plan(
                task.goal,
                tools=self.tools.specs(),
                lessons=lessons,
                memory_context=self.memory.render_context(query=task.goal, max_chars=1500),
                max_steps=min(task.max_steps or self.config.max_steps, plan_limit),
            )
            self.memory.set_plan(plan)
            tracer.event("plan", plan=plan.to_json(), version=plan.version)
            status, answer, stop_reason, plan = self._loop(
                task=task,
                plan=plan,
                client=client,
                critic=critic,
                reflector=reflector,
                budget=budget,
                tracer=tracer,
            )
        except BudgetExceeded as exc:
            status = RunStatus.BUDGET_EXHAUSTED
            stop_reason = str(exc)
            tracer.event("budget_exhausted", reason=str(exc), metrics=budget.metrics.as_dict())
        except LLMError as exc:
            status = RunStatus.ERROR
            error = f"{type(exc).__name__}: {exc}"
            tracer.event("error", where="llm", message=error)
        except AgiLabError as exc:
            status = RunStatus.ERROR
            error = f"{type(exc).__name__}: {exc}"
            tracer.event("error", where="agi-lab", message=error)
        except Exception as exc:  # noqa: BLE001 - 单次 run 的未知异常不得打断批量评测
            status = RunStatus.ERROR
            error = f"{type(exc).__name__}: {exc}"
            tracer.event("error", where="unexpected", message=error, traceback=traceback.format_exc(limit=8))
        finally:
            metrics = budget.finish()
            result = RunResult(
                run_id=run_id,
                task_id=task.id,
                status=status,
                success=False,
                answer=self.policy.redact(answer),
                metrics=metrics,
                stop_reason=stop_reason,
                error=error,
                trace_path=str(tracer.path),
                plan=plan,
            )
            tracer.event("run_end", status=status.value, answer=result.answer, metrics=metrics.as_dict())
            result.report_path = str(write_report(self.run_dir, result, tracer.events()))
            self.memory.flush()
            if owns_tracer:
                tracer.close()
        return result

    # ------------------------------------------------------------------
    # 外层循环：计划执行与重规划
    # ------------------------------------------------------------------

    def _loop(
        self,
        *,
        task: TaskSpec,
        plan: Plan,
        client: StructuredClient,
        critic: Critic,
        reflector: Reflector,
        budget: BudgetTracker,
        tracer: Tracer,
    ) -> tuple[RunStatus, str, str, Plan]:
        answer = ""
        while True:
            budget.check()
            step = plan.next_ready()
            if step is None:
                if plan.deadlocked():
                    tracer.event("plan_deadlock", plan=plan.to_json())
                    updated = self._reflect_and_replan(
                        task=task,
                        plan=plan,
                        step=None,
                        failure="计划死锁：剩余步骤的依赖无法满足（依赖的步骤已失败或被删除）",
                        attempts=0,
                        reflector=reflector,
                        budget=budget,
                        tracer=tracer,
                    )
                    if updated is None:
                        return RunStatus.FAILED, answer, "计划死锁且重规划失败", plan
                    plan = updated
                    continue
                if any(item.status is StepStatus.FAILED for item in plan.steps):
                    return RunStatus.FAILED, answer, "存在无法恢复的失败步骤", plan
                final_answer = answer or _fallback_answer(plan)
                verdict: FinalVerdict = critic.judge_final(
                    goal=task.goal,
                    plan=plan,
                    answer=final_answer,
                    observations=self.memory.working.observations(limit=20),
                )
                tracer.event(
                    "critic_final",
                    complete=verdict.complete,
                    confidence=verdict.confidence,
                    missing=list(verdict.missing),
                    reason=verdict.reason,
                )
                if verdict.complete:
                    return RunStatus.SUCCESS, verdict.answer or final_answer, "", plan
                reason = "；".join(verdict.missing) or verdict.reason or "未说明"
                return RunStatus.FAILED, verdict.answer or final_answer, f"终审未通过：{reason}", plan

            budget.record_step()
            step.status = StepStatus.RUNNING
            step.attempts += 1
            tracer.event(
                "step_start",
                step=step.id,
                description=step.description,
                attempt=step.attempts,
                criteria=step.success_criteria,
            )
            outcome = self._execute_step(
                task=task,
                plan=plan,
                step=step,
                client=client,
                critic=critic,
                budget=budget,
                tracer=tracer,
            )
            if outcome.kind == "blocked":
                tracer.event("step_blocked", step=step.id, question=outcome.failure)
                return RunStatus.BLOCKED, answer, outcome.failure, plan
            if outcome.kind == "done":
                step.status = StepStatus.DONE
                step.result = outcome.summary
                if outcome.answer:
                    answer = outcome.answer
                self.memory.add_episode(
                    f"[{step.id}] {outcome.summary}",
                    step_id=step.id,
                    tags=("done",),
                )
                tracer.event("step_done", step=step.id, result=outcome.summary)
                continue
            failure = outcome.failure or "步骤失败（无详情）"
            tracer.event(
                "step_failed",
                step=step.id,
                failure=failure,
                missing=list(outcome.missing),
                attempts=step.attempts,
            )
            if outcome.kind == "failed" and step.attempts < self.config.max_attempts:
                step.status = StepStatus.PENDING
                budget.record_retry()
                note = Observation(
                    source="runtime",
                    content=f"第 {step.attempts} 次尝试失败：{failure}。请改变策略后重试。",
                    ok=False,
                    step_id=step.id,
                    untrusted=False,
                )
                self.memory.working.record(note)
                tracer.event("retry", step=step.id, attempts=step.attempts, failure=failure)
                continue
            updated = self._reflect_and_replan(
                task=task,
                plan=plan,
                step=step,
                failure=failure,
                attempts=step.attempts,
                reflector=reflector,
                budget=budget,
                tracer=tracer,
            )
            if updated is None:
                return RunStatus.FAILED, answer, f"重规划失败，最后失败：{failure}", plan
            plan = updated

    def _reflect_and_replan(
        self,
        *,
        task: TaskSpec,
        plan: Plan,
        step: PlanStep | None,
        failure: str,
        attempts: int,
        reflector: Reflector,
        budget: BudgetTracker,
        tracer: Tracer,
    ) -> Plan | None:
        """反思失败并应用计划补丁；返回新计划，失败返回 ``None``。"""
        if budget.metrics.replans >= self.config.max_replans:
            tracer.event("replan_exhausted", replans=budget.metrics.replans, limit=self.config.max_replans)
            return None
        reflection = reflector.reflect(
            goal=task.goal,
            plan=plan,
            step=step,
            failure=failure,
            attempts=attempts,
            observations=self.memory.working.observations(limit=12),
        )
        budget.record_lesson()
        self.memory.add_lesson(
            reflection.lesson,
            tags=(reflection.failure_type,),
            task_id=task.id,
        )
        tracer.event(
            "reflection",
            step=step.id if step is not None else None,
            failure_type=reflection.failure_type,
            diagnosis=reflection.diagnosis,
            lesson=reflection.lesson,
            patch=reflection.plan_patch,
        )
        try:
            updated = apply_patch(plan, reflection.plan_patch, max_steps=self.config.max_steps)
        except PlanFormatError as exc:
            tracer.event("replan_failed", error=str(exc), patch=reflection.plan_patch)
            return None
        budget.record_replan()
        self.memory.set_plan(updated)
        tracer.event("plan_update", version=updated.version, plan=updated.to_json(), notes=updated.notes)
        return updated

    # ------------------------------------------------------------------
    # 内层循环：ReAct 单步执行
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        *,
        task: TaskSpec,
        plan: Plan,
        step: PlanStep,
        client: StructuredClient,
        critic: Critic,
        budget: BudgetTracker,
        tracer: Tracer,
    ) -> StepOutcome:
        observations: list[Observation] = []
        answer = ""
        last_failure = ""
        inner_turns = task.inner_steps or self.config.inner_steps
        lessons = self.memory.search(
            f"{task.goal} {step.description}",
            k=3,
            kinds=(MemoryKind.REFLECTION, MemoryKind.SEMANTIC),
        )
        signatures: list[str] = []
        for turn in range(1, inner_turns + 1):
            budget.check()
            messages = actor_messages(
                goal=task.goal,
                plan=plan,
                step=step,
                tools=self.tools.specs(),
                lessons=lessons,
                memory_context=self.memory.render_context(
                    query=f"{task.goal} {step.description}",
                    max_chars=2200,
                ),
                observations=observations[-8:],
                turn=turn,
                attempt=step.attempts,
            )
            try:
                action: Action = client.call(messages, role="actor", validator=Action.from_json)
            except LLMResponseFormatError as exc:
                last_failure = f"动作输出无法解析：{exc}"
                tracer.event("actor_format_error", step=step.id, turn=turn, message=str(exc))
                break
            tracer.event("action", step=step.id, turn=turn, action=action.to_json())

            if action.type is ActionType.ASK:
                question = action.question or ""
                reply = self.answer_fn(question) if self.answer_fn is not None else ""
                if not reply.strip():
                    return StepOutcome(kind="blocked", failure=f"需要人类澄清但无人应答：{question}")
                self.memory.add_fact(reply, tags=("human",), source="human")
                observation = Observation(
                    source="human",
                    content=self.policy.redact(reply),
                    ok=True,
                    step_id=step.id,
                    untrusted=False,
                )
                observations.append(observation)
                self.memory.working.record(observation)
                tracer.event("human_answer", question=question, answer=observation.content)
                continue
            if action.type is ActionType.REPLAN:
                return StepOutcome(kind="replan", failure=action.reason or "模型请求重规划")
            if action.type is ActionType.FINAL:
                answer = self.policy.redact(action.answer or "")
                tracer.event("actor_final", step=step.id, turn=turn, answer=answer)
                break

            call = action.tool_call
            if call is None:  # Action 构造期已校验，此处仅防御性兜底
                last_failure = "工具动作缺少 tool_call"
                break
            signature = f"{call.tool}:{json.dumps(dict(call.arguments), sort_keys=True, ensure_ascii=False)}"
            signatures.append(signature)
            repeats = signatures.count(signature)
            if repeats >= 3:
                last_failure = f"检测到重复工具调用 {repeats} 次：{signature[:120]}"
                note = Observation(
                    source="runtime",
                    content=last_failure + "。请改变策略。",
                    ok=False,
                    step_id=step.id,
                    untrusted=False,
                )
                observations.append(note)
                self.memory.working.record(note)
                tracer.event("loop_detected", step=step.id, tool=call.tool, repeats=repeats)
                if repeats >= 4:
                    break
                continue

            budget.check()
            result = self.tools.execute(call)
            budget.record_tool(result.ok)
            if self.policy.is_safety_block(result):
                budget.record_safety_block()
                tracer.event(
                    "safety_event",
                    step=step.id,
                    tool=call.tool,
                    arguments=dict(call.arguments),
                    error=result.error,
                )
            rendered = self.policy.redact(result.render(max_chars=_MAX_OBSERVATION_RENDER))
            observation = Observation(
                source=call.tool,
                content=rendered,
                ok=result.ok,
                step_id=step.id,
                data=result.metadata,
            )
            observations.append(observation)
            self.memory.working.record(observation)
            self.memory.add_episode(
                f"[{step.id}] {call.tool}({_short(json.dumps(dict(call.arguments), ensure_ascii=False), 80)}) → {_short(rendered, 160)}",
                step_id=step.id,
                tags=(call.tool, "tool"),
            )
            tracer.event(
                "observation",
                step=step.id,
                tool=call.tool,
                ok=result.ok,
                observation=observation.to_json(),
            )

        verdict_step: StepVerdict = critic.judge_step(
            goal=task.goal,
            step=step,
            observations=observations,
        )
        tracer.event(
            "critic_step",
            step=step.id,
            ok=verdict_step.ok,
            missing=list(verdict_step.missing),
            reason=verdict_step.reason,
        )
        if verdict_step.ok:
            return StepOutcome.done(
                answer=answer,
                summary=self.policy.redact(_summarize(observations)),
                observations=observations,
            )
        failure = last_failure or "；".join(verdict_step.missing) or verdict_step.reason or "步骤未通过评审"
        return StepOutcome.failed(
            failure=failure,
            missing=verdict_step.missing,
            answer=answer,
            observations=observations,
        )

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    def _prepare_fixtures(self, task: TaskSpec, tracer: Tracer) -> None:
        """把任务 fixture 写入工作区，路径越界直接失败。"""
        if not task.initial_files:
            return
        guard = PathGuard(self.workspace)
        for relative, content in task.initial_files.items():
            target = guard.resolve(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            tracer.event("fixture", path=str(relative), chars=len(content))


def _new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _short(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "…"


def _summarize(observations: Sequence[Observation]) -> str:
    if not observations:
        return "（本步无观测）"
    last = observations[-1]
    head = f"{last.source}{' 成功' if last.ok else ' 失败'}"
    return f"{head}：{_short(last.content, 240)}"


def _fallback_answer(plan: Plan) -> str:
    results = [f"- {step.id}: {step.result}" for step in plan.steps if step.result]
    return "\n".join(results) if results else ""
