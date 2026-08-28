"""agi-lab 的共享数据契约与能力接口。

本模块是全局冻结的内核：``llm``、``memory``、``tools``、``cognition``、``runtime``、``eval``
都只从这里导入跨包类型。修改本文件必须同时更新 ``docs/02-architecture.md`` 的契约表。

设计约定：

- 跨模块传递的数据一律是 dataclass 或 Enum，不用裸 dict 传参。
- 能力接口是 ``typing.Protocol``（结构化类型），实现方无需继承，便于替换与测试。
- 时间统一用 Unix 秒（``time.time()``），便于 trace 排序与跨进程对齐。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

# ---------------------------------------------------------------------------
# JSON 别名
# ---------------------------------------------------------------------------

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    """计划步骤状态机：``PENDING → RUNNING → DONE | FAILED | SKIPPED``。"""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionType(str, Enum):
    """模型单轮决策的类型。"""

    TOOL = "tool"
    FINAL = "final"
    REPLAN = "replan"
    ASK = "ask"


class MemoryKind(str, Enum):
    """四层记忆类型。"""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    REFLECTION = "reflection"


class RunStatus(str, Enum):
    """一次 run 的终态。除 ``SUCCESS`` 外都必须携带 ``stop_reason``。"""

    SUCCESS = "success"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BLOCKED = "blocked"
    ERROR = "error"


# ---------------------------------------------------------------------------
# 模型调用
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    """一条模型消息。``role`` 取 ``system`` / ``user`` / ``assistant``。"""

    role: str
    content: str
    name: str | None = None

    def to_dict(self) -> JSONObject:
        """转换为 OpenAI 兼容的消息字典。"""
        payload: JSONObject = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class Usage:
    """token 用量。provider 不返回用量时全部为 0，预算据此保守估计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """输入与输出 token 之和。"""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    """一次模型补全的返回值。"""

    text: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    raw: JSONObject | None = None


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _empty_parameters() -> JSONObject:
    return {"type": "object", "properties": {}, "required": []}


@dataclass(frozen=True)
class ToolSpec:
    """工具的静态描述，用于提示词渲染与注册表校验。"""

    name: str
    description: str
    parameters: JSONObject = field(default_factory=_empty_parameters)
    side_effects: bool = False
    requires_approval: bool = False

    def render(self) -> str:
        """渲染为提示词中的单行工具说明。"""
        flags = []
        if self.side_effects:
            flags.append("有副作用")
        if self.requires_approval:
            flags.append("需审批")
        suffix = f" [{'/'.join(flags)}]" if flags else ""
        return f"- {self.name}({_render_params(self.parameters)}): {self.description}{suffix}"


def _render_params(parameters: JSONObject) -> str:
    props = parameters.get("properties")
    if not isinstance(props, Mapping):
        return ""
    required = parameters.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    parts: list[str] = []
    for name, schema in props.items():
        type_name = schema.get("type", "any") if isinstance(schema, Mapping) else "any"
        marker = "" if name in required_names else "?"
        parts.append(f"{name}{marker}: {type_name}")
    return ", ".join(parts)


@dataclass(frozen=True)
class ToolCall:
    """模型请求的一次工具调用。"""

    tool: str
    arguments: JSONObject = field(default_factory=dict)
    call_id: str = ""

    def to_json(self) -> JSONObject:
        """序列化为 trace 用的 JSON 对象。"""
        return {"tool": self.tool, "arguments": dict(self.arguments), "call_id": self.call_id}


@dataclass(frozen=True)
class ToolResult:
    """工具执行结果。失败用 ``ok=False`` 表达，不向上抛异常。"""

    ok: bool
    output: str = ""
    error: str | None = None
    metadata: JSONObject = field(default_factory=dict)

    def render(self, *, max_chars: int = 2000) -> str:
        """渲染为提示词可读文本，过长截断。"""
        head = "ok" if self.ok else "error"
        body = self.output if self.ok else (self.error or self.output or "unknown error")
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n…[截断，共 {len(body)} 字符]"
        return f"[{head}] {body}"


# ---------------------------------------------------------------------------
# 模型动作
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """模型单轮决策，对应 ``actor`` 角色的 JSON 输出。

    ``TOOL`` 必须带 ``tool_call``；``FINAL`` 必须带 ``answer``；``REPLAN`` 必须带
    ``reason``；``ASK`` 必须带 ``question``。构造期校验，非法组合抛
    :class:`agi_lab.errors.ActionFormatError`。
    """

    type: ActionType
    thought: str = ""
    tool_call: ToolCall | None = None
    answer: str | None = None
    reason: str | None = None
    question: str | None = None

    def __post_init__(self) -> None:
        from .errors import ActionFormatError

        if self.type is ActionType.TOOL and self.tool_call is None:
            raise ActionFormatError("action type=tool 缺少 tool_call")
        if self.type is ActionType.FINAL and not (self.answer or "").strip():
            raise ActionFormatError("action type=final 缺少 answer")
        if self.type is ActionType.REPLAN and not (self.reason or "").strip():
            raise ActionFormatError("action type=replan 缺少 reason")
        if self.type is ActionType.ASK and not (self.question or "").strip():
            raise ActionFormatError("action type=ask 缺少 question")

    @classmethod
    def call_tool(cls, tool: str, arguments: Mapping[str, Any] | None = None, *, thought: str = "") -> "Action":
        """构造一次工具调用动作。"""
        return cls(ActionType.TOOL, thought=thought, tool_call=ToolCall(tool, dict(arguments or {})))

    @classmethod
    def final(cls, answer: str, *, thought: str = "") -> "Action":
        """构造一次终结动作。"""
        return cls(ActionType.FINAL, thought=thought, answer=answer)

    @classmethod
    def replan(cls, reason: str, *, thought: str = "") -> "Action":
        """构造一次重规划动作。"""
        return cls(ActionType.REPLAN, thought=thought, reason=reason)

    @classmethod
    def ask(cls, question: str, *, thought: str = "") -> "Action":
        """构造一次向人类提问的动作。"""
        return cls(ActionType.ASK, thought=thought, question=question)

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Action":
        """严格解析 ``actor`` 角色的 JSON 对象。

        参数：
            obj: 至少包含 ``action`` 的对象；``action`` 可为对象或字符串。

        返回：
            校验通过的 :class:`Action`。

        异常：
            ActionFormatError: 缺少字段、类型未知或字段组合不合法。
        """
        from .errors import ActionFormatError

        if not isinstance(obj, Mapping):
            raise ActionFormatError(f"actor 输出不是 JSON 对象：{type(obj).__name__}")
        thought = str(obj.get("thought") or "").strip()
        raw_action: Any = obj.get("action")
        if raw_action is None and ("tool" in obj or "answer" in obj):
            raw_action = obj
        if isinstance(raw_action, str):
            raw_action = {"type": raw_action}
        if not isinstance(raw_action, Mapping):
            raise ActionFormatError("actor 输出缺少 action 对象")
        kind = str(raw_action.get("type") or raw_action.get("action") or "").strip().lower()
        if not kind:
            # 容忍省略 type 的简写形式：按字段推断动作类型
            if "tool" in raw_action or "arguments" in raw_action or "args" in raw_action:
                kind = "tool"
            elif "answer" in raw_action:
                kind = "final"
            elif "reason" in raw_action:
                kind = "replan"
            elif "question" in raw_action:
                kind = "ask"
        if kind == "tool":
            tool = str(raw_action.get("tool") or raw_action.get("name") or "").strip()
            if not tool:
                raise ActionFormatError("action type=tool 缺少 tool 名称")
            arguments = raw_action.get("arguments") or raw_action.get("args") or {}
            if isinstance(arguments, str):
                from .jsonutil import extract_json_object

                arguments = extract_json_object(arguments)
            if not isinstance(arguments, Mapping):
                raise ActionFormatError("action type=tool 的 arguments 必须是对象")
            return cls(ActionType.TOOL, thought=thought, tool_call=ToolCall(tool, dict(arguments)))
        if kind == "final":
            answer = raw_action.get("answer") or raw_action.get("content") or ""
            return cls(ActionType.FINAL, thought=thought, answer=str(answer))
        if kind == "replan":
            reason = raw_action.get("reason") or raw_action.get("why") or ""
            return cls(ActionType.REPLAN, thought=thought, reason=str(reason))
        if kind == "ask":
            question = raw_action.get("question") or raw_action.get("content") or ""
            return cls(ActionType.ASK, thought=thought, question=str(question))
        raise ActionFormatError(f"未知 action.type：{kind!r}")

    def to_json(self) -> JSONObject:
        """序列化为 trace 用的 JSON 对象。"""
        payload: JSONObject = {"type": self.type.value}
        if self.thought:
            payload["thought"] = self.thought
        if self.tool_call is not None:
            payload["tool"] = self.tool_call.tool
            payload["arguments"] = dict(self.tool_call.arguments)
        if self.answer is not None:
            payload["answer"] = self.answer
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.question is not None:
            payload["question"] = self.question
        return payload


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """计划中的一步。``depends_on`` 存步骤 id，依赖必须先 ``DONE`` 才可执行。"""

    id: str
    description: str
    success_criteria: str = ""
    depends_on: tuple[str, ...] = ()
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    result: str = ""

    def to_json(self) -> JSONObject:
        """序列化为 trace 用的 JSON 对象。"""
        return {
            "id": self.id,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "depends_on": list(self.depends_on),
            "status": self.status.value,
            "attempts": self.attempts,
            "result": self.result,
        }


@dataclass
class Plan:
    """任务分解结果。``version`` 在每次重规划后递增，便于追踪计划漂移。"""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    version: int = 1
    notes: str = ""

    def __post_init__(self) -> None:
        from .errors import PlanFormatError

        seen: set[str] = set()
        for step in self.steps:
            if not step.id:
                raise PlanFormatError("计划步骤缺少 id")
            if step.id in seen:
                raise PlanFormatError(f"计划步骤 id 重复：{step.id}")
            seen.add(step.id)
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in seen:
                    raise PlanFormatError(f"步骤 {step.id} 依赖不存在的步骤 {dep}")
                if dep == step.id:
                    raise PlanFormatError(f"步骤 {step.id} 依赖自身")

    def get(self, step_id: str) -> PlanStep | None:
        """按 id 查找步骤。"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def next_ready(self) -> PlanStep | None:
        """返回第一个依赖已全部 ``DONE`` 的 ``PENDING`` 步骤。"""
        done = {step.id for step in self.steps if step.status is StepStatus.DONE}
        for step in self.steps:
            if step.status is not StepStatus.PENDING:
                continue
            if all(dep in done for dep in step.depends_on):
                return step
        return None

    def deadlocked(self) -> bool:
        """是否存在无法推进的步骤（依赖失败或依赖成环）。"""
        if any(step.status is StepStatus.RUNNING for step in self.steps):
            return False
        return self.next_ready() is None and any(step.status is StepStatus.PENDING for step in self.steps)

    def mark(self, step_id: str, status: StepStatus, *, result: str = "") -> PlanStep:
        """更新步骤状态，返回更新后的步骤。"""
        from .errors import PlanFormatError

        step = self.get(step_id)
        if step is None:
            raise PlanFormatError(f"未知步骤 id：{step_id}")
        step.status = status
        if result:
            step.result = result
        return step

    def progress(self) -> tuple[int, int]:
        """返回 (已完成或跳过的步数, 总步数)。"""
        settled = sum(1 for s in self.steps if s.status in (StepStatus.DONE, StepStatus.SKIPPED))
        return settled, len(self.steps)

    def render(self, *, max_chars: int = 2000) -> str:
        """渲染为提示词中的计划文本，包含状态与依赖。"""
        done, total = self.progress()
        lines = [f"目标：{self.goal}", f"计划 v{self.version}（{done}/{total} 完成）"]
        for index, step in enumerate(self.steps, start=1):
            deps = f" ←依赖 {','.join(step.depends_on)}" if step.depends_on else ""
            criteria = f" | 验收：{step.success_criteria}" if step.success_criteria else ""
            lines.append(f"{index}. [{step.status.value}] {step.id}: {step.description}{deps}{criteria}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…[计划截断]"
        return text

    def to_json(self) -> JSONObject:
        """序列化为 trace 用的 JSON 对象。"""
        return {
            "goal": self.goal,
            "version": self.version,
            "notes": self.notes,
            "steps": [step.to_json() for step in self.steps],
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Plan":
        """从 JSON 对象重建计划。

        参数：
            obj: 含 ``goal`` 与 ``steps`` 的对象；``steps`` 每项为步骤对象。

        返回：
            校验通过的 :class:`Plan`。

        异常：
            PlanFormatError: 步骤缺字段、id 重复或依赖不存在。
        """
        from .errors import PlanFormatError

        if not isinstance(obj, Mapping):
            raise PlanFormatError("plan 输出不是 JSON 对象")
        goal = str(obj.get("goal") or "").strip()
        raw_steps = obj.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanFormatError("plan 输出缺少非空 steps 数组")
        steps: list[PlanStep] = []
        for index, raw in enumerate(raw_steps, start=1):
            if not isinstance(raw, Mapping):
                raise PlanFormatError(f"steps[{index - 1}] 不是对象")
            step_id = str(raw.get("id") or f"s{index}").strip()
            description = str(raw.get("description") or "").strip()
            if not description:
                raise PlanFormatError(f"步骤 {step_id} 缺少 description")
            deps_raw = raw.get("depends_on") or []
            if isinstance(deps_raw, str):
                deps_raw = [deps_raw]
            if not isinstance(deps_raw, list) or not all(isinstance(d, str) for d in deps_raw):
                raise PlanFormatError(f"步骤 {step_id} 的 depends_on 必须是字符串数组")
            steps.append(
                PlanStep(
                    id=step_id,
                    description=description,
                    success_criteria=str(raw.get("success_criteria") or "").strip(),
                    depends_on=tuple(deps_raw),
                )
            )
        notes = str(obj.get("notes") or "").strip()
        return cls(goal=goal, steps=steps, notes=notes)


# ---------------------------------------------------------------------------
# 观测
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """一次工具执行或运行时事件的观测结果。

    ``untrusted`` 表示内容来自模型可控的外部输入（文件内容、命令输出、网页），
    进入提示时必须包裹在隔离数据块中，不得被当作指令执行。
    """

    source: str
    content: str
    ok: bool = True
    step_id: str = ""
    data: JSONObject = field(default_factory=dict)
    untrusted: bool = True

    def render(self, *, max_chars: int = 1200) -> str:
        """渲染为提示词可读文本。不可信内容带显式标记。"""
        body = self.content
        if len(body) > max_chars:
            body = body[:max_chars] + f"…[截断，共 {len(body)} 字符]"
        tag = "不可信数据" if self.untrusted else "运行时事实"
        status = "ok" if self.ok else "失败"
        return f"<observation source={self.source} status={status} trust={tag}>\n{body}\n</observation>"

    def to_json(self) -> JSONObject:
        """序列化为 trace 用的 JSON 对象。"""
        return {
            "source": self.source,
            "content": self.content,
            "ok": self.ok,
            "step_id": self.step_id,
            "data": dict(self.data),
            "untrusted": self.untrusted,
        }


# ---------------------------------------------------------------------------
# 记忆
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryItem:
    """一条记忆。``importance`` 影响注入顺序与截断优先级。"""

    text: str
    kind: MemoryKind
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tags: tuple[str, ...] = ()
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    metadata: JSONObject = field(default_factory=dict)

    def to_json(self) -> JSONObject:
        """序列化为持久化用的 JSON 对象。"""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "text": self.text,
            "tags": list(self.tags),
            "importance": self.importance,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "MemoryItem":
        """从持久化 JSON 重建记忆条目。"""
        return cls(
            text=str(obj.get("text") or ""),
            kind=MemoryKind(str(obj.get("kind") or MemoryKind.SEMANTIC.value)),
            id=str(obj.get("id") or uuid.uuid4().hex[:12]),
            tags=tuple(str(tag) for tag in (obj.get("tags") or [])),
            importance=float(obj.get("importance", 0.5)),
            created_at=float(obj.get("created_at", time.time())),
            metadata=dict(obj.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MemoryQuery:
    """检索请求。``kinds`` 为 ``None`` 表示全部层。"""

    text: str
    k: int = 5
    kinds: tuple[MemoryKind, ...] | None = None
    tags: tuple[str, ...] = ()
    min_score: float = 0.0


@dataclass(frozen=True)
class MemoryHit:
    """检索命中项与打分。分数不做跨实现可比承诺。"""

    item: MemoryItem
    score: float


# ---------------------------------------------------------------------------
# 任务与运行结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """一个待执行任务。

    ``mock_script`` 只在离线机制自检中使用：按角色名映射到脚本化模型输出队列。
    真实能力评测必须使用真实 provider 且不带脚本。
    """

    id: str
    goal: str
    grader_id: str = ""
    initial_files: dict[str, str] = field(default_factory=dict)
    max_steps: int | None = None
    inner_steps: int | None = None
    mock_script: dict[str, list[JSONObject]] | None = None
    metadata: JSONObject = field(default_factory=dict)

    def to_json(self) -> JSONObject:
        """序列化为 trace 用的 JSON 对象（不包含脚本内容）。"""
        return {
            "id": self.id,
            "goal": self.goal,
            "grader_id": self.grader_id,
            "files": sorted(self.initial_files),
            "metadata": dict(self.metadata),
        }


@dataclass
class Metrics:
    """一次 run 的资源与行为计数。全部字段可累加，粒度到步骤级。"""

    steps: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    tool_calls: int = 0
    tool_failures: int = 0
    replans: int = 0
    retries: int = 0
    lessons: int = 0
    safety_blocks: int = 0
    format_repairs: int = 0

    @property
    def total_tokens(self) -> int:
        """输入与输出 token 之和。"""
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> JSONObject:
        """序列化为报告用的 JSON 对象。"""
        return {
            "steps": self.steps,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "seconds": round(self.seconds, 3),
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "replans": self.replans,
            "retries": self.retries,
            "lessons": self.lessons,
            "safety_blocks": self.safety_blocks,
            "format_repairs": self.format_repairs,
        }


@dataclass
class RunResult:
    """一次 run 的最终结果。``success`` 只由外部 grader 判定，不采信 agent 自评。"""

    run_id: str
    task_id: str
    status: RunStatus
    success: bool = False
    answer: str = ""
    metrics: Metrics = field(default_factory=Metrics)
    stop_reason: str = ""
    error: str | None = None
    trace_path: str = ""
    report_path: str = ""
    plan: Plan | None = None

    def summary(self) -> str:
        """渲染为命令行与报告共用的一行摘要。"""
        return (
            f"{self.task_id}: {self.status.value}"
            f"{' ✓' if self.success else ''}"
            f" | steps={self.metrics.steps} calls={self.metrics.llm_calls}"
            f" tokens={self.metrics.total_tokens} replans={self.metrics.replans}"
            f"{' | ' + self.stop_reason if self.stop_reason else ''}"
        )


# ---------------------------------------------------------------------------
# 能力接口（Protocol）
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """模型补全能力。离线 mock 与真实 HTTP provider 实现同一接口。"""

    name: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        """执行一次补全。

        参数：
            messages: 对话消息序列。
            temperature: 采样温度，``None`` 用 provider 默认值。
            max_tokens: 输出上限，``None`` 用 provider 默认值。
            json_mode: 是否要求提供方返回严格 JSON（OpenAI 兼容接口的 response_format）。
            timeout: 本次请求超时秒数，``None`` 用 provider 默认值。

        返回：
            :class:`LLMResponse`。

        异常：
            LLMTransportError: 网络或 HTTP 层失败且重试耗尽。
            LLMResponseFormatError: 提供方返回体结构不可解析。
        """
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """单层记忆存储。四层实现共享该接口，便于统一检索与替换。"""

    kind: MemoryKind

    def add(self, item: MemoryItem) -> str:
        """写入一条记忆，返回其 id。"""
        ...

    def search(self, query: MemoryQuery) -> list[MemoryHit]:
        """按查询检索，返回按分数降序的命中列表。"""
        ...

    def recent(self, *, k: int = 10) -> list[MemoryItem]:
        """返回最近的 k 条记忆（新→旧）。"""
        ...

    def __len__(self) -> int:
        """返回已存条目数。"""
        ...


@runtime_checkable
class Tool(Protocol):
    """环境工具。``run`` 不抛异常，失败用 ``ok=False`` 表达。"""

    spec: ToolSpec

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """执行工具。

        参数：
            arguments: 已解析的参数对象，需由实现自行校验必需字段。

        返回：
            :class:`ToolResult`；参数非法或执行失败时 ``ok`` 为 ``False``。
        """
        ...


ApprovalFn = Callable[[ToolSpec, Mapping[str, Any]], bool]
"""审批回调：返回 ``True`` 允许执行，``False`` 拒绝。非交互模式下默认实现恒返回 ``False``。"""

AnswerFn = Callable[[str], str]
"""人类应答回调：给 ``ASK`` 动作提供答案；返回空串视为无人应答。"""
