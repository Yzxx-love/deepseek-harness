"""工作记忆：当前目标、计划与最近观测，纯内存、按字符预算截断。

工作记忆不实现 ``MemoryStore``：它不参与跨任务检索、不落盘，只负责给当前 run 提供
短上下文。容量有两道上限：条数（``max_items``）与字符数（``max_chars``，只统计
``Observation.content``）。超限时丢弃最旧的观测；唯一的例外是最新一条观测本身
超过字符上限时保留它，否则工作记忆会丢光全部上下文。

:class:`LineBudget` 是渲染期的字符预算工具，``MemoryHub.render_context`` 复用同一套
截断语义：逐行追加，整行放不下时截断该行并结束。
"""

from __future__ import annotations

from typing import Iterable

from ..types import JSONObject, Observation, Plan

_MAX_PLAN_STEPS = 6
_OBSERVATION_PREVIEW_CHARS = 400


class LineBudget:
    """按字符预算逐行拼接文本。

    行之间插入一个换行符；整行放不下时按剩余预算截断该行并补 ``…``，之后
    :attr:`exhausted` 为 ``True``，后续 :meth:`push` 一律返回 ``False``。
    :meth:`render` 结果的 ``len`` 永不超过构造时的预算。
    """

    def __init__(self, max_chars: int) -> None:
        """构造预算。

        参数：
            max_chars: 允许的最大字符数（含换行符）。

        异常：
            ValueError: ``max_chars < 0``。
        """
        if max_chars < 0:
            raise ValueError(f"max_chars 不能为负数，收到 {max_chars}")
        self._max_chars = max_chars
        self._lines: list[str] = []
        self._length = 0
        self.exhausted = False

    def push(self, line: str) -> bool:
        """追加一行。

        参数：
            line: 单行文本（内部换行会被原样保留，长度照算）。

        返回：
            ``True`` 表示整行写入；``False`` 表示预算已耗尽（该行可能被截断）。
        """
        if self.exhausted:
            return False
        separator = 1 if self._lines else 0
        room = self._max_chars - self._length - separator
        if room <= 1:
            self.exhausted = True
            return False
        if len(line) > room:
            self._lines.append(line[: room - 1] + "…")
            self._length = self._max_chars
            self.exhausted = True
            return False
        self._lines.append(line)
        self._length += separator + len(line)
        return True

    def extend(self, lines: Iterable[str]) -> None:
        """依次追加多行，预算耗尽即停止。"""
        for line in lines:
            if not self.push(line):
                break

    def render(self) -> str:
        """返回拼接后的文本。"""
        return "\n".join(self._lines)

    def __len__(self) -> int:
        """返回已写入的行数。"""
        return len(self._lines)


class WorkingMemory:
    """当前 run 的工作记忆：目标、计划与最近观测。"""

    def __init__(self, max_items: int = 20, max_chars: int = 8000) -> None:
        """构造工作记忆。

        参数：
            max_items: 保留观测的条数上限（``>= 1``）。
            max_chars: 保留观测内容的总字符上限（``>= 1``）。

        异常：
            ValueError: 任一上限小于 1。
        """
        if max_items < 1:
            raise ValueError(f"max_items 必须 >= 1，收到 {max_items}")
        if max_chars < 1:
            raise ValueError(f"max_chars 必须 >= 1，收到 {max_chars}")
        self.max_items = max_items
        self.max_chars = max_chars
        self._goal = ""
        self._plan: Plan | None = None
        self._observations: list[Observation] = []
        self._chars = 0

    @property
    def goal(self) -> str:
        """当前目标；未设置时为空串。"""
        return self._goal

    @property
    def plan(self) -> Plan | None:
        """当前计划引用；运行时可直接改步骤状态，渲染时读取最新值。"""
        return self._plan

    def set_goal(self, goal: str) -> None:
        """设置当前目标（去除首尾空白）。"""
        self._goal = goal.strip()

    def set_plan(self, plan: Plan) -> None:
        """保存当前计划引用（不复制，渲染时读取实时状态）。"""
        self._plan = plan

    def record(self, observation: Observation) -> None:
        """记录一条观测，并按条数与字符上限淘汰最旧的条目。

        ``max_chars`` 只统计 ``Observation.content`` 的字符数；若唯一剩下的最新观测
        本身超过上限则保留它，否则工作记忆会在长输出后丢失全部上下文。
        """
        self._observations.append(observation)
        self._chars += len(observation.content)
        while (
            len(self._observations) > 1
            and (len(self._observations) > self.max_items or self._chars > self.max_chars)
        ):
            dropped = self._observations.pop(0)
            self._chars -= len(dropped.content)

    def clear(self) -> None:
        """清空目标、计划与全部观测。"""
        self._goal = ""
        self._plan = None
        self._observations.clear()
        self._chars = 0

    def observations(self, limit: int | None = None) -> list[Observation]:
        """返回观测副本（旧→新）。

        参数：
            limit: 只保留最近 ``limit`` 条；``None`` 表示全部；``< 1`` 返回空列表。
        """
        if limit is None:
            return list(self._observations)
        if limit < 1:
            return []
        return list(self._observations[-limit:])

    def render(self, max_chars: int = 4000) -> str:
        """渲染目标、计划摘要与最近观测，总长度不超过 ``max_chars``。

        参数：
            max_chars: 输出字符上限；放不下的行被截断并补 ``…``。

        返回：
            按 ``目标`` → ``计划`` → ``最近观测`` 顺序排版的文本；空工作记忆返回空串。
            观测按新→旧排列，预算不足时优先保留最新的观测。
        """
        budget = LineBudget(max_chars)
        if self._goal:
            budget.push(f"目标：{self._goal}")
        if self._plan is not None:
            budget.extend(self._plan_summary())
        if self._observations:
            if budget.push("最近观测："):
                for observation in reversed(self._observations):
                    if not budget.push(self._observation_line(observation)):
                        break
        return budget.render()

    def snapshot(self) -> JSONObject:
        """返回 JSON 可序列化的状态快照，供 trace 与调试使用。"""
        return {
            "goal": self._goal,
            "plan": self._plan.to_json() if self._plan is not None else None,
            "observations": [observation.to_json() for observation in self._observations],
            "count": len(self._observations),
            "chars": self._chars,
            "max_items": self.max_items,
            "max_chars": self.max_chars,
        }

    def _plan_summary(self) -> list[str]:
        """渲染计划摘要：版本进度 + 至多 6 步。"""
        plan = self._plan
        if plan is None:
            return []
        done, total = plan.progress()
        lines = [f"计划 v{plan.version}（{done}/{total} 完成）"]
        shown = plan.steps[:_MAX_PLAN_STEPS]
        for index, step in enumerate(shown, start=1):
            lines.append(f"{index}. [{step.status.value}] {step.id}: {step.description}")
        hidden = len(plan.steps) - len(shown)
        if hidden > 0:
            lines.append(f"…其余 {hidden} 步省略")
        return lines

    @staticmethod
    def _observation_line(observation: Observation) -> str:
        """把单条观测压成一行，保留来源、状态与信任标记。"""
        trust = "不可信数据" if observation.untrusted else "运行时事实"
        status = "ok" if observation.ok else "失败"
        step = f" step={observation.step_id}" if observation.step_id else ""
        content = " ".join(observation.content.split())
        if len(content) > _OBSERVATION_PREVIEW_CHARS:
            content = content[:_OBSERVATION_PREVIEW_CHARS] + "…"
        return f"- [{observation.source}{step} {status} {trust}] {content}"
