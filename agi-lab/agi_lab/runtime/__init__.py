"""运行时：Agent 循环、预算、安全策略、追踪。

对外主入口：:class:`agi_lab.runtime.agent.Agent`。一次 :meth:`Agent.run` 完成
"召回教训 → 规划 → 计划执行（内层 ReAct）→ 评审/重试 → 反思重规划 → 终审 →
落盘 trace 与报告"的完整闭环。
"""

from .agent import Agent
from .budget import BudgetTracker
from .safety import PolicyGuard
from .tracing import Tracer, read_events, render_timeline, write_report

__all__ = ["Agent", "BudgetTracker", "PolicyGuard", "Tracer", "read_events", "render_timeline", "write_report"]
