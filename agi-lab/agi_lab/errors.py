"""异常层级。

约定：可预期的运行时失败（预算耗尽、工具失败、模型输出格式错误）都有独立类型，
调用方据此选择"停止 / 反思 / 重试"中的一种，禁止笼统 ``except Exception``。
所有异常携带足够信息写入 trace。
"""

from __future__ import annotations


class AgiLabError(Exception):
    """本项目的异常基类。"""


class ConfigError(AgiLabError):
    """配置缺失或取值非法。加载期即失败，不延迟到运行中。"""


class LLMError(AgiLabError):
    """模型调用相关错误基类。"""


class LLMTransportError(LLMError):
    """网络、HTTP 状态码或超时错误；重试耗尽后抛出。"""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class LLMResponseFormatError(LLMError):
    """提供方返回体不可解析（非 JSON、缺 choices、内容为空等）。"""

    def __init__(self, message: str, *, excerpt: str = "") -> None:
        super().__init__(message)
        self.excerpt = excerpt


class ActionFormatError(AgiLabError):
    """模型动作无法解析为合法 :class:`agi_lab.types.Action`。"""


class PlanFormatError(AgiLabError):
    """模型计划无法解析为合法 :class:`agi_lab.types.Plan`。"""


class ToolError(AgiLabError):
    """工具实现内部错误基类。工具注册表据此转为 ``ToolResult(ok=False)``。"""


class ToolInvocationError(ToolError):
    """工具调用参数非法（缺字段、类型不符、越界取值）。"""


class SafetyViolation(AgiLabError):
    """安全策略拒绝执行。必须记录为安全事件，不得重试绕过。"""


class BudgetExceeded(AgiLabError):
    """预算耗尽。携带具体维度，便于报告说明"停在哪一步"。"""

    def __init__(self, limit_name: str, limit: float, used: float) -> None:
        super().__init__(f"预算耗尽：{limit_name} 上限 {limit:g}，已用 {used:g}")
        self.limit_name = limit_name
        self.limit = limit
        self.used = used


class TaskError(AgiLabError):
    """任务定义或评测运行错误（任务不存在、grader 未知、fixture 写入失败）。"""
