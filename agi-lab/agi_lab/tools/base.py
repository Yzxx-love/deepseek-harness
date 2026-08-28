"""工具注册表、工具基类与工具实现共用的小工具。

注册表是运行时的安全强制点：

- 必需字段与基础 JSON 类型在调用工具前校验；
- ``requires_approval`` 工具按 ``approval_mode`` 放行或拒绝，未提供审批回调时 fail closed；
- 工具抛出的任何异常都转成 ``ToolResult(ok=False)``，:meth:`ToolRegistry.execute`
  对调用方永不抛异常。

检查顺序固定为"参数校验 → 审批 → 工具执行"：审批先于工具内部的工作区围栏等校验，
未获批准的工具完全不会被调用，因此审批拒绝时不会透露路径之外的信息，也不会先做
无关校验。这是 fail closed 的优先级约定。
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

from ..errors import SafetyViolation, ToolError, ToolInvocationError
from ..types import ApprovalFn, JSONObject, Tool, ToolCall, ToolResult, ToolSpec

# 工具输出在进入提示词前统一截断，避免一次观测挤掉上下文。
DEFAULT_TRUNCATE_CHARS = 4000

_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET")


def truncate_text(text: str, max_chars: int) -> str:
    """截断文本并附加总字符数提示。

    参数：
        text: 原始文本。
        max_chars: 保留的最大字符数。

    返回：
        未超长时原样返回；超长时返回前 ``max_chars`` 字符加一行截断提示。
    """
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n…[截断，共 {len(text)} 字符]"


def combine_output(stdout: str | None, stderr: str | None) -> str:
    """把子进程的 stdout 与 stderr 合并为一段文本。

    参数：
        stdout: 标准输出；``None`` 与空串等价。
        stderr: 标准错误；``None`` 与空串等价。

    返回：
        各段去掉行尾换行后按换行连接；均无内容时返回空串。
    """
    parts = [part.rstrip("\r\n") for part in (stdout or "", stderr or "") if part]
    return "\n".join(parts)


def sanitized_environment() -> dict[str, str]:
    """返回剔除敏感变量后的进程环境副本。

    变量名含 ``KEY`` / ``TOKEN`` / ``SECRET``（不区分大小写）的一律不传给子进程，
    防止模型通过子进程读取宿主密钥。

    返回：
        新的环境变量字典，修改它不影响当前进程环境。
    """
    return {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }


class ToolRegistry:
    """按名字管理工具，统一执行、参数校验与审批。

    参数：
        approval: 审批回调，签名见 :data:`agi_lab.types.ApprovalFn`；``None`` 时
            ``prompt`` 语义按拒绝处理。
        approval_mode: ``"allow"`` 直接放行需审批工具；``"deny"`` 一律拒绝；
            其余取值（含 ``"prompt"``）在提供回调时询问，未提供回调时拒绝。
    """

    def __init__(self, *, approval: ApprovalFn | None = None, approval_mode: str = "deny") -> None:
        self._approval = approval
        self._approval_mode = approval_mode
        self._tools: dict[str, Tool] = {}
        # 最近一次调用的工具名与耗时，供运行时 trace 与测试观察。
        self.last_tool_name: str | None = None
        self.last_duration: float = 0.0
        self.call_count: int = 0

    def register(self, tool: Tool) -> None:
        """注册工具。

        参数：
            tool: 满足 :class:`agi_lab.types.Tool` 协议的工具实例。

        异常：
            ValueError: 工具名为空或与已注册工具重名。
        """
        name = getattr(tool.spec, "name", "")
        if not isinstance(name, str) or not name:
            raise ValueError("工具 spec.name 不能为空")
        if name in self._tools:
            raise ValueError(f"工具重名: {name}")
        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """移除工具。

        异常：
            KeyError: 工具未注册。
        """
        del self._tools[name]

    def get(self, name: str) -> Tool:
        """按名字取工具。

        异常：
            KeyError: 工具未注册。
        """
        return self._tools[name]

    def names(self) -> list[str]:
        """返回全部工具名，按字母序排序（确定性渲染）。"""
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        """返回全部工具 spec，顺序与 :meth:`names` 一致。"""
        return [self._tools[name].spec for name in self.names()]

    def describe(self, *, max_chars: int = 4000) -> str:
        """渲染工具列表，供提示词使用；每行一条 :meth:`ToolSpec.render`。

        参数：
            max_chars: 文本字符上限。

        返回：
            多行工具说明；超长时截断并附加提示；空注册表返回"（无可用工具）"。
        """
        specs = self.specs()
        if not specs:
            return "（无可用工具）"
        text = "\n".join(spec.render() for spec in specs)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…[工具列表截断]"
        return text

    def execute(self, call: ToolCall) -> ToolResult:
        """执行一次工具调用，永不向调用方抛异常。

        参数：
            call: 模型给出的工具名与参数对象。

        返回：
            :class:`agi_lab.types.ToolResult`。检查顺序为"未知工具 → 参数校验 → 审批 →
            工具执行"，``ok=False`` 时：

            - 未知工具：``error`` 为"未知工具: x"，``metadata["unknown_tool"]`` 为真；
            - 参数缺失或类型不符：``metadata["invalid_arguments"]`` 为真；
            - 审批拒绝：``metadata["safety"]`` 为真且 ``metadata["denied"] == "approval_required"``；
            - 工具抛 :class:`agi_lab.errors.SafetyViolation`：``metadata["safety"]`` 为真；
            - 其他异常：``error`` 含异常类名与消息，``metadata["tool_error"]`` 为真。
        """
        self.call_count += 1
        started = time.perf_counter()
        try:
            self.last_tool_name = call.tool
            result = self._execute(call)
        except Exception as exc:  # execute 对调用方永不抛异常，任何实现缺陷都转为工具失败
            result = _exception_result(exc)
        self.last_duration = time.perf_counter() - started
        return result

    def _execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.tool)
        if tool is None:
            return ToolResult(ok=False, error=f"未知工具: {call.tool}", metadata={"unknown_tool": True})
        raw_arguments = getattr(call, "arguments", None)
        if not isinstance(raw_arguments, Mapping):
            return ToolResult(
                ok=False,
                error=f"工具 {call.tool} 的 arguments 必须是 JSON 对象",
                metadata={"invalid_arguments": True},
            )
        arguments: JSONObject = dict(raw_arguments)
        invalid = _validate_arguments(tool.spec, arguments)
        if invalid is not None:
            return invalid
        if tool.spec.requires_approval:
            denial = self._approval_denial(tool.spec, arguments)
            if denial is not None:
                return denial
        try:
            result = tool.run(arguments)
        except SafetyViolation as exc:
            return ToolResult(ok=False, error=str(exc) or "安全策略拒绝执行", metadata={"safety": True})
        except ToolError as exc:
            return _exception_result(exc)
        except Exception as exc:  # 工具实现未按约定抛 ToolError 时也归为工具失败
            return _exception_result(exc)
        if not isinstance(result, ToolResult):
            return ToolResult(
                ok=False,
                error=f"工具 {call.tool} 的返回值不是 ToolResult",
                metadata={"tool_error": True},
            )
        return result

    def _approval_denial(self, spec: ToolSpec, arguments: JSONObject) -> ToolResult | None:
        if self._approval_mode == "allow":
            return None
        if self._approval_mode != "deny" and self._approval is not None:
            try:
                approved = bool(self._approval(spec, dict(arguments)))
            except Exception:  # 审批回调自身出错时 fail closed
                approved = False
            if approved:
                return None
        return ToolResult(
            ok=False,
            error=f"审批未通过，拒绝执行: {spec.name}",
            metadata={"safety": True, "denied": "approval_required"},
        )


def _exception_result(exc: BaseException) -> ToolResult:
    label = type(exc).__name__
    message = str(exc).strip()
    return ToolResult(ok=False, error=f"{label}: {message}" if message else label, metadata={"tool_error": True})


def _validate_arguments(spec: ToolSpec, arguments: JSONObject) -> ToolResult | None:
    parameters = spec.parameters
    raw_required = parameters.get("required")
    required = [str(name) for name in raw_required] if isinstance(raw_required, list) else []
    missing = [name for name in required if name not in arguments or arguments[name] is None]
    if missing:
        return ToolResult(
            ok=False,
            error=f"缺少参数: {', '.join(missing)}",
            metadata={"invalid_arguments": True},
        )
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        return None
    for name, schema in properties.items():
        if name not in arguments or arguments[name] is None:
            continue
        if not isinstance(schema, Mapping):
            continue
        expected = schema.get("type")
        if not isinstance(expected, str) or _matches_json_type(arguments[name], expected):
            continue
        return ToolResult(
            ok=False,
            error=f"参数 {name} 必须是 {expected} 类型，收到 {type(arguments[name]).__name__}",
            metadata={"invalid_arguments": True},
        )
    return None


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, Mapping)
    return True


class BaseTool:
    """工具实现辅助基类：``spec`` 在子类构造器里赋值，``run`` 只做分发。

    子类实现 :meth:`_execute`。参数非法时抛 :class:`ToolInvocationError`，安全拒绝时抛
    :class:`SafetyViolation`，二者由 :class:`ToolRegistry` 归类为 ``ok=False``。
    """

    spec: ToolSpec

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """执行工具；异常交给注册表归类，不在工具内部吞掉。"""
        return self._execute(arguments)

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raise NotImplementedError(f"{type(self).__name__} 未实现 _execute")

    @staticmethod
    def _require_str(arguments: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
        """取必需字符串参数，缺失、类型不符或空串时抛 :class:`ToolInvocationError`。"""
        value = arguments.get(key)
        if value is None:
            raise ToolInvocationError(f"缺少参数: {key}")
        if not isinstance(value, str):
            raise ToolInvocationError(f"参数 {key} 必须是字符串")
        if not allow_empty and not value.strip():
            raise ToolInvocationError(f"参数 {key} 不能为空")
        return value

    @staticmethod
    def _require_optional_int(
        arguments: Mapping[str, Any],
        key: str,
        default: int,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        """取可选的整数参数，缺失时返回 ``default``，越界抛 :class:`ToolInvocationError`。"""
        value = arguments.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolInvocationError(f"参数 {key} 必须是整数")
        if minimum is not None and value < minimum:
            raise ToolInvocationError(f"参数 {key} 不能小于 {minimum}")
        if maximum is not None and value > maximum:
            raise ToolInvocationError(f"参数 {key} 不能大于 {maximum}")
        return value

    @staticmethod
    def _require_optional_float(
        arguments: Mapping[str, Any],
        key: str,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """取可选的数字参数，缺失时返回 ``default``，越界抛 :class:`ToolInvocationError`。"""
        value = arguments.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolInvocationError(f"参数 {key} 必须是数字")
        number = float(value)
        if minimum is not None and number < minimum:
            raise ToolInvocationError(f"参数 {key} 不能小于 {minimum:g}")
        if maximum is not None and number > maximum:
            raise ToolInvocationError(f"参数 {key} 不能大于 {maximum:g}")
        return number

    @staticmethod
    def _require_optional_str_list(arguments: Mapping[str, Any], key: str) -> list[str]:
        """取可选的字符串数组参数，缺失时返回空列表，元素非字符串时抛 :class:`ToolInvocationError`。"""
        value = arguments.get(key)
        if value is None:
            return []
        if not isinstance(value, list):
            raise ToolInvocationError(f"参数 {key} 必须是字符串数组")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ToolInvocationError(f"参数 {key} 只能包含字符串")
            items.append(item)
        return items
