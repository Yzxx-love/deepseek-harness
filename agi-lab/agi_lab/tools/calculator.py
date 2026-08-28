"""计算器工具：用 AST 白名单求值算术表达式，绝不执行 Python 代码。

允许的节点只有数字常量、二元算术运算与一元正负号；``Name``、``Call``、``Attribute``、
``Subscript``、推导式等一律拒绝。表达式长度、``**`` 指数与最终结果都有硬上限，防止
超大整数或浮点溢出拖垮进程。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Mapping

from ..errors import ToolInvocationError
from ..types import JSONObject, ToolResult, ToolSpec
from .base import BaseTool

MAX_EXPRESSION_CHARS = 500
MAX_EXPONENT_ABS = 100
MAX_RESULT_ABS = 1e15
# 中间整数结果的位数上限：阻止 ((2**100)**100)**100 这类表达式在求值期爆炸。
MAX_INTERMEDIATE_BITS = 20_000

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


class CalculatorTool(BaseTool):
    """纯算术计算器，输入一条表达式，返回数字文本。"""

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="calculator",
            description="计算算术表达式（只支持数字与 + - * / // % **，不执行任意代码）",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的算术表达式，如 (2+3)*4"},
                },
                "required": ["expression"],
            },
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        expression = self._require_str(arguments, "expression")
        if len(expression) > MAX_EXPRESSION_CHARS:
            raise ToolInvocationError(f"表达式过长: {len(expression)} 字符，上限 {MAX_EXPRESSION_CHARS}")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolInvocationError(f"表达式语法错误: {exc.msg or exc}") from None
        except (ValueError, MemoryError, RecursionError) as exc:
            raise ToolInvocationError(f"表达式无法解析: {type(exc).__name__}") from exc
        value = self._eval(tree)
        if abs(value) > MAX_RESULT_ABS:
            raise ToolInvocationError(f"结果绝对值超过 {MAX_RESULT_ABS:g}，拒绝计算")
        output = _format_number(value)
        metadata: JSONObject = {"expression": expression, "value": value}
        return ToolResult(ok=True, output=output, metadata=metadata)

    def _eval(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return self._eval(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ToolInvocationError(f"表达式只允许数字常量，收到 {type(value).__name__}")
            return value
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ToolInvocationError(f"不支持的一元运算符: {type(node.op).__name__}")
        raise ToolInvocationError(f"不支持的表达式节点: {type(node).__name__}")

    def _eval_binop(self, node: ast.BinOp) -> int | float:
        if isinstance(node.op, ast.Pow):
            return self._eval_power(node)
        function = _BINARY_OPERATORS.get(type(node.op))
        if function is None:
            raise ToolInvocationError(f"不支持的运算符: {type(node.op).__name__}")
        left = self._eval(node.left)
        right = self._eval(node.right)
        try:
            value = function(left, right)
        except ZeroDivisionError:
            raise ToolInvocationError("除零错误：表达式中存在除以 0 的运算") from None
        except OverflowError as exc:
            raise ToolInvocationError(f"数值溢出: {exc}") from exc
        return _check_intermediate(value)

    def _eval_power(self, node: ast.BinOp) -> int | float:
        exponent = self._eval(node.right)
        if abs(exponent) > MAX_EXPONENT_ABS:
            raise ToolInvocationError(f"** 的指数绝对值不能超过 {MAX_EXPONENT_ABS}")
        base = self._eval(node.left)
        try:
            value = base**exponent
        except ZeroDivisionError:
            raise ToolInvocationError("除零错误：0 不能取负数次幂") from None
        except OverflowError as exc:
            raise ToolInvocationError(f"数值溢出: {exc}") from exc
        return _check_intermediate(value)


def _check_intermediate(value: int | float) -> int | float:
    if isinstance(value, complex):
        raise ToolInvocationError("数学结果不是实数，拒绝计算")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolInvocationError("中间结果不是有限数值，拒绝计算")
        return value
    if isinstance(value, int):
        if value.bit_length() > MAX_INTERMEDIATE_BITS:
            raise ToolInvocationError("中间整数结果过大，拒绝计算")
        return value
    raise ToolInvocationError(f"不支持的数值类型: {type(value).__name__}")


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) <= MAX_RESULT_ABS:
        return str(int(value))
    return repr(value)
