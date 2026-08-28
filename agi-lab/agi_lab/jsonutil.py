"""模型输出 JSON 的提取、校验与可序列化转换。

模型输出不可信，这一层负责三类处理：

1. 从自由文本或 Markdown 代码块中提取第一个合法 JSON 对象；
2. 校验必需键，失败抛 :class:`agi_lab.errors.LLMResponseFormatError`（可触发一次修复重试）；
3. 把 dataclass / Enum / Path 递归转成 JSON 可序列化值，供 trace 落盘。
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import LLMResponseFormatError
from .types import JSONObject, JSONValue


def extract_json_object(text: str) -> JSONObject:
    """从模型输出文本中提取一个 JSON 对象。

    处理顺序：整体解析 → Markdown 代码块内解析 → 扫描首个括号平衡的 ``{...}``。
    全部失败时抛 :class:`LLMResponseFormatError`，异常中带 200 字符原文摘要。

    参数：
        text: 模型输出的原始文本。

    返回：
        解析出的 JSON 对象（顶层必须是对象）。

    异常：
        LLMResponseFormatError: 找不到可解析的 JSON 对象。
    """
    raw = (text or "").strip()
    if not raw:
        raise LLMResponseFormatError("模型输出为空", excerpt="")
    candidates: list[str] = [raw]
    candidates.extend(_fenced_blocks(raw))
    balanced = _first_balanced_object(raw)
    if balanced:
        candidates.append(balanced)
    for candidate in candidates:
        parsed = _try_loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    raise LLMResponseFormatError("模型输出中找不到 JSON 对象", excerpt=raw[:200])


def _fenced_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    remainder = text
    while True:
        start = remainder.find("```")
        if start < 0:
            break
        nl = remainder.find("\n", start)
        if nl < 0:
            break
        end = remainder.find("```", nl)
        if end < 0:
            break
        block = remainder[nl + 1 : end].strip()
        if block.startswith("{") or "json" in remainder[start:nl].lower():
            blocks.append(block)
        remainder = remainder[end + 3 :]
    return blocks


def _first_balanced_object(text: str) -> str | None:
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None


def _try_loads(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None


def expect_keys(obj: Mapping[str, Any], required: Sequence[str], *, where: str) -> None:
    """校验 JSON 对象包含必需键。

    参数：
        obj: 待校验对象。
        required: 必需键名序列。
        where: 出错时用于定位的上下文字符串（如 ``planner``）。

    异常：
        LLMResponseFormatError: 缺少任一必需键。
    """
    missing = [key for key in required if key not in obj]
    if missing:
        raise LLMResponseFormatError(
            f"{where} 输出缺少必需字段 {missing}",
            excerpt=json.dumps(dict(obj), ensure_ascii=False)[:200],
        )


def to_jsonable(value: Any) -> JSONValue:
    """把任意运行期值递归转换为 JSON 可序列化值。

    支持 dataclass、Enum、Path、tuple/set、Mapping、Sequence。未知类型退化为
    ``repr`` 字符串，保证 trace 落盘永不因序列化失败而中断。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "to_json") and callable(value.to_json):
        try:
            return to_jsonable(value.to_json())
        except Exception:  # trace 序列化不得因单个对象失败而中断整份日志
            return repr(value)
    return repr(value)
