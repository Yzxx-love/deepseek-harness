"""记忆工具：通过 duck-typed ``memory`` 对象访问记忆层，不导入具体实现。

只要求传入对象具备 ``search(text=..., k=...)`` 与 ``add_fact(text=..., tags=...)``；
方法缺失时返回 ``ok=False`` 而不是抛错，这样记忆实现可以独立演进与替换。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import ToolInvocationError
from ..types import JSONObject, ToolResult, ToolSpec
from .base import DEFAULT_TRUNCATE_CHARS, BaseTool, truncate_text

MAX_HIT_TEXT_CHARS = 160
DEFAULT_SEARCH_K = 5
MAX_SEARCH_K = 50


def _require_method(memory: object, name: str) -> Any:
    method = getattr(memory, name, None)
    if not callable(method):
        raise ToolInvocationError(f"memory 对象缺少可用方法: {name}()")
    return method


class MemorySearchTool(BaseTool):
    """检索跨任务记忆。"""

    def __init__(self, memory: object) -> None:
        self._memory = memory
        self.spec = ToolSpec(
            name="memory_search",
            description="检索跨任务记忆（事实、情节、教训），返回命中的 kind/score/摘要",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词或问题"},
                    "k": {
                        "type": "integer",
                        "description": f"最多返回条数，默认 {DEFAULT_SEARCH_K}",
                    },
                },
                "required": ["query"],
            },
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = self._require_str(arguments, "query")
        k = self._require_optional_int(arguments, "k", DEFAULT_SEARCH_K, minimum=1, maximum=MAX_SEARCH_K)
        search = _require_method(self._memory, "search")
        hits = list(search(text=query, k=k) or [])
        metadata: JSONObject = {"query": query, "k": k, "count": len(hits)}
        if not hits:
            return ToolResult(ok=True, output="无相关记忆", metadata=metadata)
        lines = [_render_hit(index, hit) for index, hit in enumerate(hits, start=1)]
        return ToolResult(ok=True, output=truncate_text("\n".join(lines), DEFAULT_TRUNCATE_CHARS), metadata=metadata)


class MemoryRememberTool(BaseTool):
    """把一条事实写入语义记忆。"""

    def __init__(self, memory: object) -> None:
        self._memory = memory
        self.spec = ToolSpec(
            name="memory_remember",
            description="把一条事实写入长期语义记忆，返回记忆 id",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要记住的事实文本"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选标签，用于检索过滤",
                    },
                },
                "required": ["text"],
            },
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        text = self._require_str(arguments, "text")
        tags = self._require_optional_str_list(arguments, "tags")
        add_fact = _require_method(self._memory, "add_fact")
        memory_id = add_fact(text=text, tags=tags)
        if memory_id is None:
            memory_id = ""
        elif not isinstance(memory_id, str):
            memory_id = str(getattr(memory_id, "id", memory_id))
        output = f"已写入语义记忆，id={memory_id}" if memory_id else "已写入语义记忆"
        metadata: JSONObject = {"id": memory_id, "tags": tags, "text": text[:200]}
        return ToolResult(ok=True, output=output, metadata=metadata)


def _render_hit(index: int, hit: object) -> str:
    score = getattr(hit, "score", None)
    item = getattr(hit, "item", hit)
    kind = getattr(item, "kind", "")
    kind_text = getattr(kind, "value", kind)
    text = str(getattr(item, "text", item))
    summary = text[:MAX_HIT_TEXT_CHARS] + "…" if len(text) > MAX_HIT_TEXT_CHARS else text
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
    return f"{index}. [kind={kind_text or '?'} score={score_text}] {summary}"
