"""记忆门面：把四层记忆装配成运行时唯一依赖的对象。

``MemoryHub`` 是运行时唯一直接使用的记忆接口（见 ``docs/02-architecture.md`` 的记忆设计）。
它不实现新检索算法，只负责：

- 生命周期：``persist=True`` 且 ``root`` 不为 ``None`` 时，三层磁盘记忆写
  ``root/<kind>.jsonl``；否则纯内存、不写盘；
- 写入：为各层补齐 kind / importance / tags / metadata；
- 检索：合并三层命中，按分数降序、同分按 ``created_at`` 新者优先排序并去重；
- 渲染：按教训 → 事实 → 情节的优先级注入提示，受字符预算约束。

各层默认重要度：情节 ``0.4``、事实 ``0.6``、教训 ``0.8``。分数只在本门面内用于排序，
不做跨实现的绝对分数承诺。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from ..types import (
    JSONObject,
    MemoryHit,
    MemoryItem,
    MemoryKind,
    MemoryQuery,
    Observation,
    Plan,
)
from .episodic import EpisodicMemory
from .reflection import ReflectionMemory
from .semantic import PersistentMemoryStore, SemanticMemory
from .working import LineBudget, WorkingMemory

_LAYER_ORDER: tuple[MemoryKind, ...] = (
    MemoryKind.REFLECTION,
    MemoryKind.SEMANTIC,
    MemoryKind.EPISODIC,
)
_LABELS: dict[MemoryKind, str] = {
    MemoryKind.REFLECTION: "教训",
    MemoryKind.SEMANTIC: "事实",
    MemoryKind.EPISODIC: "情节",
}
_DEFAULT_IMPORTANCE: dict[MemoryKind, float] = {
    MemoryKind.EPISODIC: 0.4,
    MemoryKind.SEMANTIC: 0.6,
    MemoryKind.REFLECTION: 0.8,
}
_CONTEXT_PER_KIND = 8
_SNAPSHOT_PER_KIND = 10


class MemoryHub:
    """四层记忆的门面：工作记忆 + 可持久化的情节/语义/反思三层。"""

    working: WorkingMemory
    episodic: EpisodicMemory
    semantic: SemanticMemory
    reflection: ReflectionMemory

    def __init__(self, root: Path | None = None, *, persist: bool = True) -> None:
        """装配记忆层。

        参数：
            root: 磁盘记忆根目录；各层文件为 ``root/episodic.jsonl`` 等。
            persist: 是否写盘。只有 ``persist=True`` 且 ``root`` 不为 ``None`` 才落盘；
                其余组合都是纯内存，不创建目录、不写文件。
        """
        self._persist = bool(persist and root is not None)
        self._root: Path | None = Path(root) if self._persist and root is not None else None
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(self._layer_path(MemoryKind.EPISODIC))
        self.semantic = SemanticMemory(self._layer_path(MemoryKind.SEMANTIC))
        self.reflection = ReflectionMemory(self._layer_path(MemoryKind.REFLECTION))

    @property
    def root(self) -> Path | None:
        """磁盘记忆根目录；纯内存模式为 ``None``。"""
        return self._root

    @property
    def persistent(self) -> bool:
        """是否写盘。"""
        return self._persist

    # ------------------------------------------------------------------
    # 工作记忆
    # ------------------------------------------------------------------

    def set_goal(self, goal: str) -> None:
        """设置当前目标，写入工作记忆。"""
        self.working.set_goal(goal)

    def set_plan(self, plan: Plan) -> None:
        """保存当前计划引用，写入工作记忆。"""
        self.working.set_plan(plan)

    def record_observation(self, step_id: str, observation: Observation) -> None:
        """把一条观测记入工作记忆。

        参数：
            step_id: 当前步骤 id；非空时覆盖观测自带的 ``step_id``。
            observation: 待记录观测。
        """
        if step_id and step_id != observation.step_id:
            observation = replace(observation, step_id=step_id)
        self.working.record(observation)

    # ------------------------------------------------------------------
    # 可持久化记忆写入
    # ------------------------------------------------------------------

    def add_episode(self, text: str, *, step_id: str = "", tags: Sequence[str] = ()) -> str:
        """写入一条情节记忆（执行摘要）。

        参数：
            text: 摘要文本。
            step_id: 对应步骤 id，非空时写入 metadata。
            tags: 标签元组。

        返回：
            条目 id。

        异常：
            ValueError: 文本为空。
        """
        metadata: JSONObject = {}
        if step_id:
            metadata["step_id"] = step_id
        return self.episodic.add(self._make_item(MemoryKind.EPISODIC, text, tags=tags, metadata=metadata))

    def add_fact(self, text: str, *, tags: Sequence[str] = (), source: str = "") -> str:
        """写入一条语义事实。

        参数：
            text: 事实文本。
            tags: 标签元组。
            source: 事实来源，非空时写入 metadata。

        返回：
            条目 id。

        异常：
            ValueError: 文本为空。
        """
        metadata: JSONObject = {}
        if source:
            metadata["source"] = source
        return self.semantic.add(self._make_item(MemoryKind.SEMANTIC, text, tags=tags, metadata=metadata))

    def add_lesson(self, text: str, *, tags: Sequence[str] = (), task_id: str = "") -> str:
        """写入一条反思教训；近似重复条目不新增，改为提升原条目重要度。

        参数：
            text: 教训文本。
            tags: 标签元组。
            task_id: 来源任务 id，非空时写入 metadata。

        返回：
            条目 id；命中重复时是已存在条目的 id。

        异常：
            ValueError: 文本为空。
        """
        metadata: JSONObject = {}
        if task_id:
            metadata["task_id"] = task_id
        return self.reflection.add(
            self._make_item(MemoryKind.REFLECTION, text, tags=tags, metadata=metadata)
        )

    # ------------------------------------------------------------------
    # 检索与渲染
    # ------------------------------------------------------------------

    def search(
        self,
        text: str,
        *,
        k: int = 5,
        kinds: Sequence[MemoryKind] | None = None,
    ) -> list[MemoryHit]:
        """跨层检索并合并结果。

        参数：
            text: 查询文本。
            k: 返回条数上限；``k < 1`` 返回空列表。
            kinds: 限定记忆层；``None`` 表示全部三层，工作记忆永远不参与检索。

        返回：
            按 ``(-score, -created_at, id)`` 排序、至多 ``k`` 条命中。同 id 或
            同 ``kind + text`` 的重复命中只保留分数最高者。
        """
        if k <= 0:
            return []
        selected = tuple(kinds) if kinds is not None else None
        collected: list[MemoryHit] = []
        for kind in _LAYER_ORDER:
            if selected is not None and kind not in selected:
                continue
            collected.extend(
                self._store(kind).search(MemoryQuery(text=text, k=k))
            )
        collected.sort(key=lambda hit: (-hit.score, -hit.item.created_at, hit.item.id))
        merged: list[MemoryHit] = []
        seen_ids: set[str] = set()
        seen_text: set[tuple[MemoryKind, str]] = set()
        for hit in collected:
            content_key = (hit.item.kind, hit.item.text)
            if hit.item.id in seen_ids or content_key in seen_text:
                continue
            seen_ids.add(hit.item.id)
            seen_text.add(content_key)
            merged.append(hit)
            if len(merged) >= k:
                break
        return merged

    def render_context(self, *, query: str = "", max_chars: int = 4000) -> str:
        """渲染跨层记忆上下文。

        优先级固定为教训 → 事实 → 情节。``query`` 非空时每层按各自检索分排序后注入；
        为空时教训按 importance 降序、事实与情节按 ``created_at`` 倒序。总字符数不超过
        ``max_chars``，放不下的内容按剩余预算截断并补 ``…``。

        参数：
            query: 检索词；空串表示不按相关度排序。
            max_chars: 返回文本的字符上限（含换行与标签）。

        返回：
            以 ``[教训]`` / ``[事实]`` / ``[情节]`` 为小节的文本；无记忆时为空串。
        """
        budget = LineBudget(max_chars)
        for kind in _LAYER_ORDER:
            texts = self._context_texts(kind, query)
            if not texts:
                continue
            if not budget.push(f"[{_LABELS[kind]}]"):
                break
            for text in texts:
                if not budget.push(f"- {text}"):
                    break
        return budget.render()

    # ------------------------------------------------------------------
    # 状态导出
    # ------------------------------------------------------------------

    def snapshot(self) -> JSONObject:
        """返回全部记忆层的 JSON 可序列化快照；磁盘层只含最近 10 条。"""
        return {
            "persist": self._persist,
            "root": str(self._root) if self._root is not None else None,
            "working": self.working.snapshot(),
            "episodic": self._layer_snapshot(self.episodic),
            "semantic": self._layer_snapshot(self.semantic),
            "reflection": self._layer_snapshot(self.reflection),
        }

    def stats(self) -> JSONObject:
        """返回各层条目数与持久化状态的 JSON 可序列化统计。"""
        return {
            "persist": self._persist,
            "root": str(self._root) if self._root is not None else None,
            "working_observations": len(self.working.observations()),
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "reflection": len(self.reflection),
            "total": len(self.episodic) + len(self.semantic) + len(self.reflection),
            "bad_lines": (
                self.episodic.bad_lines + self.semantic.bad_lines + self.reflection.bad_lines
            ),
        }

    def flush(self) -> None:
        """把三层磁盘记忆的内存状态重写回各自 JSONL；纯内存模式为空操作。"""
        for store in (self.episodic, self.semantic, self.reflection):
            store.flush()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _layer_path(self, kind: MemoryKind) -> Path | None:
        """返回某层的 JSONL 路径；纯内存模式返回 ``None``。"""
        if self._root is None:
            return None
        return self._root / f"{kind.value}.jsonl"

    def _make_item(
        self,
        kind: MemoryKind,
        text: str,
        *,
        tags: Sequence[str],
        metadata: JSONObject,
    ) -> MemoryItem:
        """按层默认重要度构造记忆条目。"""
        return MemoryItem(
            text=text,
            kind=kind,
            tags=tuple(tags),
            importance=_DEFAULT_IMPORTANCE[kind],
            metadata=metadata,
        )

    def _store(self, kind: MemoryKind) -> PersistentMemoryStore:
        """按记忆层取对应存储对象。"""
        if kind is MemoryKind.EPISODIC:
            return self.episodic
        if kind is MemoryKind.SEMANTIC:
            return self.semantic
        if kind is MemoryKind.REFLECTION:
            return self.reflection
        raise KeyError(f"MemoryHub 没有可检索的 {kind.value} 存储")

    def _context_texts(self, kind: MemoryKind, query: str) -> list[str]:
        """按层收集待注入的文本，query 非空时按相关度排序。"""
        store = self._store(kind)
        if query.strip():
            hits = store.search(MemoryQuery(text=query, k=_CONTEXT_PER_KIND))
            texts = [hit.item.text for hit in hits]
        elif kind is MemoryKind.REFLECTION:
            items = sorted(
                store.recent(k=_CONTEXT_PER_KIND),
                key=lambda item: (-item.importance, -item.created_at, item.id),
            )
            texts = [item.text for item in items]
        else:
            texts = [item.text for item in store.recent(k=_CONTEXT_PER_KIND)]
        return list(dict.fromkeys(texts))

    @staticmethod
    def _layer_snapshot(store: PersistentMemoryStore) -> JSONObject:
        """导出一个磁盘层的条目数与最近条目。"""
        return {
            "count": len(store),
            "bad_lines": store.bad_lines,
            "recent": [item.to_json() for item in store.recent(k=_SNAPSHOT_PER_KIND)],
        }
