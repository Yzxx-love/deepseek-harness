"""反思记忆：失败诊断与教训，写入时近似去重，检索按词法相关 + 重要度。

去重判据（任一满足即视为同一条教训，返回已存在条目的 id，不新增）：

1. 归一化文本相同：``casefold`` 后只保留字母数字字符（去掉空白与标点）；
2. 词集合 Jaccard 相似度 ``|A ∩ B| / |A ∪ B| >= 0.85``，词由
   :func:`agi_lab.memory.semantic.tokenize` 产出（任一集合为空时相似度为 0）。

命中重复时把原条目的 ``importance`` 提升为
``min(1.0, max(原值 + 0.1, 新条目 importance))``，并立即重写 JSONL。

检索公式（``Q`` 为查询词集合，``D`` 为条目词集合，``coverage = |Q ∩ D| / |Q|``）：

    score = coverage * (0.75 + 0.25 * min(1.0, importance))

重要度只放大词法相关的条目，不会让完全无关的教训进入结果。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..types import MemoryItem, MemoryKind
from .semantic import PersistentMemoryStore, tokenize

_DEDUP_JACCARD = 0.85
_IMPORTANCE_PROMOTION = 0.1
_LEXICAL_WEIGHT = 0.75
_IMPORTANCE_WEIGHT = 0.25


def _normalize_text(text: str) -> str:
    """归一化教训文本：casefold 后仅保留字母数字字符。"""
    return "".join(char for char in text.casefold() if char.isalnum())


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """两个词集合的 Jaccard 相似度；任一为空时返回 0.0。"""
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


class ReflectionMemory(PersistentMemoryStore):
    """反思记忆：失败诊断与教训条目，JSONL 持久化到 ``reflection.jsonl``。"""

    kind = MemoryKind.REFLECTION

    def add(self, item: MemoryItem) -> str:
        """写入一条教训；近似重复时改为提升原条目重要度。

        参数：
            item: 待写入条目；``kind`` 会被归一化为 ``MemoryKind.REFLECTION``。

        返回：
            条目 id；命中重复时是已存在条目的 id。

        异常：
            ValueError: 文本为空，或 id 已存在（仅非重复条目会触发）。
        """
        duplicate = self._find_duplicate(item.text)
        if duplicate is not None:
            promoted = min(
                1.0,
                max(duplicate.importance + _IMPORTANCE_PROMOTION, item.importance),
            )
            if promoted > duplicate.importance:
                self._replace_item(replace(duplicate, importance=promoted))
            return duplicate.id
        return super().add(item)

    def _find_duplicate(self, text: str) -> MemoryItem | None:
        """按归一化文本或词集合 Jaccard 查找已存在的近似重复条目。"""
        normalized = _normalize_text(text)
        new_terms = frozenset(tokenize(text))
        for item in self._items:
            if _normalize_text(item.text) == normalized:
                return item
            if _jaccard(new_terms, frozenset(tokenize(item.text))) >= _DEDUP_JACCARD:
                return item
        return None

    def _score_candidates(
        self,
        candidates: Sequence[MemoryItem],
        query_terms: Sequence[str],
    ) -> list[tuple[MemoryItem, float]]:
        """按词法覆盖与重要度加权打分。"""
        query_set = frozenset(query_terms)
        if not query_set:
            return [(item, 0.0) for item in candidates]
        scored: list[tuple[MemoryItem, float]] = []
        for item in candidates:
            document_set = frozenset(tokenize(item.text))
            coverage = len(query_set & document_set) / len(query_set)
            importance = min(1.0, max(0.0, item.importance))
            scored.append(
                (item, coverage * (_LEXICAL_WEIGHT + _IMPORTANCE_WEIGHT * importance))
            )
        return scored
