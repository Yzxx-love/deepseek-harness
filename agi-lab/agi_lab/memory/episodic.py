"""情节记忆：已完成步骤的执行摘要，按“近因 + 词法相关”检索。

打分公式（``Q`` 为查询词集合，``D`` 为条目词集合，``coverage = |Q ∩ D| / |Q|``）：

    recency = 0.5 ** (age_seconds / (7 * 86400))    # 半衰期 7 天，未来时间戳视为 0
    score   = 0.7 * coverage + 0.3 * recency

词法覆盖占七成，保证与查询相关的摘要优先；近因占三成，使足够新的摘要即使没有词面
重叠也会以较低分返回，空查询因此退化为纯“按新→旧”。持久化复用
:class:`agi_lab.memory.semantic.PersistentMemoryStore`（JSONL，坏行跳过）。
"""

from __future__ import annotations

import time
from typing import Sequence

from ..types import MemoryItem, MemoryKind
from .semantic import PersistentMemoryStore, tokenize

_LEXICAL_WEIGHT = 0.7
_RECENCY_WEIGHT = 0.3
_RECENCY_HALF_LIFE_SECONDS = 7 * 86400.0


class EpisodicMemory(PersistentMemoryStore):
    """情节记忆：执行摘要条目，JSONL 持久化到 ``episodic.jsonl``。"""

    kind = MemoryKind.EPISODIC

    def _score_candidates(
        self,
        candidates: Sequence[MemoryItem],
        query_terms: Sequence[str],
    ) -> list[tuple[MemoryItem, float]]:
        """按词法覆盖与时间近因加权打分；不涉及重要度。"""
        query_set = frozenset(query_terms)
        now = time.time()
        scored: list[tuple[MemoryItem, float]] = []
        for item in candidates:
            document_set = frozenset(tokenize(item.text))
            coverage = len(query_set & document_set) / len(query_set) if query_set else 0.0
            age = max(0.0, now - item.created_at)
            recency = 0.5 ** (age / _RECENCY_HALF_LIFE_SECONDS)
            scored.append((item, _LEXICAL_WEIGHT * coverage + _RECENCY_WEIGHT * recency))
        return scored
