"""语义记忆：BM25 词法检索与 JSONL 持久化。

本模块同时提供三层磁盘记忆共用的基础设施：:func:`tokenize` 分词器与
:class:`PersistentMemoryStore` 持久化基类；情节层与反思层只实现各自的打分公式。

分词规则（确定性）：ASCII 字母数字串整体小写后作为一个词；汉字（``U+4E00``–``U+9FFF``）
按出现顺序投影成序列，逐字产出单字，再对相邻两字产出双字（bigram）；投影忽略标点与空白，
因此 ``中文，检索`` 与 ``中文检索`` 产出同一组词。不做停用词过滤：常见词由 IDF 抑制。

BM25 公式（``Q`` 为去重后的查询词集合，语料为按 ``kinds``/``tags`` 过滤后的候选条目集合，
``N = |D|``）：

- ``idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))``，``df(t)`` 是包含 t 的条目数；
- ``tf_part(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl(d) / avgdl))``；
- ``raw = Σ_{t ∈ Q} idf(t) * tf_part(t, d)``；
- ``bm25 = raw / (1 + raw)``：把无界原始分压入 ``[0, 1)``，使
  :class:`agi_lab.memory.hub.MemoryHub` 的跨层合并排序有意义。

最终分数叠加轻微的时效与重要度加权（原始分为 0 时仍为 0）：

    score = bm25 * (1 + 0.1 * importance) * (1 + 0.05 * recency)
    recency = 0.5 ** (age_seconds / (30 * 86400))    # 半衰期 30 天，未来时间戳视为 0

权重刻意很小：排序仍由词法相关度主导，重要度与时效只用于打破接近的排名。
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from ..types import MemoryHit, MemoryItem, MemoryKind, MemoryQuery

_IMPORTANCE_WEIGHT = 0.1
_RECENCY_WEIGHT = 0.05
_RECENCY_HALF_LIFE_SECONDS = 30 * 86400.0

_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """把文本切成检索词序列。

    参数：
        text: 待切分文本。

    返回：
        词序列，保留重复以便统计词频；空文本返回空列表。

    规则：
        - ASCII 字母数字串整体转小写后作为一个词（``Hello`` → ``hello``）；
        - 汉字投影成单字序列后，逐字产出单字，再对相邻两字产出双字（bigram）；
        - 投影忽略标点与空白，所以 ``中文，检索`` 与 ``中文检索`` 结果相同；
        - 不过滤停用词，常见词由 BM25 的 IDF 抑制。
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    cjk_chars: list[str] = []
    for match in _TOKEN.finditer(lowered):
        token = match.group()
        tokens.append(token)
        if _CJK_CHAR.fullmatch(token):
            cjk_chars.append(token)
    tokens.extend(left + right for left, right in zip(cjk_chars, cjk_chars[1:]))
    return tokens


def _dump(item: MemoryItem) -> str:
    """把条目序列化为单行 JSON 文本（不换行，中文不转义）。"""
    return json.dumps(item.to_json(), ensure_ascii=False, sort_keys=True)


class PersistentMemoryStore:
    """三层磁盘记忆的公共基类：JSONL 载入/追加/重写与候选过滤。

    子类必须设置类属性 ``kind`` 并实现 :meth:`_score_candidates`。本类保证：

    - 构造时从 ``path`` 载入已有条目；非法 JSON 行或无法还原的行被跳过，
      数量记入 :attr:`bad_lines`（空白行不计）；
    - :meth:`add` 追加一行并立即 flush；``path`` 为 ``None`` 时纯内存、不写盘；
    - 载入与写入都把条目 kind 归一化为本层 ``kind``，保证单层存储只含一种类型。

    该类只在本包内复用，不属于公开 API。
    """

    kind: MemoryKind

    def __init__(self, path: Path | None = None) -> None:
        self._items: list[MemoryItem] = []
        self._index: dict[str, int] = {}
        self._path: Path | None = Path(path) if path is not None else None
        self.bad_lines = 0
        self._load()

    # ------------------------------------------------------------------
    # MemoryStore 接口
    # ------------------------------------------------------------------

    def add(self, item: MemoryItem) -> str:
        """写入一条记忆并持久化。

        参数：
            item: 待写入条目；``item.kind`` 与本层不一致时归一化为本层类型。

        返回：
            条目 id。

        异常：
            ValueError: 文本为空，或同 id 条目已存在（避免覆盖已持久化数据）。
        """
        if not item.text.strip():
            raise ValueError("记忆文本不能为空")
        if item.kind is not self.kind:
            item = replace(item, kind=self.kind)
        if item.id in self._index:
            raise ValueError(f"记忆 id 已存在：{item.id}")
        self._append(item)
        self._index[item.id] = len(self._items)
        self._items.append(item)
        return item.id

    def search(self, query: MemoryQuery) -> list[MemoryHit]:
        """按查询检索本层记忆。

        参数：
            query: 检索请求。``kinds`` 不含本层时返回空列表；``tags`` 非空时要求
                条目同时具备全部标签；``k < 1`` 返回空列表。

        返回：
            至多 ``k`` 条命中，按 ``(-score, -created_at, id)`` 排序。分数为 0 的
            条目（完全无关）不返回；``min_score`` 是额外下限。
        """
        if query.k <= 0:
            return []
        candidates = self._candidates(query)
        if not candidates:
            return []
        scored = self._score_candidates(candidates, tokenize(query.text))
        hits = [
            MemoryHit(item, score)
            for item, score in scored
            if score > 0.0 and score >= query.min_score
        ]
        hits.sort(key=lambda hit: (-hit.score, -hit.item.created_at, hit.item.id))
        return hits[: query.k]

    def recent(self, *, k: int = 10) -> list[MemoryItem]:
        """返回最近的 ``k`` 条记忆（新→旧）；``k < 1`` 返回空列表。"""
        if k <= 0:
            return []
        ordered = sorted(self._items, key=lambda item: (-item.created_at, item.id))
        return ordered[:k]

    def __len__(self) -> int:
        """返回已存条目数。"""
        return len(self._items)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """把内存中的全部条目重写回 JSONL；纯内存模式为空操作。"""
        if self._path is not None:
            self._rewrite()

    def _load(self) -> None:
        """从 JSONL 载入条目，坏行跳过并计数。"""
        path = self._path
        if path is None or not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                self.bad_lines += 1
                continue
            if not isinstance(payload, Mapping):
                self.bad_lines += 1
                continue
            try:
                item = MemoryItem.from_json(payload)
            except (TypeError, ValueError):
                self.bad_lines += 1
                continue
            item = replace(item, kind=self.kind)
            if item.id in self._index:
                self._items[self._index[item.id]] = item
            else:
                self._index[item.id] = len(self._items)
                self._items.append(item)

    def _append(self, item: MemoryItem) -> None:
        """追加一行并 flush；纯内存模式直接返回。"""
        path = self._path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_dump(item) + "\n")
            handle.flush()

    def _rewrite(self) -> None:
        """原子重写整个 JSONL 文件（先写临时文件再替换）。"""
        path = self._path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for item in self._items:
                    handle.write(_dump(item) + "\n")
                handle.flush()
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _replace_item(self, item: MemoryItem) -> None:
        """用同 id 的新版本替换内存条目并立即重写文件。"""
        position = self._index.get(item.id)
        if position is None:
            raise KeyError(f"未知记忆 id：{item.id}")
        self._items[position] = item
        if self._path is not None:
            self._rewrite()

    # ------------------------------------------------------------------
    # 子类扩展点
    # ------------------------------------------------------------------

    def _candidates(self, query: MemoryQuery) -> list[MemoryItem]:
        """按 ``kinds``/``tags`` 过滤出候选条目。"""
        if query.kinds is not None and self.kind not in query.kinds:
            return []
        if not query.tags:
            return list(self._items)
        required = tuple(query.tags)
        return [item for item in self._items if all(tag in item.tags for tag in required)]

    def _score_candidates(
        self,
        candidates: Sequence[MemoryItem],
        query_terms: Sequence[str],
    ) -> list[tuple[MemoryItem, float]]:
        """给候选条目打分，返回 ``(item, score)`` 列表；分数必须非负。"""
        raise NotImplementedError


class SemanticMemory(PersistentMemoryStore):
    """语义记忆：事实、文件摘要等知识条目，BM25 词法打分。

    继承 :class:`PersistentMemoryStore` 的 JSONL 持久化与过滤排序；分数公式见模块文档。
    """

    kind = MemoryKind.SEMANTIC

    def __init__(self, path: Path | None = None, *, k1: float = 1.5, b: float = 0.75) -> None:
        """构造语义记忆。

        参数：
            path: JSONL 文件路径；``None`` 表示纯内存。
            k1: BM25 词频饱和参数，必须 ``> 0``。
            b: BM25 长度归一化参数，必须在 ``[0, 1]`` 内。

        异常：
            ValueError: ``k1`` 或 ``b`` 越界。
        """
        if k1 <= 0:
            raise ValueError(f"k1 必须 > 0，收到 {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b 必须在 [0, 1] 内，收到 {b}")
        self.k1 = k1
        self.b = b
        super().__init__(path)

    def _score_candidates(
        self,
        candidates: Sequence[MemoryItem],
        query_terms: Sequence[str],
    ) -> list[tuple[MemoryItem, float]]:
        """在候选集合内计算 BM25，再叠加重要度与时效加权。"""
        if not candidates or not query_terms:
            return [(item, 0.0) for item in candidates]
        term_frequencies = [Counter(tokenize(item.text)) for item in candidates]
        lengths = [sum(frequencies.values()) for frequencies in term_frequencies]
        total = len(candidates)
        average_length = sum(lengths) / total
        document_frequency: Counter[str] = Counter()
        for frequencies in term_frequencies:
            document_frequency.update(frequencies.keys())
        unique_terms = list(dict.fromkeys(query_terms))
        now = time.time()
        scored: list[tuple[MemoryItem, float]] = []
        for item, frequencies, length in zip(candidates, term_frequencies, lengths):
            raw = 0.0
            length_ratio = length / average_length if average_length > 0 else 0.0
            for term in unique_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                frequency_in_documents = document_frequency[term]
                idf = math.log(
                    1.0 + (total - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
                )
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                raw += idf * frequency * (self.k1 + 1.0) / denominator
            bm25 = raw / (1.0 + raw) if raw > 0.0 else 0.0
            age = max(0.0, now - item.created_at)
            recency = 0.5 ** (age / _RECENCY_HALF_LIFE_SECONDS)
            score = bm25 * (1.0 + _IMPORTANCE_WEIGHT * item.importance) * (1.0 + _RECENCY_WEIGHT * recency)
            scored.append((item, score))
        return scored
