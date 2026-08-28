"""记忆层单元测试：四层存储、MemoryHub 门面与 JSONL 持久化。

全部使用标准库 unittest 与 ``tempfile.TemporaryDirectory()``，不联网、不 sleep；
时间戳显式传入，保证断言确定性。
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from agi_lab.memory import (
    EpisodicMemory,
    MemoryHub,
    ReflectionMemory,
    SemanticMemory,
    WorkingMemory,
    tokenize,
)
from agi_lab.types import (
    JSONObject,
    MemoryItem,
    MemoryKind,
    MemoryQuery,
    MemoryStore,
    Observation,
    Plan,
    PlanStep,
    StepStatus,
)

_SECONDS_PER_DAY = 86400.0


def make_item(
    text: str,
    kind: MemoryKind,
    *,
    importance: float = 0.5,
    created_at: float | None = None,
    tags: tuple[str, ...] = (),
    metadata: JSONObject | None = None,
) -> MemoryItem:
    """构造测试用记忆条目；``created_at`` 缺省取当前时间。"""
    return MemoryItem(
        text=text,
        kind=kind,
        tags=tags,
        importance=importance,
        created_at=time.time() if created_at is None else created_at,
        metadata=dict(metadata) if metadata is not None else {},
    )


def make_observation(
    content: str,
    *,
    source: str = "tool",
    step_id: str = "",
    ok: bool = True,
    untrusted: bool = True,
) -> Observation:
    """构造测试用观测。"""
    return Observation(
        source=source,
        content=content,
        ok=ok,
        step_id=step_id,
        untrusted=untrusted,
    )


def make_plan() -> Plan:
    """构造一个两步计划。"""
    return Plan(
        goal="完成报告",
        steps=[
            PlanStep(id="s1", description="读取数据"),
            PlanStep(id="s2", description="生成报告", depends_on=("s1",)),
        ],
    )


class TestTokenize(unittest.TestCase):
    """分词器行为。"""

    def test_ascii_words_are_lowercased(self) -> None:
        self.assertEqual(tokenize("Hello WORLD 42"), ["hello", "world", "42"])

    def test_chinese_unigrams_and_bigrams(self) -> None:
        self.assertEqual(
            tokenize("中文检索"),
            ["中", "文", "检", "索", "中文", "文检", "检索"],
        )

    def test_punctuation_does_not_split_chinese_bigrams(self) -> None:
        self.assertEqual(tokenize("中文，检索"), tokenize("中文检索"))

    def test_mixed_scripts(self) -> None:
        self.assertEqual(tokenize("Python 缩进"), ["python", "缩", "进", "缩进"])

    def test_empty_text(self) -> None:
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   "), [])
        self.assertEqual(tokenize("，。！"), [])

    def test_deterministic(self) -> None:
        self.assertEqual(tokenize("缓存 cache 策略"), tokenize("缓存 cache 策略"))


class TestPersistence(unittest.TestCase):
    """三层磁盘记忆共用的 JSONL 持久化行为。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_jsonl_roundtrip_for_all_persistent_layers(self) -> None:
        layers = (
            (SemanticMemory, MemoryKind.SEMANTIC),
            (EpisodicMemory, MemoryKind.EPISODIC),
            (ReflectionMemory, MemoryKind.REFLECTION),
        )
        for index, (store_type, kind) in enumerate(layers):
            with self.subTest(layer=kind.value):
                path = self.root / f"{kind.value}.jsonl"
                store = store_type(path)
                item = make_item(
                    f"{kind.value} 记忆条目 {index}",
                    kind,
                    tags=("t1",),
                    metadata={"source": "test"},
                )
                item_id = store.add(item)
                self.assertTrue(path.exists())
                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 1)
                payload = json.loads(lines[0])
                self.assertEqual(payload["id"], item_id)
                reloaded = store_type(path)
                self.assertEqual(len(reloaded), 1)
                self.assertEqual(reloaded.bad_lines, 0)
                restored = reloaded.recent()[0]
                self.assertEqual(restored.id, item_id)
                self.assertEqual(restored.text, item.text)
                self.assertEqual(restored.kind, kind)
                self.assertEqual(restored.tags, ("t1",))
                self.assertEqual(restored.metadata, {"source": "test"})

    def test_add_appends_one_line_immediately(self) -> None:
        path = self.root / "semantic.jsonl"
        store = SemanticMemory(path)
        store.add(make_item("第一条", MemoryKind.SEMANTIC))
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
        store.add(make_item("第二条", MemoryKind.SEMANTIC))
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_kind_is_normalized_to_layer(self) -> None:
        store = EpisodicMemory()
        item_id = store.add(make_item("跨层写入的条目", MemoryKind.SEMANTIC))
        restored = store.recent()[0]
        self.assertEqual(restored.id, item_id)
        self.assertEqual(restored.kind, MemoryKind.EPISODIC)

    def test_bad_jsonl_lines_are_skipped_and_counted(self) -> None:
        path = self.root / "semantic.jsonl"
        valid = make_item("有效条目 路径", MemoryKind.SEMANTIC)
        path.write_text(
            json.dumps(valid.to_json(), ensure_ascii=False)
            + "\n"
            + "{not json}\n"
            + "\n"
            + json.dumps({"kind": "no-such-kind", "text": "坏条目"}, ensure_ascii=False)
            + "\n"
            + json.dumps([1, 2, 3])
            + "\n",
            encoding="utf-8",
        )
        store = SemanticMemory(path)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.bad_lines, 3)
        self.assertEqual(store.recent()[0].text, "有效条目 路径")

    def test_empty_text_and_duplicate_id_are_rejected(self) -> None:
        store = SemanticMemory()
        with self.assertRaises(ValueError):
            store.add(make_item("   ", MemoryKind.SEMANTIC))
        item = make_item("唯一条目", MemoryKind.SEMANTIC)
        store.add(item)
        with self.assertRaises(ValueError):
            store.add(item)
        self.assertEqual(len(store), 1)

    def test_flush_rewrites_the_whole_file(self) -> None:
        path = self.root / "semantic.jsonl"
        store = SemanticMemory(path)
        store.add(make_item("需要保住的条目", MemoryKind.SEMANTIC))
        path.write_text("", encoding="utf-8")
        store.flush()
        reloaded = SemanticMemory(path)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded.recent()[0].text, "需要保住的条目")

    def test_semantic_parameters_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            SemanticMemory(k1=0)
        with self.assertRaises(ValueError):
            SemanticMemory(b=-0.1)
        with self.assertRaises(ValueError):
            SemanticMemory(b=1.1)


class TestSemanticMemory(unittest.TestCase):
    """BM25 语义记忆。"""

    def test_more_relevant_entry_scores_higher(self) -> None:
        common = 1_700_000_000.0
        relevant = "Python 缩进错误会导致语法错误"
        weak = "Python 是一种编程语言"
        store = SemanticMemory()
        store.add(make_item(relevant, MemoryKind.SEMANTIC, created_at=common))
        store.add(make_item(weak, MemoryKind.SEMANTIC, created_at=common))
        hits = store.search(MemoryQuery(text="Python 缩进错误"))
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].item.text, relevant)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_chinese_query_hits_chinese_fact(self) -> None:
        store = SemanticMemory()
        store.add(make_item("文件路径必须位于工作区根目录内", MemoryKind.SEMANTIC))
        store.add(make_item("今天天气很好", MemoryKind.SEMANTIC))
        hits = store.search(MemoryQuery(text="工作区路径"))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].item.text, "文件路径必须位于工作区根目录内")

    def test_tags_and_kinds_filters(self) -> None:
        store = SemanticMemory()
        store.add(make_item("根目录内的文件读取", MemoryKind.SEMANTIC, tags=("files",)))
        store.add(make_item("根目录内的 shell 命令", MemoryKind.SEMANTIC, tags=("shell",)))
        files_hits = store.search(MemoryQuery(text="根目录", tags=("files",)))
        self.assertEqual(len(files_hits), 1)
        self.assertIn("files", files_hits[0].item.tags)
        self.assertEqual(store.search(MemoryQuery(text="根目录", kinds=(MemoryKind.EPISODIC,))), [])
        self.assertEqual(len(store.search(MemoryQuery(text="根目录", kinds=(MemoryKind.SEMANTIC,)))), 2)

    def test_k_truncation_min_score_and_ordering(self) -> None:
        store = SemanticMemory()
        for index in range(4):
            store.add(
                make_item(
                    f"缓存策略 {index}",
                    MemoryKind.SEMANTIC,
                    created_at=1_700_000_000.0 + index,
                )
            )
        self.assertEqual(len(store.search(MemoryQuery(text="缓存策略", k=2))), 2)
        self.assertEqual(store.search(MemoryQuery(text="缓存策略", k=0)), [])
        ranked = store.search(MemoryQuery(text="缓存策略", k=10))
        scores = [hit.score for hit in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        strict = store.search(MemoryQuery(text="缓存策略", min_score=ranked[0].score + 0.001))
        self.assertEqual(strict, [])

    def test_importance_breaks_ties(self) -> None:
        common = 1_700_000_000.0
        store = SemanticMemory()
        store.add(
            make_item("cache invalidation strategy", MemoryKind.SEMANTIC, importance=0.1, created_at=common)
        )
        store.add(
            make_item("cache invalidation strategy", MemoryKind.SEMANTIC, importance=0.9, created_at=common)
        )
        hits = store.search(MemoryQuery(text="cache invalidation"))
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].item.importance, 0.9)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_newer_entry_scores_higher(self) -> None:
        now = time.time()
        store = SemanticMemory()
        store.add(
            make_item("stale memory entry", MemoryKind.SEMANTIC, created_at=now - 400 * _SECONDS_PER_DAY)
        )
        store.add(make_item("stale memory entry", MemoryKind.SEMANTIC, created_at=now))
        hits = store.search(MemoryQuery(text="stale memory"))
        self.assertEqual(len(hits), 2)
        self.assertGreater(hits[0].score, hits[1].score)
        self.assertGreater(hits[0].item.created_at, hits[1].item.created_at)

    def test_empty_store_is_safe(self) -> None:
        store = SemanticMemory()
        self.assertEqual(len(store), 0)
        self.assertEqual(store.recent(), [])
        self.assertEqual(store.search(MemoryQuery(text="任意查询")), [])
        self.assertEqual(store.search(MemoryQuery(text="")), [])
        store.flush()

    def test_recent_orders_newest_first(self) -> None:
        store = SemanticMemory()
        for offset in range(3):
            store.add(make_item(f"条目 {offset}", MemoryKind.SEMANTIC, created_at=100.0 + offset))
        self.assertEqual([item.text for item in store.recent(k=2)], ["条目 2", "条目 1"])
        self.assertEqual(store.recent(k=0), [])
        self.assertEqual(len(store.recent()), 3)


class TestEpisodicMemory(unittest.TestCase):
    """近因 + 词法相关的情节检索。"""

    def test_lexical_relevance_outweighs_staleness(self) -> None:
        now = time.time()
        store = EpisodicMemory()
        store.add(
            make_item("读取配置文件", MemoryKind.EPISODIC, created_at=now - 400 * _SECONDS_PER_DAY)
        )
        store.add(make_item("生成报告", MemoryKind.EPISODIC, created_at=now))
        hits = store.search(MemoryQuery(text="配置文件"))
        self.assertEqual([hit.item.text for hit in hits], ["读取配置文件", "生成报告"])

    def test_newer_identical_summary_scores_higher(self) -> None:
        now = time.time()
        store = EpisodicMemory()
        store.add(make_item("执行摘要", MemoryKind.EPISODIC, created_at=now - 30 * _SECONDS_PER_DAY))
        store.add(make_item("执行摘要", MemoryKind.EPISODIC, created_at=now))
        hits = store.search(MemoryQuery(text="执行摘要"))
        self.assertEqual(len(hits), 2)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_empty_query_falls_back_to_recency(self) -> None:
        now = time.time()
        store = EpisodicMemory()
        store.add(make_item("旧摘要", MemoryKind.EPISODIC, created_at=now - 3 * _SECONDS_PER_DAY))
        store.add(make_item("新摘要", MemoryKind.EPISODIC, created_at=now))
        hits = store.search(MemoryQuery(text=""))
        self.assertEqual([hit.item.text for hit in hits], ["新摘要", "旧摘要"])

    def test_empty_store_is_safe(self) -> None:
        store = EpisodicMemory()
        self.assertEqual(len(store), 0)
        self.assertEqual(store.search(MemoryQuery(text="任意")), [])
        self.assertEqual(store.recent(), [])


class TestReflectionMemory(unittest.TestCase):
    """教训去重、重要度提升与检索。"""

    def test_normalized_duplicate_is_merged(self) -> None:
        store = ReflectionMemory()
        first_id = store.add(make_item("Always check file paths", MemoryKind.REFLECTION, importance=0.4))
        second_id = store.add(
            make_item("  always  CHECK file paths.  ", MemoryKind.REFLECTION, importance=0.4)
        )
        self.assertEqual(first_id, second_id)
        self.assertEqual(len(store), 1)
        self.assertAlmostEqual(store.recent()[0].importance, 0.5)

    def test_jaccard_duplicate_is_merged(self) -> None:
        store = ReflectionMemory()
        first_id = store.add(make_item("check path, always", MemoryKind.REFLECTION))
        second_id = store.add(make_item("always check path", MemoryKind.REFLECTION))
        self.assertEqual(first_id, second_id)
        self.assertEqual(len(store), 1)

    def test_chinese_near_duplicate_is_merged(self) -> None:
        store = ReflectionMemory()
        store.add(make_item("路径检查必须失败重试", MemoryKind.REFLECTION))
        store.add(make_item("路径检查必须失败重试啊", MemoryKind.REFLECTION))
        self.assertEqual(len(store), 1)

    def test_distinct_lessons_are_kept(self) -> None:
        store = ReflectionMemory()
        store.add(make_item("read the file", MemoryKind.REFLECTION))
        store.add(make_item("write the file", MemoryKind.REFLECTION))
        self.assertEqual(len(store), 2)

    def test_duplicate_promotion_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reflection.jsonl"
            store = ReflectionMemory(path)
            item_id = store.add(make_item("检查路径后再写入", MemoryKind.REFLECTION, importance=0.4))
            for _ in range(2):
                self.assertEqual(
                    store.add(make_item("检查路径后再写入", MemoryKind.REFLECTION, importance=0.4)),
                    item_id,
                )
            self.assertAlmostEqual(store.recent()[0].importance, 0.6)
            reloaded = ReflectionMemory(path)
            self.assertEqual(len(reloaded), 1)
            self.assertAlmostEqual(reloaded.recent()[0].importance, 0.6)

    def test_importance_weight_affects_ranking(self) -> None:
        store = ReflectionMemory()
        store.add(make_item("path missing error", MemoryKind.REFLECTION, importance=0.0))
        store.add(make_item("path missing warning", MemoryKind.REFLECTION, importance=1.0))
        hits = store.search(MemoryQuery(text="path missing"))
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].item.importance, 1.0)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_unrelated_lesson_is_not_returned(self) -> None:
        store = ReflectionMemory()
        store.add(make_item("回答要简洁", MemoryKind.REFLECTION, importance=1.0))
        self.assertEqual(store.search(MemoryQuery(text="文件路径")), [])
        self.assertEqual(store.search(MemoryQuery(text="")), [])

    def test_empty_store_is_safe(self) -> None:
        store = ReflectionMemory()
        self.assertEqual(len(store), 0)
        self.assertEqual(store.recent(), [])
        self.assertEqual(store.search(MemoryQuery(text="任意")), [])


class TestWorkingMemory(unittest.TestCase):
    """工作记忆的容量、渲染与快照。"""

    def test_record_keeps_most_recent_items(self) -> None:
        memory = WorkingMemory(max_items=3, max_chars=100_000)
        for index in range(5):
            memory.record(make_observation(f"obs-{index}"))
        self.assertEqual(
            [observation.content for observation in memory.observations()],
            ["obs-2", "obs-3", "obs-4"],
        )

    def test_observations_limit(self) -> None:
        memory = WorkingMemory()
        for index in range(4):
            memory.record(make_observation(f"obs-{index}"))
        self.assertEqual(
            [observation.content for observation in memory.observations(2)],
            ["obs-2", "obs-3"],
        )
        self.assertEqual(memory.observations(0), [])
        self.assertEqual(len(memory.observations()), 4)

    def test_char_limit_drops_oldest(self) -> None:
        memory = WorkingMemory(max_items=10, max_chars=30)
        memory.record(make_observation("a" * 20))
        memory.record(make_observation("b" * 20))
        self.assertEqual([observation.content for observation in memory.observations()], ["b" * 20])
        memory.record(make_observation("c" * 20))
        self.assertEqual([observation.content for observation in memory.observations()], ["c" * 20])

    def test_single_oversized_observation_is_kept(self) -> None:
        memory = WorkingMemory(max_items=2, max_chars=5)
        memory.record(make_observation("x" * 100))
        self.assertEqual(len(memory.observations()), 1)

    def test_render_contains_goal_plan_and_observations(self) -> None:
        memory = WorkingMemory()
        memory.set_goal("完成报告")
        plan = make_plan()
        memory.set_plan(plan)
        memory.record(make_observation("读取数据成功", source="read_file", step_id="s1"))
        rendered = memory.render()
        self.assertIn("目标：完成报告", rendered)
        self.assertIn("计划 v1（0/2 完成）", rendered)
        self.assertIn("读取数据", rendered)
        self.assertIn("读取数据成功", rendered)
        plan.mark("s1", StepStatus.DONE)
        self.assertIn("[done] s1", memory.render())

    def test_render_respects_budget(self) -> None:
        memory = WorkingMemory()
        memory.set_goal("目标" * 100)
        memory.record(make_observation("观测" * 100))
        rendered = memory.render(max_chars=40)
        self.assertLessEqual(len(rendered), 40)
        self.assertTrue(rendered.startswith("目标："))
        self.assertTrue(rendered.endswith("…"))
        self.assertEqual(memory.render(max_chars=0), "")

    def test_render_orders_observations_newest_first(self) -> None:
        memory = WorkingMemory()
        memory.record(make_observation("first"))
        memory.record(make_observation("second"))
        rendered = memory.render()
        self.assertLess(rendered.index("second"), rendered.index("first"))

    def test_render_marks_untrusted_observations(self) -> None:
        memory = WorkingMemory()
        memory.record(make_observation("来自工具的输出", untrusted=True, ok=False))
        rendered = memory.render()
        self.assertIn("不可信数据", rendered)
        self.assertIn("失败", rendered)

    def test_clear_resets_state(self) -> None:
        memory = WorkingMemory()
        memory.set_goal("目标")
        memory.set_plan(make_plan())
        memory.record(make_observation("观测"))
        memory.clear()
        self.assertEqual(memory.goal, "")
        self.assertIsNone(memory.plan)
        self.assertEqual(memory.observations(), [])
        self.assertEqual(memory.render(), "")

    def test_snapshot_is_json_serializable(self) -> None:
        memory = WorkingMemory(max_items=2, max_chars=100)
        memory.set_goal("目标")
        memory.set_plan(make_plan())
        memory.record(make_observation("观测", step_id="s1"))
        snapshot = memory.snapshot()
        json.dumps(snapshot, ensure_ascii=False)
        self.assertEqual(snapshot["goal"], "目标")
        self.assertEqual(snapshot["count"], 1)
        self.assertEqual(snapshot["max_items"], 2)
        plan_payload = snapshot["plan"]
        self.assertIsInstance(plan_payload, dict)
        assert isinstance(plan_payload, dict)
        self.assertEqual(plan_payload["goal"], "完成报告")

    def test_invalid_limits_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WorkingMemory(max_items=0)
        with self.assertRaises(ValueError):
            WorkingMemory(max_chars=0)


class TestStoreProtocol(unittest.TestCase):
    """三层磁盘记忆与冻结的 ``MemoryStore`` Protocol 保持一致。"""

    def test_persistent_layers_implement_memory_store(self) -> None:
        for store in (SemanticMemory(), EpisodicMemory(), ReflectionMemory()):
            with self.subTest(store=type(store).__name__):
                self.assertIsInstance(store, MemoryStore)

    def test_layer_kinds_are_distinct(self) -> None:
        self.assertEqual(SemanticMemory.kind, MemoryKind.SEMANTIC)
        self.assertEqual(EpisodicMemory.kind, MemoryKind.EPISODIC)
        self.assertEqual(ReflectionMemory.kind, MemoryKind.REFLECTION)

    def test_working_memory_is_not_a_memory_store(self) -> None:
        self.assertNotIsInstance(WorkingMemory(), MemoryStore)


class TestMemoryHub(unittest.TestCase):
    """门面：装配、跨层合并、渲染与持久化开关。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "memory"

    @staticmethod
    def _populate(hub: MemoryHub) -> tuple[str, str, str]:
        """向三层各写一条与“缓存”相关的记忆。"""
        lesson_id = hub.add_lesson("缓存失效通常源于键设计不一致", tags=("cache",), task_id="t1")
        fact_id = hub.add_fact("缓存可以降低数据库访问延迟", tags=("cache",), source="doc")
        episode_id = hub.add_episode("加入缓存后响应时间下降", step_id="s2", tags=("cache",))
        return lesson_id, fact_id, episode_id

    def test_working_memory_delegation(self) -> None:
        hub = MemoryHub()
        hub.set_goal("目标 A")
        hub.set_plan(make_plan())
        hub.record_observation("s1", make_observation("工具输出"))
        self.assertEqual(hub.working.goal, "目标 A")
        self.assertIsNotNone(hub.working.plan)
        self.assertEqual(hub.working.observations()[-1].step_id, "s1")
        hub.record_observation("", make_observation("自带步骤号", step_id="s0"))
        self.assertEqual(hub.working.observations()[-1].step_id, "s0")

    def test_add_writes_tags_importance_and_metadata(self) -> None:
        hub = MemoryHub(self.root)
        lesson_id, fact_id, episode_id = self._populate(hub)
        self.assertTrue(lesson_id and fact_id and episode_id)
        lesson = hub.reflection.recent()[0]
        self.assertEqual(lesson.metadata, {"task_id": "t1"})
        self.assertEqual(lesson.tags, ("cache",))
        self.assertAlmostEqual(lesson.importance, 0.8)
        fact = hub.semantic.recent()[0]
        self.assertEqual(fact.metadata, {"source": "doc"})
        self.assertAlmostEqual(fact.importance, 0.6)
        episode = hub.episodic.recent()[0]
        self.assertEqual(episode.metadata, {"step_id": "s2"})
        self.assertAlmostEqual(episode.importance, 0.4)

    def test_search_merges_layers_sorted_by_score(self) -> None:
        hub = MemoryHub()
        self._populate(hub)
        hits = hub.search("缓存", k=10)
        self.assertEqual(len(hits), 3)
        self.assertEqual(
            {hit.item.kind for hit in hits},
            {MemoryKind.REFLECTION, MemoryKind.SEMANTIC, MemoryKind.EPISODIC},
        )
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(hub.search("缓存", k=2)), 2)
        self.assertEqual(hub.search("缓存", k=0), [])

    def test_search_kinds_filter(self) -> None:
        hub = MemoryHub()
        self._populate(hub)
        lesson_hits = hub.search("缓存", kinds=(MemoryKind.REFLECTION,))
        self.assertEqual(len(lesson_hits), 1)
        self.assertEqual(lesson_hits[0].item.kind, MemoryKind.REFLECTION)
        self.assertEqual(hub.search("缓存", kinds=(MemoryKind.WORKING,)), [])
        self.assertEqual(hub.search("缓存", kinds=()), [])

    def test_search_dedupes_same_text(self) -> None:
        hub = MemoryHub()
        hub.add_fact("重复的事实文本")
        hub.add_fact("重复的事实文本")
        self.assertEqual(len(hub.semantic), 2)
        hits = hub.search("重复的事实")
        matching = [hit for hit in hits if hit.item.text == "重复的事实文本"]
        self.assertEqual(len(matching), 1)

    def test_empty_hub_is_safe(self) -> None:
        hub = MemoryHub()
        self.assertEqual(hub.search("任意"), [])
        self.assertEqual(hub.render_context(), "")
        self.assertEqual(hub.stats()["total"], 0)
        hub.flush()

    def test_render_context_uses_priority_order(self) -> None:
        hub = MemoryHub()
        self._populate(hub)
        rendered = hub.render_context(query="缓存")
        self.assertIn("[教训]", rendered)
        self.assertIn("[事实]", rendered)
        self.assertIn("[情节]", rendered)
        self.assertLess(rendered.index("[教训]"), rendered.index("[事实]"))
        self.assertLess(rendered.index("[事实]"), rendered.index("[情节]"))

    def test_render_context_truncates_to_budget(self) -> None:
        hub = MemoryHub()
        hub.add_lesson("很长" * 200)
        hub.add_fact("事实" * 200)
        hub.add_episode("情节" * 200)
        rendered = hub.render_context(max_chars=80)
        self.assertLessEqual(len(rendered), 80)
        self.assertEqual(hub.render_context(max_chars=0), "")

    def test_render_context_query_filters_unrelated(self) -> None:
        hub = MemoryHub()
        hub.add_lesson("文件路径必须限制在工作区内")
        hub.add_lesson("回答要保持简洁")
        rendered = hub.render_context(query="文件路径")
        self.assertIn("文件路径必须限制在工作区内", rendered)
        self.assertNotIn("回答要保持简洁", rendered)

    def test_render_context_without_query_uses_priority(self) -> None:
        hub = MemoryHub()
        self._populate(hub)
        rendered = hub.render_context()
        self.assertIn("缓存失效通常源于键设计不一致", rendered)
        self.assertIn("缓存可以降低数据库访问延迟", rendered)
        self.assertIn("加入缓存后响应时间下降", rendered)

    def test_persist_roundtrip(self) -> None:
        hub = MemoryHub(self.root)
        self.assertTrue(hub.persistent)
        self._populate(hub)
        for name in ("episodic.jsonl", "semantic.jsonl", "reflection.jsonl"):
            self.assertTrue((self.root / name).exists(), name)
        hub.flush()
        reloaded = MemoryHub(self.root)
        self.assertTrue(reloaded.persistent)
        self.assertEqual(reloaded.stats()["total"], 3)
        self.assertEqual(len(reloaded.search("缓存", k=10)), 3)

    def test_persist_false_writes_nothing(self) -> None:
        hub = MemoryHub(self.root, persist=False)
        self.assertFalse(hub.persistent)
        self.assertIsNone(hub.root)
        self._populate(hub)
        hub.flush()
        self.assertFalse(self.root.exists())
        self.assertEqual(len(hub.search("缓存", k=10)), 3)
        fresh = MemoryHub(self.root, persist=False)
        self.assertEqual(fresh.stats()["total"], 0)

    def test_memory_only_when_root_is_none(self) -> None:
        hub = MemoryHub()
        self.assertFalse(hub.persistent)
        self.assertIsNone(hub.root)
        hub.add_fact("无盘事实")
        self.assertEqual(len(hub.search("无盘事实")), 1)
        hub.flush()

    def test_bad_jsonl_line_is_counted(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        valid = make_item("有效的事实", MemoryKind.SEMANTIC)
        (self.root / "semantic.jsonl").write_text(
            json.dumps(valid.to_json(), ensure_ascii=False) + "\n" + "损坏的行\n",
            encoding="utf-8",
        )
        hub = MemoryHub(self.root)
        self.assertEqual(len(hub.semantic), 1)
        self.assertEqual(hub.semantic.bad_lines, 1)
        self.assertEqual(hub.stats()["bad_lines"], 1)

    def test_snapshot_and_stats_are_json_serializable(self) -> None:
        hub = MemoryHub()
        self._populate(hub)
        hub.set_goal("目标")
        hub.record_observation("s1", make_observation("观测"))
        snapshot = hub.snapshot()
        json.dumps(snapshot, ensure_ascii=False)
        working_payload = snapshot["working"]
        semantic_payload = snapshot["semantic"]
        self.assertIsInstance(working_payload, dict)
        self.assertIsInstance(semantic_payload, dict)
        assert isinstance(working_payload, dict) and isinstance(semantic_payload, dict)
        self.assertEqual(working_payload["count"], 1)
        self.assertEqual(semantic_payload["count"], 1)
        stats = hub.stats()
        self.assertEqual(stats["episodic"], 1)
        self.assertEqual(stats["semantic"], 1)
        self.assertEqual(stats["reflection"], 1)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["working_observations"], 1)

    def test_empty_text_is_rejected(self) -> None:
        hub = MemoryHub()
        with self.assertRaises(ValueError):
            hub.add_fact("  ")
        with self.assertRaises(ValueError):
            hub.add_lesson("")
        with self.assertRaises(ValueError):
            hub.add_episode("")

    def test_hub_lesson_dedupe_promotes_importance(self) -> None:
        hub = MemoryHub()
        first = hub.add_lesson("路径检查必须失败重试")
        second = hub.add_lesson("路径检查必须失败重试啊")
        self.assertEqual(first, second)
        self.assertEqual(len(hub.reflection), 1)
        self.assertGreater(hub.reflection.recent()[0].importance, 0.8)


if __name__ == "__main__":
    unittest.main()
