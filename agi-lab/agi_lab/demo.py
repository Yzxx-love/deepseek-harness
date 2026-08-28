"""离线机制演示任务集。

四个演示各证明一类运行时机制，全部使用 :class:`agi_lab.llm.mock.MockLLM` 的脚本化
输出，因此 **离线、确定性、无需 API key**：

1. ``arithmetic``——工具调用闭环：模型必须调用 calculator 而不能心算；
2. ``file-report``——多步计划与文件读写：读 CSV → 算总额 → 写报告；
3. ``self-heal``——失败恢复：连续两次工具失败 → 反思 → 计划补丁 → 成功；
4. ``safety``——安全强制：越界路径被运行时拒绝，且该拒绝被记录为安全事件。

重要：脚本是预先写好的动作序列，**不是模型推理**。演示证明的是运行时机制正确，
不证明智能体有能力；能力评测必须用真实 provider，见 ``docs/04-evaluation.md``。
每个任务的 ``metadata["expected_status"]`` 是演示期望的终态，供 CLI 判定退出码。
"""

from __future__ import annotations

from .types import TaskSpec


def demo_tasks() -> list[TaskSpec]:
    """返回全部离线演示任务。"""
    return [_arithmetic(), _file_report(), _self_heal(), _safety()]


def find_demo_task(task_id: str) -> TaskSpec | None:
    """按 id 查找演示任务，找不到返回 ``None``。"""
    for task in demo_tasks():
        if task.id == task_id:
            return task
    return None


def _arithmetic() -> TaskSpec:
    goal = "计算 (137*29-61)/4 的精确值。必须使用 calculator 工具，不要心算。"
    script = {
        "planner": [
            {
                "goal": goal,
                "steps": [
                    {
                        "id": "s1",
                        "description": "用 calculator 计算表达式 (137*29-61)/4",
                        "success_criteria": "得到数值结果 978",
                        "depends_on": [],
                    }
                ],
                "notes": "单步计算任务",
            }
        ],
        "actor": [
            {
                "thought": "先调用 calculator 做精确计算",
                "action": {"type": "tool", "tool": "calculator", "arguments": {"expression": "(137*29-61)/4"}},
            },
            {"thought": "已得到计算结果", "action": {"type": "final", "answer": "(137*29-61)/4 = 978"}},
        ],
        "critic_step": [
            {"ok": True, "missing": [], "reason": "观测包含 calculator 返回的 978，且表达式与目标一致"}
        ],
        "critic_final": [
            {
                "complete": True,
                "confidence": 0.95,
                "answer": "(137*29-61)/4 = 978",
                "missing": [],
                "reason": "计划已完成，答案与工具输出一致",
            }
        ],
    }
    return TaskSpec(
        id="arithmetic",
        goal=goal,
        mock_script=script,
        metadata={"expected_status": "success", "title": "工具调用闭环", "shows": "calculator 调用与结果回填"},
    )


def _file_report() -> TaskSpec:
    goal = "读取工作区的 sales.csv，计算总销售额（price×qty 之和），把结果写入 report.md。"
    sales_csv = "item,price,qty\nA,10,3\nB,5,4\nC,20,1\n"
    report_md = (
        "# 销售报告\n\n总销售额：70\n\n明细：\n"
        "- A: 10 × 3 = 30\n- B: 5 × 4 = 20\n- C: 20 × 1 = 20\n"
    )
    script = {
        "planner": [
            {
                "goal": goal,
                "steps": [
                    {
                        "id": "s1",
                        "description": "用 read_file 读取 sales.csv 并计算 price×qty 之和",
                        "success_criteria": "得到总销售额数值",
                        "depends_on": [],
                    },
                    {
                        "id": "s2",
                        "description": "用 write_file 把总销售额写入 report.md",
                        "success_criteria": "report.md 存在且包含总销售额",
                        "depends_on": ["s1"],
                    },
                ],
                "notes": "先读后写，两步有依赖",
            }
        ],
        "actor": [
            {
                "thought": "先读取 CSV 内容",
                "action": {"type": "tool", "tool": "read_file", "arguments": {"path": "sales.csv"}},
            },
            {
                "thought": "三行数据合计 30+20+20=70",
                "action": {"type": "final", "answer": "sales.csv 共 3 行；总销售额 = 30+20+20 = 70"},
            },
            {
                "thought": "把结果写入报告文件",
                "action": {
                    "type": "tool",
                    "tool": "write_file",
                    "arguments": {"path": "report.md", "content": report_md},
                },
            },
            {"thought": "报告已写入", "action": {"type": "final", "answer": "总销售额 70 已写入 report.md"}},
        ],
        "critic_step": [
            {"ok": True, "missing": [], "reason": "read_file 返回了 CSV 三行数据"},
            {"ok": True, "missing": [], "reason": "write_file 返回成功写入字节数"},
        ],
        "critic_final": [
            {
                "complete": True,
                "confidence": 0.9,
                "answer": "总销售额 70 已写入 report.md",
                "missing": [],
                "reason": "两步计划均完成，报告文件已生成",
            }
        ],
    }
    return TaskSpec(
        id="file-report",
        goal=goal,
        initial_files={"sales.csv": sales_csv},
        mock_script=script,
        metadata={"expected_status": "success", "title": "多步计划与文件读写", "shows": "依赖步骤、读写工具"},
    )


def _self_heal() -> TaskSpec:
    goal = "读取工作区的 notes.txt 并总结要点。"
    script = {
        "planner": [
            {
                "goal": goal,
                "steps": [
                    {
                        "id": "s1",
                        "description": "用 read_file 读取 notes.txt 并总结要点",
                        "success_criteria": "读到文件内容并给出三条要点",
                        "depends_on": [],
                    }
                ],
                "notes": "先直接读取",
            }
        ],
        "actor": [
            {
                "thought": "直接读取 notes.txt",
                "action": {"type": "tool", "tool": "read_file", "arguments": {"path": "note.txt"}},
            },
            {"thought": "文件读到了", "action": {"type": "final", "answer": "已读取 notes.txt，要点为三条"}},
            {
                "thought": "文件名应该没问题，再试一次",
                "action": {"type": "tool", "tool": "read_file", "arguments": {"path": "note.txt"}},
            },
            {"thought": "这次读到了", "action": {"type": "final", "answer": "已读取 notes.txt"}},
            {
                "thought": "先列目录确认真实文件名",
                "action": {"type": "tool", "tool": "list_dir", "arguments": {"path": "."}},
            },
            {
                "thought": "目录里确有 notes.txt，现在读取它",
                "action": {"type": "tool", "tool": "read_file", "arguments": {"path": "notes.txt"}},
            },
        ],
        "critic_step": [
            {"ok": False, "missing": ["没有成功读取的内容"], "reason": "观测显示 read_file 失败，无文件内容证据"},
            {"ok": False, "missing": ["没有成功读取的内容"], "reason": "第二次仍然是同一路径失败"},
            {"ok": True, "missing": [], "reason": "list_dir 与 read_file 均成功，观测含三条要点"},
        ],
        "reflector": [
            {
                "failure_type": "tool_error",
                "diagnosis": "文件名拼写错误：目标是 note.txt，实际文件是 notes.txt；连续两次重试前没有先确认文件名。",
                "lesson": "读取文件失败时，先用 list_dir 确认实际文件名，再按确认结果重试，不要重复同一路径。",
                "plan_patch": {
                    "remove": ["s1"],
                    "add": [
                        {
                            "id": "s2",
                            "description": "先用 list_dir 列出工作区文件，再用正确文件名读取并总结",
                            "success_criteria": "列出文件、读取到内容并给出要点",
                            "depends_on": [],
                        }
                    ],
                },
            }
        ],
        "critic_final": [
            {
                "complete": True,
                "confidence": 0.88,
                "answer": "notes.txt 要点：1) 运行测试 2) 检查文档 3) 打 tag",
                "missing": [],
                "reason": "重规划后成功读取文件并总结",
            }
        ],
    }
    return TaskSpec(
        id="self-heal",
        goal=goal,
        initial_files={"notes.txt": "发布检查单：1. 运行测试 2. 检查文档 3. 打 tag\n"},
        inner_steps=2,
        mock_script=script,
        metadata={
            "expected_status": "success",
            "title": "失败恢复与重规划",
            "shows": "两次重试、反思教训、计划补丁",
            "expect_replans_min": 1,
        },
    )


def _safety() -> TaskSpec:
    goal = "尝试读取工作区外的文件 ../secret.txt；如果被拒绝，请说明是安全策略生效。"
    script = {
        "planner": [
            {
                "goal": goal,
                "steps": [
                    {
                        "id": "s1",
                        "description": "尝试用 read_file 读取 ../secret.txt，并报告结果",
                        "success_criteria": "说明读取是否成功，若被拒绝则指出是安全策略",
                        "depends_on": [],
                    }
                ],
                "notes": "安全策略演示",
            }
        ],
        "actor": [
            {
                "thought": "尝试读取工作区外的路径",
                "action": {"type": "tool", "tool": "read_file", "arguments": {"path": "../secret.txt"}},
            },
            {
                "thought": "工具返回了安全拒绝",
                "action": {
                    "type": "final",
                    "answer": "读取被拒绝：目标路径越出工作区，运行时安全策略（PathGuard）拦截了该调用。",
                },
            },
        ],
        "critic_step": [
            {"ok": True, "missing": [], "reason": "观测为安全拒绝，答案正确说明了拒绝原因"}
        ],
        "critic_final": [
            {
                "complete": True,
                "confidence": 0.9,
                "answer": "读取被拒绝：目标路径越出工作区，运行时安全策略（PathGuard）拦截了该调用。",
                "missing": [],
                "reason": "目标要求的尝试与说明均已完成",
            }
        ],
    }
    return TaskSpec(
        id="safety",
        goal=goal,
        mock_script=script,
        metadata={
            "expected_status": "success",
            "title": "安全强制与审计",
            "shows": "路径围栏拒绝、安全事件写入 trace",
            "expect_safety_min": 1,
        },
    )
