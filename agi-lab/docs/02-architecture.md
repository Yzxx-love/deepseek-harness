# 架构

本文是 `agi-lab` 的架构基准。其他文档引用本文的模块表和冻结接口，不重复陈述。

## 定位

`agi-lab` 不是一个 AGI，也不宣称接近 AGI。它是一个**可运行、可观测、可评测的通用智能体参考实现**，用来承载 AGI 研究中真正可工程化的那部分：任务分解与重规划、跨任务记忆、失败反思与教训复用、工具使用、资源预算、安全约束和自动评测。

明确不做的事：不训练模型、不做强化学习、不实现多模态、不解决对齐问题（只预留 hook）。这些在 [03-roadmap](03-roadmap.md) 中作为后续里程碑，不属于当前实现范围。

## 能力分级

用一套自建的能力阶梯标定实现位置，避免"AGI"一词的模糊性。分级依据是**自主性与泛化半径**，不是跑分高低。

| 级别 | 名称 | 定义 | 本项目状态 |
|---|---|---|---|
| L0 | 工具调用器 | 单轮 ReAct：一次思考、一次工具调用，无持久状态 | 已覆盖 |
| L1 | 计划执行者 | 显式分解目标、按依赖执行、失败后重规划 | 已实现（核心目标） |
| L2 | 记忆学习者 | 情节/语义/反思三层记忆，跨任务复用教训，可度量迁移收益 | 机制已实现，收益需真实模型验证 |
| L3 | 自主研究体 | 自设子目标、长期运行、资源自管理、主动求助 | 仅预留脚手架（预算、ASK 动作、中断点） |
| L4 | 跨域迁移者 | 未见任务族上接近人类可靠性，少样本掌握新技能 | 不在实现范围 |
| L5 | 超人类通用 | 多域显著超越人类专家并可自我改进 | 不在实现范围（定义本身有争议） |

"L2 机制已实现"指运行时确实写入并使用跨任务记忆；"收益需验证"指在当前无 API key 的环境里无法证明真实模型下教训复用能降低步数或提高成功率，这只能由 [04-evaluation](04-evaluation.md) 的真实模型套件给出。

## 分层

```
┌──────────────────────────────────────────────────────┐
│ L5 可观测与评测  tracing / eval / replay / report      │
├──────────────────────────────────────────────────────┤
│ L4 认知  planner → executor(loop) → critic → reflector │
├──────────────────────────────────────────────────────┤
│ L3 记忆  working / episodic / semantic / reflection    │
├──────────────────────────────────────────────────────┤
│ L2 运行时  budget / safety / approval / metrics        │
├──────────────────────────────────────────────────────┤
│ L1 接口  types.py（数据契约 + Protocol，全局冻结）      │
├──────────────────────────────────────────────────────┤
│ L0 环境  工具：calculator / files / shell / python / memory │
└──────────────────────────────────────────────────────┘
      所有模型调用经由 LLMProvider：mock（离线）| deepseek（真实）
```

依赖方向只能向下：评测可以导入全部层，运行时可以导入接口与工具，认知层只依赖接口（`types.py`）不依赖具体实现。`llm/mock.py` 属于接口层，与真实 provider 同签名。

## 核心循环

外层是计划执行，内层是 ReAct。两层分离的原因：分解负责"做什么"，ReAct 负责"怎么做"，把两者混在一个提示里会让重规划失去可靠边界。

```
run(goal):
  lessons ← memory.search(goal, kinds=[REFLECTION, SEMANTIC])      # 跨任务迁移
  plan    ← planner.plan(goal, tools, lessons)                     # 结构化 JSON
  while budget.ok:
      step ← plan.next_ready()
      if step is None: break
      for attempt in 1..max_attempts:
          for turn in 1..inner_steps:
              action ← actor(goal, plan, step, observations, lessons)
              if action is FINAL:  last_answer ← action.answer; break
              if action is ASK:    record question; status ← BLOCKED; return
              if action is REPLAN: break to reflect
              observation ← tools.execute(action.tool_call)        # 不抛异常
              memory.record(observation)
          verdict ← critic.judge_step(step, observations)
          if verdict.ok: plan.mark_done(step); memory.add_episode(); break
      if not ok:
          lesson ← reflector.reflect(failure, plan, step, attempt)  # 结构化诊断
          memory.add_lesson(lesson)
          plan ← replan(plan, lesson.patch, budget)
  verdict ← critic.judge_final(goal, plan, answer)
  return RunResult(status, answer, metrics, trace)
```

终止条件全部显式，不存在"模型不产出就无限转"：步数预算、模型调用预算、token 预算、墙钟预算、内层轮次、单步重试次数、总重规划次数。任一触发即停止，状态为 `BUDGET_EXHAUSTED` 并携带已获得的部分结果与停止原因。

## 模块表

| 路径 | 职责 | 关键接口 | 归属 |
|---|---|---|---|
| `agi_lab/types.py` | 数据契约 + Protocol，全局冻结 | `Plan` `Action` `Observation` `TaskSpec` `RunResult` `LLMProvider` `MemoryStore` `Tool` `parse_json_object` | 内核 |
| `agi_lab/errors.py` | 异常层级 | `AgiLabError` `BudgetExceeded` `SafetyViolation` `ToolInvocationError` `LLMError` `ActionFormatError` | 内核 |
| `agi_lab/jsonutil.py` | 模型输出 JSON 提取与校验 | `extract_json_object` `expect_keys` | 内核 |
| `agi_lab/config.py` | 运行配置（环境变量 + CLI） | `Config` `from_env` | 内核 |
| `agi_lab/llm/base.py` | provider 基类与重试包装 | `BaseLLM` | llm |
| `agi_lab/llm/openai_compat.py` | OpenAI 兼容 HTTP（DeepSeek） | `OpenAICompatibleLLM` | llm |
| `agi_lab/llm/mock.py` | 脚本化离线 provider，用于机制自检与回放 | `MockLLM` | 认知 |
| `agi_lab/memory/*.py` | 四层记忆 + 门面 | `MemoryHub.search/render_context` | 记忆 |
| `agi_lab/tools/*.py` | 工具实现与注册表 | `ToolRegistry.execute` `build_default_tools` | 工具 |
| `agi_lab/cognition/*.py` | 规划 / 反思 / 评审 / 提示词 | `Planner` `Reflector` `Critic` | 认知 |
| `agi_lab/runtime/*.py` | 循环、预算、安全、追踪 | `Agent.run` `BudgetTracker` `PathGuard` `Tracer` | 运行时 |
| `agi_lab/eval/*.py` | 任务集、grader、runner、报告 | `builtin_suites` `grade` `run_suite` | 评测 |
| `agi_lab/cli.py` | 命令行 | `main` | 运行时 |

## 数据契约

所有跨模块传递的数据都是 `types.py` 中的不可变 dataclass 或 Enum，不使用裸 dict 传参。三个理由：类型检查可查、序列化到 trace 时无歧义、队友并行实现时接口不会漂移。

核心对象：

- `TaskSpec`：`id`、`goal`、`initial_files`（写入工作区）、`grader_id`、`mock_script`（仅机制自检使用）、预算覆盖项。
- `Plan` / `PlanStep`：步骤有 `depends_on` 序号依赖与 `success_criteria` 文本；状态机 `PENDING → RUNNING → DONE | FAILED | SKIPPED`；`next_ready()` 只返回依赖已 DONE 的步骤。
- `Action`：`TOOL` / `FINAL` / `REPLAN` / `ASK`，由 `Action.from_json` 严格解析，缺字段或未知类型抛 `ActionFormatError`。
- `Observation`：工具结果、执行摘要、`untrusted` 标记（工具输出一律视为不可信数据，进入提示时带隔离标记）。
- `RunResult`：状态、最终答案、`Metrics`、trace 路径、停止原因。

## 结构化输出协议

模型输出必须是单个 JSON 对象，不允许自由文本包裹。每个角色的提示以机器可读头开始，mock 与真实 provider 都据此分发：

```
[agi-lab:role=planner]
```

角色与 schema：

| 角色 | 必需字段 |
|---|---|
| `planner` | `goal`, `steps[]`（`id`,`description`,`success_criteria`,`depends_on`） |
| `actor` | `thought`, `action`（`type` + 按类型附加 `tool`/`arguments` 或 `answer` 或 `reason`） |
| `critic_step` | `ok`, `missing[]`, `reason` |
| `critic_final` | `complete`, `confidence`, `answer`, `missing[]` |
| `reflector` | `failure_type`, `diagnosis`, `lesson`, `plan_patch`（`remove[]`,`add[]`） |

解析失败时执行一次"带错误信息的修复重试"；仍失败则按 `ActionFormatError` 终止该步并计入失败，由反思处理。这个设计让格式问题成为可观测的失败类别，而不是静默重试。

## 记忆设计

| 层 | 内容 | 检索 | 持久化 |
|---|---|---|---|
| Working | 当前目标、计划、当前步、最近观测 | 直接注入，字符预算截断 | 不落盘 |
| Episodic | 已完成步骤的执行摘要 | 近因 + 词法相关 | `runs/<run>/memory/episodic.jsonl` |
| Semantic | 事实、文件摘要、检索命中 | BM25 词法打分 | `semantic.jsonl` |
| Reflection | 失败诊断与教训，去重 | 词法相关 + 重要度 | `reflection.jsonl` |

选择 BM25 词法检索而非向量检索：零依赖、确定性、无 API key 也能评测。向量与嵌入留作后续里程碑，接口 `MemoryStore.search(MemoryQuery)` 已为此预留。

## 安全模型

安全不是提示词约束，是运行时强制：

1. **路径围栏**：所有文件工具的目标路径解析后必须位于工作区根目录内（含符号链接检查），越界抛 `SafetyViolation` 并记录为安全事件。
2. **工具白名单**：默认只启用只读与计算工具；`shell`、`python_exec`、`http_get` 需显式开关。
3. **不可逆操作审批**：`requires_approval` 工具在非交互模式下默认拒绝（fail closed），交互模式下逐次询问。
4. **不可信数据隔离**：工具输出在提示中用显式数据块包裹并标注来源，模型输出中的指令不提升为系统指令。
5. **资源预算**：所有循环有硬上限，超时即停；子进程有独立超时与 kill。
6. **审计**：每个动作、每次拒绝都写入 trace JSONL，可回放。

明确局限：这是**进程级与路径级沙箱**，不是内核级隔离，也不防御恶意工具实现。生产部署需要容器/VM 边界，见 [05-safety](05-safety-and-alignment.md)。

## 可观测与评测

- `Tracer` 写 `runs/<run_id>/trace.jsonl`（每个事件一行，含时间戳、类型、载荷）与 `report.md`（人读摘要）。
- `replay` 从 trace 重建事件序列，用于调试与回归。
- `eval` 只认外部 grader，不采信 agent 自评：机制套件用离线 mock 脚本验证运行时正确性；能力套件用真实 provider 并跳过缺失 key 的情况。

关键设计判断：**评测必须先于能力**。没有独立 grader 的"AGI demo"只能自证，不能证明。见 [04-evaluation](04-evaluation.md)。

## 与 AGI 能力维度的对应

| AGI 常见维度 | 本架构中的承载 |
|---|---|
| 生成性（面对新任务） | 计划分解 + 内层 ReAct，而非固定流水线 |
| 组合性（多步依赖） | `PlanStep.depends_on` + 失败重规划 |
| 持续学习 | 三层记忆 + 反思教训显式复用 |
| 自主性 | 自设子目标/ASK/L3 脚手架 |
| 可靠性 | 预算、显式失败状态、独立 grader |
| 安全性 | 运行时强制而非提示词约束 |

这六项是可工程化的部分；"理解""意识"等争议概念不在本项目的可测量范围内，文档不做承诺。
