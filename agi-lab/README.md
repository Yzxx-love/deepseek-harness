# agi-lab

**一个可运行、可观测、可评测的通用智能体参考实现。**

它把 AGI 研究中真正可工程化的那部分做成最小系统：目标分解与失败重规划、四层跨任务记忆、反思教训复用、工具使用与沙箱、资源预算、结构化模型协议、以及不采信自评的独立评测台。

> **这不是 AGI。** 本项目不训练模型、不做强化学习、不解决对齐问题、不支持多模态。它在自建能力阶梯上实现 **L1（计划执行者）** 并落地 **L2（记忆学习者）机制**，L3 以上不在实现范围。完整定位与"我们不声称什么"见 [docs/01-vision-and-scope.md](docs/01-vision-and-scope.md)。

## 快速开始（离线，无需 API key）

```sh
cd agi-lab

python -m agi_lab demo                      # 4 个离线机制演示，全部确定性
python -m agi_lab run --demo-task self-heal # 单跑"失败→反思→重规划→成功"
python -m agi_lab eval --suite mechanism    # 机制评测套件
python -m agi_lab tools                     # 查看工具与安全开关
```

离线演示使用脚本化 mock provider：它回放预先写好的动作序列，**只证明运行时机制正确，不衡量智能体能力**。

## 使用真实模型

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."   # PowerShell；bash 用 export
python -m agi_lab run "读取 sales.csv 汇总销售额并写入 report.md" --provider deepseek --workspace .\work
```

真实任务走 OpenAI 兼容的 `/chat/completions`，默认 `https://api.deepseek.com`、模型 `deepseek-chat`。可用 `DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 或 `--model` 覆盖。

## 命令

| 命令 | 作用 | 退出码 |
|---|---|---|
| `run GOAL` | 执行任意任务（需真实 provider） | 0 成功 / 1 未达标 / 2 参数或配置错误 |
| `run --demo-task ID` | 跑离线演示任务 | 同上 |
| `demo` | 全部离线机制演示 | 0 全通过 |
| `eval --suite mechanism\|capability\|all` | 跑评测套件 | 0 无失败 |
| `tools` | 列出工具规格与安全开关 | 0 |
| `replay TRACE` | 渲染 `trace.jsonl` 时间线 | 0 |
| `config` | 打印生效配置（密钥脱敏） | 0 |

常用开关：`--provider` `--model` `--max-steps` `--max-seconds` `--allow-shell` `--allow-network` `--no-python-exec` `--approval deny|allow|prompt` `--no-memory` `--verbose` `--json`。

## 架构一览

```
计划-执行-反思闭环
  planner → 计划（步骤 + 依赖 + 验收标准）
  actor   → 每轮一个动作：工具 / 终答 / 重规划 / 提问
  critic  → 子目标评审 + 目标终审（只看观测证据）
  reflector → 失败诊断 + 可迁移教训 + 计划补丁
四层记忆
  working（当前上下文） / episodic（执行摘要） / semantic（事实，BM25） / reflection（教训，去重）
运行时强制
  PathGuard 路径围栏 · 工具白名单 · 不可逆操作审批（fail closed） · 硬预算 · 全量 trace
```

细节与冻结接口见 [docs/02-architecture.md](docs/02-architecture.md)。

## 目录

```
agi-lab/
├── agi_lab/
│   ├── types.py errors.py jsonutil.py config.py   # 冻结内核：数据契约与异常
│   ├── llm/        # provider 抽象、OpenAI 兼容 HTTP、离线 mock
│   ├── memory/     # 四层记忆 + MemoryHub 门面 + JSONL 持久化
│   ├── tools/      # PathGuard、8 个工具、注册表（审批 + 参数校验）
│   ├── cognition/  # 规划、评审、反思、提示词、结构化调用通道
│   ├── runtime/    # Agent 双层循环、预算、安全策略、trace 与报告
│   ├── eval/       # 评测套件、独立 grader、报告
│   ├── demo.py     # 4 个离线机制演示
│   └── cli.py      # 命令行
├── docs/           # 愿景/架构/路线图/评测/安全/ADR
└── tests/          # 纯标准库 unittest，无第三方依赖
```

## 能力阶梯（自建标定）

| 级别 | 名称 | 状态 |
|---|---|---|
| L0 | 工具调用器（单轮 ReAct） | 已覆盖 |
| L1 | 计划执行者（分解 + 依赖 + 重规划） | 已实现（核心目标） |
| L2 | 记忆学习者（三层记忆 + 教训复用） | 机制已实现，收益需真实模型验证 |
| L3 | 自主研究体（自设子目标、长期运行） | 仅脚手架：预算、ASK 动作、中断点 |
| L4/L5 | 跨域迁移者 / 超人类通用 | 不在实现范围 |

## 评测立场

- 评测**只认外部 grader**：检查答案文本、工作区文件、指标与 trace 事件；不采信 agent 自己的 `status` 或置信度。
- **机制套件**（离线 mock）验证运行时正确性；**能力套件**（真实 provider）验证能力，无 key 时整组跳过并在报告注明。
- mock 的成功率只能说明"循环没坏"，不能说明"智能体变强"。

指标定义、消融与反自欺清单见 [docs/04-evaluation.md](docs/04-evaluation.md)。

## 安全模型

运行时强制而非提示词约束：路径围栏（含 `..`、绝对路径越界、符号链接）、工具白名单（`shell`/`network`/`python_exec` 默认关闭或受控）、不可逆操作审批（非交互环境 fail closed）、工具输出标记为不可信数据、每次拒绝写入 trace。

明确局限：这是**进程级 + 路径级**沙箱，不是内核级隔离，也不防御恶意工具实现；生产部署需要容器/VM 边界。威胁模型见 [docs/05-safety-and-alignment.md](docs/05-safety-and-alignment.md)。

## 测试

纯标准库，无 pytest、无网络、无真实等待：

```sh
cd agi-lab
python -m unittest discover -s tests -t .
```

分层覆盖：数据契约与 JSON 提取、认知层解析与修复重试、记忆检索与持久化、工具沙箱与审批、预算与 trace、CLI 退出码、离线端到端。

## 已知局限

- 记忆检索是纯词法（BM25 + 中文字/bigram），无词面重叠的同义迁移召回为 0；向量检索留待后续里程碑。
- 三层记忆分数各自归一化，跨层排序是工程折中，不承诺量纲可比。
- 反思去重为 O(n²)，无 TTL/遗忘，单进程假设。
- 真实 provider 的重试/退避与错误映射有单元测试（本地 HTTP 桩），但未在真实网络上回归。
- 计划质量、评审可靠性、教训迁移收益都依赖底层模型能力，本项目提供的是承载它们的运行时。

## 文档

| 文档 | 内容 |
|---|---|
| [01 愿景与范围](docs/01-vision-and-scope.md) | 定位、操作定义、不声称什么 |
| [02 架构](docs/02-architecture.md) | 分层、循环、模块表、冻结契约、安全模型 |
| [03 路线图](docs/03-roadmap.md) | M0–M5 里程碑、可测量退出条件、风险 |
| [04 评测](docs/04-evaluation.md) | 指标公式、套件设计、反自欺检查单 |
| [05 安全与对齐](docs/05-safety-and-alignment.md) | 威胁模型、强制点、部署局限 |
| [06 设计决策](docs/06-design-decisions.md) | ADR：为什么这样做，放弃了什么 |

## License

MIT
