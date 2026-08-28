"""命令行入口：``run`` / ``demo`` / ``eval`` / ``tools`` / ``replay`` / ``config``。

命令与退出码：

- ``0``：成功（``run`` 的 status 为 success；``demo`` 全部符合预期；``eval`` 无失败）
- ``1``：运行完成但未达标（failed / budget_exhausted / blocked / error）
- ``2``：参数或配置错误、缺少 API key、模块不可用
- ``130``：用户中断

``run`` 与 ``eval`` 使用真实 provider 时需要 ``DEEPSEEK_API_KEY``；``demo`` 与
``tools`` 强制使用离线 mock，不需要密钥也不联网。
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import Config
from .demo import demo_tasks, find_demo_task
from .errors import AgiLabError, ConfigError
from .llm import build_llm
from .llm.mock import MockLLM
from .memory.hub import MemoryHub
from .runtime.agent import Agent
from .runtime.safety import PolicyGuard
from .runtime.tracing import render_timeline
from .tools import build_default_tools
from .types import RunResult, RunStatus, TaskSpec

_MAX_REPLAY_EVENTS = 80


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="agi-lab",
        description="通用智能体参考实现：计划-执行-反思闭环、四层记忆、工具沙箱、独立评测。",
        epilog="注意：本项目的 mock provider 只回放预写脚本，用于离线机制演示，不代表模型能力。",
    )
    parser.add_argument("--version", action="version", version=f"agi-lab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="执行一个任务（真实 provider 需要 API key）")
    run_parser.add_argument("goal", nargs="?", default="", help="任务目标（自然语言）")
    run_parser.add_argument("--demo-task", dest="demo_task", default=None, help="改为运行离线演示任务（指定 id）")
    run_parser.add_argument("--task-id", dest="task_id", default=None, help="自定义任务 id")
    run_parser.add_argument("--workspace", default=None, help="显式指定工作区（默认 <run_dir>/workspace）")
    _add_runtime_flags(run_parser, provider=True)

    demo_parser = sub.add_parser("demo", help="运行离线机制演示（mock，无需 API key）")
    demo_parser.add_argument("--runs-dir", default=None)
    demo_parser.add_argument("--state-dir", default=None)
    demo_parser.add_argument("--json", action="store_true")
    demo_parser.add_argument("--verbose", action="store_true")
    demo_parser.add_argument("--approval", choices=["deny", "allow", "prompt"], default=None)

    eval_parser = sub.add_parser("eval", help="运行评测套件（机制套件可离线，能力套件需要真实 provider）")
    eval_parser.add_argument("--suite", choices=["mechanism", "capability", "all"], default="mechanism")
    eval_parser.add_argument("--out", default=None, help="报告输出目录")
    _add_runtime_flags(eval_parser, provider=True)

    tools_parser = sub.add_parser("tools", help="列出可用工具与安全开关")
    tools_parser.add_argument("--allow-shell", dest="allow_shell", action="store_true")
    tools_parser.add_argument("--allow-network", dest="allow_network", action="store_true")
    tools_parser.add_argument("--no-python-exec", dest="no_python_exec", action="store_true")
    tools_parser.add_argument("--approval", choices=["deny", "allow", "prompt"], default=None)
    tools_parser.add_argument("--json", action="store_true")

    replay_parser = sub.add_parser("replay", help="渲染 trace 时间线")
    replay_parser.add_argument("trace", help="trace.jsonl 路径")
    replay_parser.add_argument("--max-events", dest="max_events", type=int, default=_MAX_REPLAY_EVENTS)

    config_parser = sub.add_parser("config", help="打印生效配置（密钥脱敏）")
    config_parser.add_argument("--provider", choices=["mock", "deepseek"], default=None)

    return parser


def _add_runtime_flags(parser: argparse.ArgumentParser, *, provider: bool) -> None:
    if provider:
        parser.add_argument("--provider", choices=["mock", "deepseek"], default=None)
        parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--max-seconds", dest="max_seconds", type=float, default=None)
    parser.add_argument("--allow-shell", dest="allow_shell", action="store_true")
    parser.add_argument("--allow-network", dest="allow_network", action="store_true")
    parser.add_argument("--no-python-exec", dest="no_python_exec", action="store_true")
    parser.add_argument("--approval", choices=["deny", "allow", "prompt"], default=None)
    parser.add_argument("--no-memory", dest="no_memory", action="store_true")
    parser.add_argument("--runs-dir", dest="runs_dir", default=None)
    parser.add_argument("--state-dir", dest="state_dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    """命令行主入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "demo": _cmd_demo,
        "eval": _cmd_eval,
        "tools": _cmd_tools,
        "replay": _cmd_replay,
        "config": _cmd_config,
    }
    try:
        return handlers[args.command](args)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except AgiLabError as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 边界：未知异常给出可读信息与可选 traceback
        print(f"未预期错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    run_id = _new_run_id("run")
    run_dir = config.runs_dir / run_id
    task, llm = _select_task_and_llm(args, config)
    if task is None or llm is None:
        return 2
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else run_dir / "workspace"
    memory = _build_memory(config)
    tools = _build_tools(config, workspace, memory)
    agent = Agent(
        config=config,
        llm=llm,
        tools=tools,
        memory=memory,
        workspace=workspace,
        run_dir=run_dir,
        run_id=run_id,
    )
    result = agent.run(task)
    return _emit_result(result, args)


def _cmd_demo(args: argparse.Namespace) -> int:
    config = _resolve_config(args, force_provider="mock")
    run_root = config.runs_dir / _new_run_id("demo")
    memory = MemoryHub(root=run_root / "memory", persist=True)
    rows: list[tuple[TaskSpec, RunResult, bool]] = []
    failures = 0
    for task in demo_tasks():
        workspace = run_root / task.id / "workspace"
        run_dir = run_root / task.id
        tools = _build_tools(config, workspace, memory)
        agent = Agent(
            config=config,
            llm=MockLLM(task.mock_script),
            tools=tools,
            memory=memory,
            workspace=workspace,
            run_dir=run_dir,
            run_id=f"{run_root.name}-{task.id}",
        )
        result = agent.run(task)
        expected = str(task.metadata.get("expected_status", "success"))
        passed = result.status.value == expected
        if not passed:
            failures += 1
        rows.append((task, result, passed))

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "task": task.id,
                        "title": task.metadata.get("title", ""),
                        "status": result.status.value,
                        "expected": str(task.metadata.get("expected_status", "success")),
                        "passed": passed,
                        "metrics": result.metrics.as_dict(),
                        "report": result.report_path,
                    }
                    for task, result, passed in rows
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("agi-lab 离线机制演示（mock：回放预写脚本，不衡量能力）\n")
        print(f"{'结果':<6}{'任务':<16}{'说明':<22}{'状态':<10}{'步数':<6}{'调用':<6}{'重规划':<8}{'教训':<6}{'安全':<6}")
        print("-" * 92)
        for task, result, passed in rows:
            print(
                f"{'PASS' if passed else 'FAIL':<6}"
                f"{task.id:<16}"
                f"{str(task.metadata.get('title', '')):<22}"
                f"{result.status.value:<10}"
                f"{result.metrics.steps:<6}"
                f"{result.metrics.llm_calls:<6}"
                f"{result.metrics.replans:<8}"
                f"{result.metrics.lessons:<6}"
                f"{result.metrics.safety_blocks:<6}"
            )
        print("-" * 92)
        print(f"共 {len(rows)} 个演示，{len(rows) - failures} 通过，{failures} 失败")
        print(f"产物目录：{run_root}")
        print("每个子目录含 trace.jsonl 与 report.md，可用 `agi-lab replay <trace>` 查看时间线。")
    return 0 if failures == 0 else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    try:
        from .eval import run_evaluation
    except ImportError as exc:  # pragma: no cover - 仅在评测模块缺失时触发
        print(f"评测模块不可用：{exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser().resolve() if args.out else config.runs_dir / "eval"
    report = run_evaluation(config=config, suite=args.suite, out_dir=out_dir)
    print(report.markdown)
    if args.json:
        print(
            json.dumps(
                {
                    "suite": report.suite,
                    "passed": report.passed,
                    "failed": report.failed,
                    "skipped": report.skipped,
                },
                ensure_ascii=False,
            )
        )
    return 0 if report.failed == 0 else 1


def _cmd_tools(args: argparse.Namespace) -> int:
    config = _resolve_config(args, force_provider="mock")
    memory = MemoryHub(root=None, persist=False)
    registry = _build_tools(config, Path.cwd(), memory)
    specs = registry.specs()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                        "side_effects": spec.side_effects,
                        "requires_approval": spec.requires_approval,
                    }
                    for spec in specs
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"已注册 {len(specs)} 个工具（工作区：{Path.cwd()}）\n")
    print(registry.describe(max_chars=8000))
    print(
        "\n安全开关："
        f"shell={'on' if config.allow_shell else 'off'} "
        f"network={'on' if config.allow_network else 'off'} "
        f"python_exec={'on' if config.allow_python_exec else 'off'} "
        f"approval={config.approval_mode}"
    )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace).expanduser()
    if not trace_path.exists():
        print(f"trace 不存在：{trace_path}", file=sys.stderr)
        return 2
    print(render_timeline(trace_path, max_events=args.max_events))
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    overrides = {"provider": args.provider} if args.provider else {}
    config = Config.from_env(overrides).resolve()
    print(json.dumps(config.redacted(), ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _resolve_config(args: argparse.Namespace, *, force_provider: str | None = None) -> Config:
    overrides: dict[str, Any] = {}
    for key in ("provider", "model", "max_steps", "max_seconds", "approval_mode"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if force_provider is not None:
        overrides["provider"] = force_provider
    if getattr(args, "allow_shell", False):
        overrides["allow_shell"] = True
    if getattr(args, "allow_network", False):
        overrides["allow_network"] = True
    if getattr(args, "no_python_exec", False):
        overrides["allow_python_exec"] = False
    if getattr(args, "no_memory", False):
        overrides["persist_memory"] = False
    if getattr(args, "runs_dir", None):
        overrides["runs_dir"] = Path(args.runs_dir)
    if getattr(args, "state_dir", None):
        overrides["state_dir"] = Path(args.state_dir)
    if getattr(args, "verbose", False):
        overrides["verbose"] = True
    return Config.from_env(overrides).resolve()


def _select_task_and_llm(
    args: argparse.Namespace,
    config: Config,
) -> tuple[TaskSpec | None, Any | None]:
    if args.demo_task:
        task = find_demo_task(args.demo_task)
        if task is None:
            available = ", ".join(item.id for item in demo_tasks())
            print(f"未知演示任务：{args.demo_task}（可用：{available}）", file=sys.stderr)
            return None, None
        return task, MockLLM(task.mock_script)
    goal = (args.goal or "").strip()
    if not goal:
        print("run 需要任务目标文本，或用 --demo-task 运行离线演示。", file=sys.stderr)
        return None, None
    if config.provider == "mock":
        print(
            "provider=mock 无法执行任意目标：它只回放预写脚本。"
            "离线体验请运行 `agi-lab demo`；真实任务请设置 DEEPSEEK_API_KEY 并使用 --provider deepseek。",
            file=sys.stderr,
        )
        return None, None
    return TaskSpec(id=args.task_id or _slug(goal), goal=goal), build_llm(config)


def _build_memory(config: Config) -> MemoryHub:
    root = config.memory_dir if config.persist_memory else None
    return MemoryHub(root=root, persist=config.persist_memory)


def _build_tools(config: Config, workspace: Path, memory: MemoryHub) -> Any:
    guard = PolicyGuard(config)
    return build_default_tools(
        workspace=workspace,
        allow_shell=config.allow_shell,
        allow_network=config.allow_network,
        allow_python_exec=config.allow_python_exec,
        approval=guard.approval(),
        approval_mode=config.approval_mode,
        memory=memory,
    )


def _emit_result(result: RunResult, args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(_result_payload(result), ensure_ascii=False, indent=2))
    else:
        print(result.summary())
        if result.answer:
            print(f"\n答案：{result.answer}")
        if result.stop_reason:
            print(f"停止原因：{result.stop_reason}")
        if result.error:
            print(f"错误：{result.error}")
        print(f"trace：{result.trace_path}")
        print(f"报告：{result.report_path}")
    return 0 if result.status is RunStatus.SUCCESS else 1


def _result_payload(result: RunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "status": result.status.value,
        "success": result.success,
        "answer": result.answer,
        "metrics": result.metrics.as_dict(),
        "stop_reason": result.stop_reason,
        "error": result.error,
        "trace": result.trace_path,
        "report": result.report_path,
        "plan": result.plan.to_json() if result.plan is not None else None,
    }


def _new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _slug(text: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in text.strip())
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return (collapsed or "task")[:32]
