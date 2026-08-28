"""运行追踪与报告。

每个 run 产生两份产物，全部落在 ``<run_dir>`` 下：

- ``trace.jsonl``：每个事件一行 JSON，机器可读，用于回放与回归分析；
- ``report.md``：人读摘要，含状态、指标、最终答案、安全事件与时间线。

设计约束：追踪层不得因序列化失败中断运行（``to_jsonable`` 有兜底），也不得记录
密钥（调用方传入 ``Config.redacted()``，工具输出经 ``PolicyGuard.redact``）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..jsonutil import to_jsonable
from ..types import RunResult

_MAX_SUMMARY_CHARS = 160


def _summarize(payload: dict[str, Any]) -> str:
    for key in ("step", "action", "observation", "answer", "message", "reason", "status", "ok"):
        if key in payload:
            value = payload[key]
            text = json.dumps(to_jsonable(value), ensure_ascii=False)
            return text[:_MAX_SUMMARY_CHARS]
    text = json.dumps(payload, ensure_ascii=False) if payload else ""
    return text[:_MAX_SUMMARY_CHARS]


class Tracer:
    """把事件追加写入 ``trace.jsonl``，可选同时回显到标准输出。

    参数：
        run_dir: run 的产物目录，不存在时自动创建。
        run_id: 本次 run 的标识，写入每条事件便于跨文件关联。
        echo: 是否把事件回显到标准输出（``--verbose``）。
    """

    def __init__(self, run_dir: Path, *, run_id: str, echo: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.echo = echo
        self.path = self.run_dir / "trace.jsonl"
        self._handle = self.path.open("a", encoding="utf-8")

    def event(self, kind: str, **payload: Any) -> None:
        """追加一条事件。

        参数：
            kind: 事件类型，如 ``run_start`` / ``action`` / ``reflection``。
            **payload: 事件载荷，任意可被 :func:`to_jsonable` 转换的值。
        """
        record = {
            "ts": round(time.time(), 6),
            "run_id": self.run_id,
            "kind": kind,
            "payload": to_jsonable(payload),
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()
        if self.echo:
            print(f"[trace] {kind}: {_summarize({k: v for k, v in payload.items()})}")

    def events(self) -> list[dict[str, Any]]:
        """读回当前 trace 的全部事件。"""
        return read_events(self.path)

    def close(self) -> None:
        """关闭文件句柄；重复调用安全。"""
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_events(trace_path: Path) -> list[dict[str, Any]]:
    """读取 trace JSONL，坏行跳过。

    参数：
        trace_path: trace 文件路径。

    返回：
        事件字典列表，按写入顺序。
    """
    path = Path(trace_path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                events.append(record)
    return events


def render_timeline(trace_path: Path, *, max_events: int = 80) -> str:
    """把 trace 渲染成人读时间线，用于 ``agi-lab replay``。

    参数：
        trace_path: trace 文件路径。
        max_events: 最多渲染的事件数，超出只显示末尾。

    返回：
        多行文本；文件不存在时返回说明文字。
    """
    events = read_events(trace_path)
    if not events:
        return f"（没有事件：{trace_path}）"
    selected = events[-max_events:]
    lines = [f"trace: {trace_path}（共 {len(events)} 个事件，显示末尾 {len(selected)} 个）", ""]
    for record in selected:
        payload = record.get("payload") or {}
        summary = _summarize(payload if isinstance(payload, dict) else {})
        lines.append(f"- {record.get('kind', '?'):<14} {summary}")
    return "\n".join(lines)


def write_report(run_dir: Path, result: RunResult, events: Sequence[dict[str, Any]] | None = None) -> Path:
    """写出 ``report.md`` 并返回路径。

    参数：
        run_dir: run 的产物目录。
        result: 运行结果。
        events: 已读取的事件列表；``None`` 时从 ``trace.jsonl`` 读取。

    返回：
        报告文件路径。
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if events is None:
        events = read_events(run_dir / "trace.jsonl")
    report_path = run_dir / "report.md"
    report_path.write_text(render_report(result, events), encoding="utf-8")
    return report_path


def render_report(result: RunResult, events: Iterable[dict[str, Any]]) -> str:
    """渲染报告 Markdown。

    参数：
        result: 运行结果。
        events: 事件序列。

    返回：
        Markdown 文本。
    """
    event_list = list(events)
    lines: list[str] = [
        f"# run 报告：{result.run_id}",
        "",
        f"- 任务：`{result.task_id}`",
        f"- 状态：**{result.status.value}**",
        f"- 是否通过独立评测：{'是' if result.success else '否（或未评测）'}",
        f"- 停止原因：{result.stop_reason or '—'}",
        f"- 错误：{result.error or '—'}",
        f"- trace：`{result.trace_path or '—'}`",
        "",
        "## 指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for key, value in result.metrics.as_dict().items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## 最终答案", "", result.answer.strip() or "（无）", ""]
    if result.plan is not None:
        lines += ["## 计划终态", "", "```", result.plan.render(max_chars=4000), "```", ""]
    safety_events = [record for record in event_list if "safety" in str(record.get("kind", ""))]
    lines += ["## 安全事件", ""]
    if safety_events:
        for record in safety_events:
            payload = record.get("payload") or {}
            lines.append(f"- {_summarize(payload if isinstance(payload, dict) else {})}")
    else:
        lines.append("（无）")
    lines += ["", "## 时间线", ""]
    for record in event_list:
        payload = record.get("payload") or {}
        summary = _summarize(payload if isinstance(payload, dict) else {})
        lines.append(f"- `{record.get('kind', '?')}` {summary}")
    lines.append("")
    return "\n".join(lines)
