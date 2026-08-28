"""运行配置：环境变量与 CLI 覆盖，加载期校验。

配置是不可变 dataclass。路径字段保存的是"相对当前工作目录"的原始值，由
``Config.resolve`` 在创建 run 前统一转成绝对路径，避免进程中途切换目录导致
trace 与工作区错位。

环境变量优先级：CLI 显式参数 > ``AGI_LAB_*`` > 提供方专用变量 > 默认值。
token 只在构造 provider 时读取，任何配置打印都必须经过 ``redacted``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

PROVIDERS = ("mock", "deepseek")
APPROVAL_MODES = ("deny", "allow", "prompt")


@dataclass(frozen=True)
class Config:
    """一次 run 的全部可调参数。"""

    provider: str = "mock"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 4096
    request_timeout: float = 120.0

    max_steps: int = 12
    inner_steps: int = 6
    max_attempts: int = 2
    max_replans: int = 2
    max_llm_calls: int = 80
    max_tokens: int = 240_000
    max_seconds: float = 300.0

    allow_shell: bool = False
    allow_network: bool = False
    allow_python_exec: bool = True
    approval_mode: str = "deny"

    persist_memory: bool = True
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    state_dir: Path = field(default_factory=lambda: Path(".agi-lab-state"))
    seed: int = 0
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ConfigError(f"provider 必须是 {PROVIDERS} 之一，收到 {self.provider!r}")
        if self.approval_mode not in APPROVAL_MODES:
            raise ConfigError(f"approval_mode 必须是 {APPROVAL_MODES} 之一，收到 {self.approval_mode!r}")
        if self.max_steps < 1 or self.inner_steps < 1:
            raise ConfigError("max_steps 与 inner_steps 必须 >= 1")
        if self.max_attempts < 1 or self.max_replans < 0:
            raise ConfigError("max_attempts 必须 >= 1，max_replans 必须 >= 0")
        if self.temperature < 0:
            raise ConfigError("temperature 必须 >= 0")
        if not 0 < self.request_timeout <= 3600:
            raise ConfigError("request_timeout 必须在 (0, 3600] 秒内")
        if self.max_seconds <= 0:
            raise ConfigError("max_seconds 必须 > 0")
        if self.provider == "deepseek" and not self.api_key:
            raise ConfigError("provider=deepseek 需要 DEEPSEEK_API_KEY（或 --api-key）；离线运行请用 --provider mock")

    @property
    def memory_dir(self) -> Path:
        """跨任务记忆目录，位于 ``state_dir`` 下。"""
        return self.state_dir / "memory"

    def resolve(self, cwd: Path | None = None) -> "Config":
        """把路径字段转成基于 ``cwd``（默认当前目录）的绝对路径。"""
        base = (cwd or Path.cwd()).resolve()
        return replace(
            self,
            runs_dir=_absolute(self.runs_dir, base),
            state_dir=_absolute(self.state_dir, base),
        )

    def redacted(self) -> dict[str, Any]:
        """返回可安全打印/落盘的配置快照（密钥只保留是否存在的布尔值）。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else None,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "inner_steps": self.inner_steps,
            "max_attempts": self.max_attempts,
            "max_replans": self.max_replans,
            "max_llm_calls": self.max_llm_calls,
            "max_tokens": self.max_tokens,
            "max_seconds": self.max_seconds,
            "allow_shell": self.allow_shell,
            "allow_network": self.allow_network,
            "allow_python_exec": self.allow_python_exec,
            "approval_mode": self.approval_mode,
            "persist_memory": self.persist_memory,
            "runs_dir": str(self.runs_dir),
            "state_dir": str(self.state_dir),
            "seed": self.seed,
        }

    @classmethod
    def from_env(cls, overrides: Mapping[str, Any] | None = None) -> "Config":
        """从环境变量构造配置，再用 ``overrides`` 覆盖显式参数。

        参数：
            overrides: 键名与 dataclass 字段一致；值为 ``None`` 的键被忽略。

        返回：
            校验通过的 :class:`Config`。

        异常：
            ConfigError: 环境变量取值非法或最终配置自相矛盾。
        """
        values: dict[str, Any] = {
            "provider": _env_str("AGI_LAB_PROVIDER", "mock"),
            "model": _env_str("DEEPSEEK_MODEL", _env_str("AGI_LAB_MODEL", "deepseek-chat")),
            "base_url": _env_str("DEEPSEEK_BASE_URL", _env_str("AGI_LAB_BASE_URL", "https://api.deepseek.com")),
            "api_key": os.environ.get("DEEPSEEK_API_KEY") or None,
            "temperature": _env_float("AGI_LAB_TEMPERATURE", 0.2),
            "max_output_tokens": _env_int("AGI_LAB_MAX_OUTPUT_TOKENS", 4096),
            "request_timeout": _env_float("AGI_LAB_REQUEST_TIMEOUT", 120.0),
            "max_steps": _env_int("AGI_LAB_MAX_STEPS", 12),
            "inner_steps": _env_int("AGI_LAB_INNER_STEPS", 6),
            "max_attempts": _env_int("AGI_LAB_MAX_ATTEMPTS", 2),
            "max_replans": _env_int("AGI_LAB_MAX_REPLANS", 2),
            "max_llm_calls": _env_int("AGI_LAB_MAX_LLM_CALLS", 80),
            "max_tokens": _env_int("AGI_LAB_MAX_TOKENS", 240_000),
            "max_seconds": _env_float("AGI_LAB_MAX_SECONDS", 300.0),
            "allow_shell": _env_bool("AGI_LAB_ALLOW_SHELL", False),
            "allow_network": _env_bool("AGI_LAB_ALLOW_NETWORK", False),
            "allow_python_exec": _env_bool("AGI_LAB_ALLOW_PYTHON_EXEC", True),
            "approval_mode": _env_str("AGI_LAB_APPROVAL_MODE", "deny"),
            "persist_memory": _env_bool("AGI_LAB_PERSIST_MEMORY", True),
            "runs_dir": Path(_env_str("AGI_LAB_RUNS_DIR", "runs")),
            "state_dir": Path(_env_str("AGI_LAB_STATE_DIR", ".agi-lab-state")),
            "seed": _env_int("AGI_LAB_SEED", 0),
            "verbose": _env_bool("AGI_LAB_VERBOSE", False),
        }
        for key, value in (overrides or {}).items():
            if value is not None:
                values[key] = value
        if isinstance(values["runs_dir"], str):
            values["runs_dir"] = Path(values["runs_dir"])
        if isinstance(values["state_dir"], str):
            values["state_dir"] = Path(values["state_dir"])
        return cls(**values)


def _absolute(path: Path, base: Path) -> Path:
    candidate = path if path.is_absolute() else base / path
    return candidate.resolve()


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是整数，收到 {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是数字，收到 {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, "")
    if raw == "":
        return default
    lowered = raw.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"环境变量 {name} 必须是布尔值，收到 {raw!r}")
