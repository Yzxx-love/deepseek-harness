"""认知层：规划、执行提示、评审、反思。

对外只暴露四个构造器与补丁函数：

- :class:`Planner`：目标 → 计划
- :class:`Reflector`：失败 → 诊断 + 教训 + 计划补丁
- :class:`Critic`：证据 → 子目标判定 / 目标终审
- :class:`StructuredClient`：统一的"补全 + JSON 校验 + 一次修复"通道
- :func:`apply_patch`：计划补丁应用
"""

from .client import StructuredClient
from .models import FinalVerdict, Reflection, StepVerdict
from .planner import Critic, Planner, Reflector, apply_patch

__all__ = [
    "Critic",
    "FinalVerdict",
    "Planner",
    "Reflection",
    "Reflector",
    "StepVerdict",
    "StructuredClient",
    "apply_patch",
]
