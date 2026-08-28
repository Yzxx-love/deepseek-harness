"""agi-lab：通用智能体参考实现。

这不是 AGI。它把 AGI 研究中可工程化的部分做成一个可运行、可观测、可评测的最小系统：
计划-执行-反思循环、四层记忆、工具沙箱、资源预算、结构化模型协议与独立评测台。

入口：

- ``python -m agi_lab run "任务"``：跑一个任务
- ``python -m agi_lab eval``：跑评测套件
- ``python -m agi_lab demo``：离线演示
- ``python -m agi_lab tools``：列出可用工具

设计与路线图见 ``docs/``，架构基准见 ``docs/02-architecture.md``。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
