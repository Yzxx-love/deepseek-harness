# Agent Note: 根据提供的 UI 参考迁移 DeepSeek 视觉场

Status: implemented


[English](2026-08-29-deepseek-background-visual-field.md) | 中文
## Problem

提供的 dsh-client-ui-deepseek-bg 包含仅深色启用的极光、星座网格、粒子鲸鱼和磨砂表面，但选择器依赖某一次 DSH 构建生成的类名，不能作为稳定的源码级接入点。本地 Harness 还需要保留官方 Logo、席位、路由和浅色主题行为。

## Decision

把视觉场接入现有 ui-theme 客户端生命周期。使用无依赖 Canvas 引擎绘制极光、90px 星座网格和轻量粒子鲸鱼；增加仅深色启用的磨砂表面规则；为 frame、侧栏、详情栏、对话、助手和用户气泡加上稳定的 data-ui-surface 属性。浅色调色板保持不变。页面隐藏时暂停，帧率节流到 30fps，并遵守减少动态和减少透明度偏好。

## Alternatives considered

**原样复制参考插件。** 不采用，因为其中的运行时哈希类名与当前 DSH 源码组件不稳定匹配。

**新增可选皮肤 bundle。** 不采用，因为此 checkout 的默认 Web 构成已经由 ui-theme 负责；把视觉场放在这里可以让本地构建无需第二次编辑 profile 就获得效果。

**两个主题都使用深色材质。** 不采用，因为提供的参考明确保持官方浅色主题不变，用户需要可靠的浅色回退。

## Consequences

默认 Web UI 现在会从常规 ui-theme bundle 加载参考风格视觉场。背景层仅用于装饰且不接收指针事件，导航、调整栏宽、composer 输入、设置和消息操作仍走原有交互路径。官方 DSH Logo 与构建元数据位置保持不变。GUI 测试在有意更新侧栏快照后保持全绿；真实 Web replay 仍存在之前的滚动/设置 fixture 失败，已单独记录。
