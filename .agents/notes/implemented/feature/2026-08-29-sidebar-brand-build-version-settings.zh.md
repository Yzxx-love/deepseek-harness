# Agent Note: 保持官方品牌稳定并将本地构建元数据移到设置

Status: implemented

[English](2026-08-29-sidebar-brand-build-version-settings.md) | 中文
## Problem

本地构建回退会在侧栏品牌旁堆叠「DSH 本地构建」与版本-提交-dirty 徽标。构建元数据让导航身份随构建模式变化，也占用了应用外壳最窄的位置。

## Decision

侧栏回退通过现有 BrandWordmark 原子组件渲染官方鱼形标记与 DeepSeek Harness 字标，不再读取或格式化 DSH_CLIENT_VERSION、DSH_CLIENT_COMMIT_HASH 与 DSH_CLIENT_GIT_DIRTY。ui-settings-general 在「设置 → 通用」注册只读的「构建版本」行，在该处展示同一份公开元数据；构建时没有版本元数据时该行不显示。

## Alternatives considered

**继续把徽标放在品牌旁。** 不采用，因为构建来源是诊断元数据而不是导航身份，两行徽标会挤压侧栏。

**完全删除元数据。** 不采用，因为本地构建仍需要可见的版本和提交状态用于支持与复现；设置页提供稳定且容易发现的位置。

**让所有本地构建都切换为 official profile。** 不采用，因为 profile 控制部署身份的 slot 注册；保留回退逻辑可以让 local 与 official profile 继续组合，同时让两者拥有相同的可见品牌。

## Consequences

本地与官方构建的侧栏都保持稳定的官方身份。设置页展示公开的构建版本、可选短提交和 dirty 标记，不新增持久化字段，也不暴露凭据。测试同时固定侧栏徽标缺席和通用设置行的格式化；该行仅负责呈现，不会进入模型请求。
