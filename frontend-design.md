# Nvex Demo 前端 Wireframe / 页面信息架构

## 副标题

面向 Investor Demo 的前端结构、页面叙事与交互定义

## 适用对象

- 产品团队
- 设计团队
- 前端工程团队
- 创始团队 / Demo 讲解人

## 文档目的

本文件用于定义 Nvex investor demo 的前端信息架构、页面结构、关键组件、状态设计与演示路径，帮助团队快速完成以下对齐：

1. 明确 Nvex demo 的产品层表达方式
2. 将前端页面组织成一条清晰的 investor narrative
3. 区分 **Nvex orchestration layer** 与 **AlphaBrain execution/runtime layer**
4. 让设计、前端、产品能够并行推进页面落地

---

# 1. 产品导航模型

Nvex demo 的导航不应沿用传统数据平台、标注后台或通用 MLOps 控制台的组织方式。传统工具通常围绕“数据集、模型、任务、作业”展开，而 Nvex 的核心价值不在于管理对象本身，而在于**识别失败、生成修复策略、驱动迭代并沉淀平台资产**。因此，主导航应围绕 **failure intelligence** 和 **iteration loop** 组织，让用户天然感知这是一个智能编排层，而不是另一个训练面板。

从 investor demo 的叙事角度，前端需要把复杂的底层执行能力抽象为一条清晰闭环：**导入项目 → 识别失败 → 生成 patch plan → 执行一次迭代 → 展示 checkpoint 改进 → 沉淀复用资产**。这也是 Nvex 与普通 annotation 工具、MLOps 平台、实验管理系统的根本差异。

## 推荐一级导航

- Home / Project Hub
- Project Overview
- Failure Map
- Patch Plan
- Iteration Runner
- Improvement Report
- Platform Memory

---

# 2. Sitemap / 信息架构树

```text
Nvex Demo
├── Home / Project Hub
│   ├── Demo 项目概览
│   ├── Recent Projects
│   ├── Featured Run
│   └── Create / Import Project
│
├── Project Overview
│   ├── Project Summary
│   ├── Current Checkpoint
│   ├── Eval Snapshot
│   ├── Risk Flags
│   └── Recommended Next Action
│
├── Failure Map
│   ├── KPI Overview
│   ├── Breakdown by Task
│   ├── Breakdown by Scene / Condition
│   ├── Failure Clusters
│   ├── Representative Episodes
│   └── Root-Cause Hypotheses
│
├── Patch Plan
│   ├── Gap Summary
│   ├── Data Targeting
│   ├── Data Recipe
│   ├── Training Strategy
│   ├── Verification Setup
│   └── Expected Uplift
│
├── Iteration Runner
│   ├── Run Timeline
│   ├── Runtime Status
│   ├── Job Artifacts
│   ├── Version Tracking
│   └── Live Logs / Stage Updates
│
├── Improvement Report
│   ├── Before vs After
│   ├── KPI Lift
│   ├── Failure Reduction
│   ├── Checkpoint Summary
│   ├── Artifacts Comparison
│   └── Recommended Next Iteration
│
└── Platform Memory
    ├── Reusable Recipes
    ├── Pipeline Templates
    ├── Failure Ontology
    ├── Reuse Insights
    └── Project-to-Project Learnings
```

---

# 3. 页面级详细定义

## A. Home / Project Hub

### 页面目标

作为 investor demo 的入口页，快速解释 Nvex 是什么、当前 demo 看什么、从哪里开始。该页应强化“从失败到改进”的产品主张，而不是做成普通项目列表页。

### 核心受众

- 投资人
- 创始团队
- 首次接触产品的客户

### 页面模块 / 组件清单

- 顶部全局导航
- 左侧侧边导航
- Hero 区 / 产品主张
- Featured Demo Project
- Recent Projects
- Quick Actions
- 平台价值摘要卡片

### 核心数据对象

- Project
- DemoRunSummary
- RecentActivity
- FeaturedProject

### 推荐主 CTA / 次 CTA

- 主 CTA：进入 Demo 项目
- 次 CTA：导入项目 / 查看平台记忆

### 页面成功标准

- 用户 10 秒内理解 Nvex 是智能编排层
- 用户明确知道本次 demo 从哪个项目开始
- 首页不暴露底层训练复杂度

---

## B. Project Overview

### 页面目标

建立当前项目上下文，展示现有 checkpoint、评测快照、风险与下一步建议，为后续 failure analysis 做铺垫。

### 核心受众

- 投资人
- 产品经理
- 解决方案团队
- 技术负责人

### 页面模块 / 组件清单

- 项目头部信息
- Checkpoint 状态卡
- Eval Snapshot 卡片组
- 风险标记区
- 推荐动作卡片
- 可用资产概览

### 核心数据对象

- Project
- Checkpoint
- EvalRun
- RiskFlag
- ReusableAsset

### 推荐主 CTA / 次 CTA

- 主 CTA：查看失败分析
- 次 CTA：查看评测详情 / 启动下一步规划

### 页面成功标准

- 用户理解当前模型状态
- 用户看到明确的“下一步建议”
- 页面成为从静态项目到动态诊断的过渡

---

## C. Failure Map

### 页面目标

将 benchmark/eval 结果转化为可理解的 failure intelligence，说明模型“为什么失败”而不是“分数低”。

### 核心受众

- 投资人
- ML / Robotics 技术负责人
- 内部研究团队

### 页面模块 / 组件清单

- KPI 总览
- Breakdown 图表
- Failure Cluster 卡片
- Representative Episode 区
- Root Cause Hypothesis 面板
- Nvex Diagnosis 总结区

### 核心数据对象

- EvalRun
- FailureCluster
- EpisodeArtifact
- RootCauseHypothesis
- MetricBreakdown

### 推荐主 CTA / 次 CTA

- 主 CTA：生成 Patch Plan
- 次 CTA：导出诊断 / 查看原始 artifact

### 页面成功标准

- 用户能说出 2-3 个明确 failure clusters
- 用户理解修复应围绕 failure，而不是盲目补数据
- 页面具备“wow moment”

---

## D. Patch Plan

### 页面目标

把 failure diagnosis 转化为结构化的下一步行动方案，体现 Nvex 的 orchestration 能力。

### 核心受众

- 投资人
- 产品经理
- 技术负责人
- 解决方案团队

### 页面模块 / 组件清单

- Gap Summary
- Data Targeting 卡片
- Data Recipe 卡片
- Training Strategy 卡片
- Verification Setup 卡片
- 影响预估 / 风险 / 置信度区
- 审批区 / CTA 区

### 核心数据对象

- PatchPlan
- DataTargetSpec
- TrainingStrategy
- VerificationSpec
- ExpectedImpact

### 推荐主 CTA / 次 CTA

- 主 CTA：批准并运行
- 次 CTA：调整方案 / 保存计划

### 页面成功标准

- 用户能看懂下一轮修复动作
- 方案以结构化方式呈现，而非一段自然语言
- 页面体现 Nvex 不只是 dashboard

---

## E. Iteration Runner

### 页面目标

展示 Nvex 正在调度底层 runtime 执行一次真实迭代，证明这是可执行系统而非静态建议工具。

### 核心受众

- 投资人
- 技术团队
- 创始团队

### 页面模块 / 组件清单

- 运行时间线
- 当前阶段状态卡
- Runtime 指标卡
- Artifact 列表
- 版本追踪区
- 日志 / 事件流

### 核心数据对象

- IterationRun
- JobStage
- RuntimeArtifact
- VersionRecord
- ExecutionStatus

### 推荐主 CTA / 次 CTA

- 主 CTA：查看结果
- 次 CTA：停止运行 / 查看日志 / 返回计划

### 页面成功标准

- 用户相信系统在真实执行
- 页面清晰表达当前跑到哪一步
- 不让底层工程细节淹没主叙事

---

## F. Improvement Report

### 页面目标

展示 before/after 的量化提升和定性变化，证明 Nvex 能把模型失败转化为更好的 checkpoint。

### 核心受众

- 投资人
- 客户决策者
- 产品与销售团队

### 页面模块 / 组件清单

- Before vs After KPI
- 失败模式收敛对比
- Rollout / Video 对比
- 新 checkpoint 摘要
- 新增资产卡片
- 下一轮建议

### 核心数据对象

- ImprovementReport
- BeforeAfterMetric
- CheckpointDiff
- ArtifactComparison
- CreatedAsset

### 推荐主 CTA / 次 CTA

- 主 CTA：查看平台沉淀
- 次 CTA：启动下一轮迭代 / 分享结果

### 页面成功标准

- 用户清楚看到 measurable uplift
- 用户理解这不只是一次实验，而是形成平台资产
- 页面成为 investor demo 的核心收束页

---

## G. Platform Memory

### 页面目标

展示项目执行后沉淀的可复用 recipe、template、failure pattern，强化平台复利逻辑。

### 核心受众

- 投资人
- 创始团队
- 产品与解决方案团队

### 页面模块 / 组件清单

- Recipe Library
- Pipeline Template Library
- Failure Ontology
- Reuse Insights
- Cross-project Learnings

### 核心数据对象

- ReusableAsset
- Recipe
- Template
- FailurePattern
- ReuseMetric

### 推荐主 CTA / 次 CTA

- 主 CTA：查看复用详情
- 次 CTA：返回项目 / 启动新项目

### 页面成功标准

- 用户理解每轮迭代会沉淀资产
- 页面帮助完成从“服务交付”到“平台系统”的认知跃迁
- 与 Improvement Report 形成闭环

---

# 4. 每页低保真 ASCII Wireframe

## A. Home / Project Hub

```text
+----------------------------------------------------------------------------------+
| Nvex                               [搜索项目]                     [导入项目]      |
+----------------------------------------------------------------------------------+
| Sidebar            | Hero / Value Proposition                                    |
| - Home             | "将模型失败转化为更好的 checkpoint"                         |
| - Overview         | Agent-in-the-loop orchestration for Physical AI             |
| - Failure Map      | [进入 Demo 项目] [查看平台记忆]                             |
| - Patch Plan       |                                                             |
| - Iteration        +-------------------------------------------------------------+
| - Report           | Featured Demo Project                                       |
| - Memory           | [LIBERO Demo]  当前成功率 62%  主要失败: 遮挡/恢复不足       |
|                    | [进入项目]                                                  |
|                    +-------------------------------------------------------------+
|                    | Recent Projects                                             |
|                    | [项目卡1] [项目卡2] [项目卡3]                               |
|                    +-------------------------------------------------------------+
|                    | Platform Signals                                            |
|                    | [失败诊断] [Patch Plan] [执行迭代] [资产沉淀]               |
+----------------------------------------------------------------------------------+
```

---

## B. Project Overview

```text
+----------------------------------------------------------------------------------+
| Nvex / Demo Project: LIBERO Kitchen                                  [查看失败分析]|
+----------------------------------------------------------------------------------+
| Sidebar            | Project Header                                              |
|                    | Checkpoint: ckpt_v0.7   Domain: manipulation                |
|                    | Eval Suite: LIBERO goal + robustness                        |
|                    +-------------------------------------------------------------+
|                    | KPI Summary         | Risk Flags                             |
|                    | - Success 62%       | - Occlusion gap                        |
|                    | - 4 fail clusters   | - Recovery weak                        |
|                    | - 2 prior runs      | - Verification limited                 |
|                    +-------------------------------------------------------------+
|                    | Available Assets     | Recommended Next Action                |
|                    | - 2 recipes          | Patch occlusion/recovery via CL run    |
|                    | - 1 template         | [生成 Patch Plan] [查看详情]           |
+----------------------------------------------------------------------------------+
```

---

## C. Failure Map

```text
+----------------------------------------------------------------------------------+
| Failure Map                                                      [生成 Patch Plan]|
+----------------------------------------------------------------------------------+
| Sidebar            | KPI Row                                                     |
|                    | [Overall Success 62%] [Top Failure: Occlusion] [Conf 0.81]  |
|                    +-------------------------------------------------------------+
|                    | Breakdown Panel      | Root Cause Panel                      |
|                    | - By Task            | 1. 遮挡场景感知不足                    |
|                    | - By Scene           | 2. 失误后恢复轨迹缺失                  |
|                    | - By Condition       | 3. 验证覆盖不完整                      |
|                    +-------------------------------------------------------------+
|                    | Failure Clusters                                            |
|                    | [Cluster A] [Cluster B] [Cluster C] [Cluster D]             |
|                    +-------------------------------------------------------------+
|                    | Representative Episodes                                     |
|                    | [Video / Rollout 1] [Video / Rollout 2]                     |
|                    +-------------------------------------------------------------+
|                    | Nvex Diagnosis                                               |
|                    | 模型在部分遮挡下抓取偏移，且缺少二次修正行为。                |
+----------------------------------------------------------------------------------+
```

---

## D. Patch Plan

```text
+----------------------------------------------------------------------------------+
| Patch Plan                                                        [批准并运行]    |
+----------------------------------------------------------------------------------+
| Sidebar            | Gap Summary                                                 |
|                    | - 遮挡场景样本不足                                           |
|                    | - 恢复行为标签缺失                                           |
|                    | - 验证环境覆盖有限                                           |
|                    +-------------------------------------------------------------+
|                    | Data Targeting      | Training Strategy                      |
|                    | - 120 patch episodes| - Continual Learning                   |
|                    | - 40 teleop fixes   | - 冻结底座，小规模增量                 |
|                    | - 1 light variant   | - 低成本短周期                         |
|                    +-------------------------------------------------------------+
|                    | Verification Setup  | Expected Outcome                       |
|                    | - Robustness subset | +8~12% uplift                          |
|                    | - Occlusion replay  | Confidence 0.73                        |
|                    | - Checkpoint gating | ETA 45 min                             |
|                    +-------------------------------------------------------------+
|                    | [调整方案] [保存计划] [批准并运行]                           |
+----------------------------------------------------------------------------------+
```

---

## E. Iteration Runner

```text
+----------------------------------------------------------------------------------+
| Iteration Runner                                                   [查看日志]     |
+----------------------------------------------------------------------------------+
| Sidebar            | Run Timeline                                                 |
|                    | [1 Patch Spec] -> [2 Train] -> [3 Re-eval] -> [4 Report]    |
|                    +-------------------------------------------------------------+
|                    | Runtime Status      | Artifacts                              |
|                    | - Stage: Re-eval    | - training_log.txt                     |
|                    | - ETA: 12 min       | - eval_result.json                     |
|                    | - GPU: 2            | - rollout_videos/                      |
|                    | - Progress: 71%     | - checkpoint_meta.json                 |
|                    +-------------------------------------------------------------+
|                    | Version Tracking                                             |
|                    | ckpt_v0.7  ->  iter_01  ->  candidate_ckpt_v0.8             |
|                    +-------------------------------------------------------------+
|                    | Live Console / Events                                        |
|                    | > loading checkpoint...                                      |
|                    | > running continual learning...                              |
|                    | > evaluating robustness subset...                            |
|                    +-------------------------------------------------------------+
|                    | [停止运行] [返回计划] [查看结果]                             |
+----------------------------------------------------------------------------------+
```

---

## F. Improvement Report

```text
+----------------------------------------------------------------------------------+
| Improvement Report                                                [分享结果]      |
+----------------------------------------------------------------------------------+
| Sidebar            | Before vs After                                              |
|                    | Success 62% -> 74%                                           |
|                    | Fail Clusters 4 -> 2                                         |
|                    | Recovery 31% -> 55%                                          |
|                    +-------------------------------------------------------------+
|                    | Artifact Comparison                                           |
|                    | [Before Video]                [After Video]                  |
|                    +-------------------------------------------------------------+
|                    | What Changed          | New Assets                            |
|                    | - Added patch data    | - occlusion_recovery_v1              |
|                    | - Improved recovery   | - clutter_patch_template             |
|                    | - Expanded verification| - grasp_after_occlusion_miss         |
|                    +-------------------------------------------------------------+
|                    | Recommended Next Iteration                                   |
|                    | 下一步建议：处理语言变化下的时序规划问题                      |
|                    | [查看平台沉淀] [启动下一轮]                                  |
+----------------------------------------------------------------------------------+
```

---

## G. Platform Memory

```text
+----------------------------------------------------------------------------------+
| Platform Memory                                                   [返回项目]      |
+----------------------------------------------------------------------------------+
| Sidebar            | Reusable Recipes                                             |
|                    | [occlusion_v1] [recovery_v2] [tactile_sync_v1]              |
|                    +-------------------------------------------------------------+
|                    | Pipeline Templates                                           |
|                    | [libero_patch_pipeline] [teleop_fix_flow]                    |
|                    +-------------------------------------------------------------+
|                    | Failure Ontology                                             |
|                    | [perception gap] [recovery missing] [env mismatch]           |
|                    +-------------------------------------------------------------+
|                    | Reuse Insights                                               |
|                    | 本项目复用了 2 个 recipe，新增 1 个 template。                |
|                    +-------------------------------------------------------------+
|                    | Cross-project Learnings                                      |
|                    | 遮挡恢复策略可迁移至相邻 manipulation 场景。                  |
+----------------------------------------------------------------------------------+
```

---

# 5. 页面模块与组件建议

## 5.1 导航类组件

| 组件名称     | 作用                     | 典型字段                               |
| ------------ | ------------------------ | -------------------------------------- |
| 顶部全局导航 | 全局品牌、搜索、关键入口 | logo、search、user menu、import action |
| 侧边导航     | Investor demo 主路径导航 | nav label、icon、active state          |
| 面包屑       | 表示当前项目/页面层级    | current page、project name             |

## 5.2 数据概览类组件

| 组件名称   | 作用           | 典型字段                                    |
| ---------- | -------------- | ------------------------------------------- |
| KPI 卡片   | 展示关键信号   | title、value、delta、status                 |
| 项目摘要卡 | 概括项目状态   | project name、domain、checkpoint、suite     |
| 风险标记卡 | 强调主要问题   | risk type、severity、note                   |
| 资产摘要卡 | 概括可复用沉淀 | recipes count、templates count、reuse score |

## 5.3 诊断类组件

| 组件名称             | 作用                                     | 典型字段                                      |
| -------------------- | ---------------------------------------- | --------------------------------------------- |
| Failure Cluster 卡片 | 聚类展示失败模式                         | cluster name、frequency、severity、confidence |
| Root Cause Panel     | 呈现原因假设                             | hypothesis、evidence、confidence              |
| Breakdown 图表       | 展示 task/scene/condition 维度的表现差异 | dimension、value、benchmark                   |
| Episode Viewer       | 展示代表性失败样本                       | video url、episode id、notes                  |

## 5.4 执行类组件

| 组件名称        | 作用             | 典型字段                                    |
| --------------- | ---------------- | ------------------------------------------- |
| Patch Plan 卡片 | 展示修复策略     | gap、target spec、strategy、expected uplift |
| 运行时间线      | 展示执行阶段     | stage、status、progress、eta                |
| Runtime 状态卡  | 展示当前作业状态 | job id、gpu、step、current phase            |
| 日志流组件      | 展示关键执行事件 | timestamp、event、severity                  |

## 5.5 结果类组件

| 组件名称            | 作用                    | 典型字段                          |
| ------------------- | ----------------------- | --------------------------------- |
| Before/After 对比卡 | 展示结果变化            | metric name、before、after、delta |
| Artifact 对比器     | 展示视频或 rollout 差异 | before artifact、after artifact   |
| Checkpoint 摘要卡   | 展示新 checkpoint 信息  | version、created at、summary      |
| 下一轮建议卡        | 连接下一次迭代          | next issue、recommended action    |

## 5.6 平台记忆类组件

| 组件名称              | 作用             | 典型字段                                   |
| --------------------- | ---------------- | ------------------------------------------ |
| Recipe 卡片           | 展示数据修复配方 | recipe name、modality mix、reuse count     |
| Template 卡片         | 展示流程模板     | template name、applies to、last used       |
| Failure Ontology 卡片 | 展示失败知识结构 | pattern name、description、linked projects |
| Reuse Insight 卡片    | 展示平台复利     | reused assets、new assets、coverage note   |

---

# 6. 关键状态设计

## 6.1 全局状态

| 状态    | 含义               | 用户提示                                         | 推荐 CTA            |
| ------- | ------------------ | ------------------------------------------------ | ------------------- |
| empty   | 当前无数据或无项目 | 暂无可展示项目，请导入 demo 数据或选择示例项目。 | 导入项目 / 打开示例 |
| loading | 数据加载中         | 正在加载项目上下文与评测结果。                   | 无 CTA，显示骨架屏  |
| running | 当前有执行中作业   | Nvex 正在编排本轮迭代，请稍后查看结果。          | 查看运行状态        |
| success | 当前流程执行完成   | 本轮迭代已完成，结果可用于对比与复盘。           | 查看结果            |
| failure | 加载或执行失败     | 当前操作未成功完成，请检查输入或重试。           | 重试 / 查看日志     |

## 6.2 页面级关键状态

### Home / Project Hub

| 状态       | 用户提示                           | 推荐 CTA            |
| ---------- | ---------------------------------- | ------------------- |
| 无项目     | 当前无项目，请导入或打开示例项目。 | 导入项目 / 打开示例 |
| 有推荐项目 | 已为本次演示准备示例项目。         | 进入 Demo 项目      |

### Project Overview

| 状态       | 用户提示                       | 推荐 CTA     |
| ---------- | ------------------------------ | ------------ |
| 无评测结果 | 当前 checkpoint 尚无评测快照。 | 运行评测     |
| 评测可用   | 已检测到可用评测结果。         | 查看失败分析 |

### Failure Map

| 状态       | 用户提示                         | 推荐 CTA                |
| ---------- | -------------------------------- | ----------------------- |
| 无诊断结果 | 评测结果不足以生成 failure map。 | 返回项目概览 / 运行评测 |
| 诊断已生成 | 已识别关键 failure clusters。    | 生成 Patch Plan         |

### Patch Plan

| 状态         | 用户提示                 | 推荐 CTA     |
| ------------ | ------------------------ | ------------ |
| 计划待生成   | 尚未生成 patch plan。    | 自动生成方案 |
| 计划待审批   | 已生成方案，待确认执行。 | 批准并运行   |
| 方案需要调整 | 当前方案存在冲突或缺失。 | 调整方案     |

### Iteration Runner

| 状态         | 用户提示                        | 推荐 CTA        |
| ------------ | ------------------------------- | --------------- |
| 等待启动     | 尚未开始本轮执行。              | 启动运行        |
| 训练进行中   | 正在执行训练或增量更新。        | 查看日志        |
| 评测进行中   | 正在验证 candidate checkpoint。 | 查看运行状态    |
| 已完成待查看 | 结果已生成。                    | 查看结果        |
| 执行失败     | 当前作业中断或失败。            | 重试 / 查看日志 |

### Improvement Report

| 状态       | 用户提示                         | 推荐 CTA                  |
| ---------- | -------------------------------- | ------------------------- |
| 无结果     | 当前尚未生成对比结果。           | 返回运行页面              |
| 结果已归档 | 本轮 improvement report 已归档。 | 查看平台沉淀 / 启动下一轮 |

### Platform Memory

| 状态     | 用户提示                     | 推荐 CTA     |
| -------- | ---------------------------- | ------------ |
| 暂无沉淀 | 当前项目尚未沉淀可复用资产。 | 完成一次迭代 |
| 已有沉淀 | 可复用资产已进入平台记忆层。 | 查看复用详情 |

---

# 7. 推荐标签与 CTA 文案

## 7.1 导航标签

- 项目中心
- 项目概览
- 失败分析
- 修复方案
- 迭代执行
- 结果报告
- 平台记忆

## 7.2 卡片标题

- 当前 Checkpoint
- 评测快照
- 主要风险
- 关键失败簇
- 根因假设
- 修复策略
- 验证方案
- 结果提升
- 新增平台资产
- 复用洞察

## 7.3 空状态提示

- 当前暂无项目数据，请导入示例项目开始演示。
- 当前 checkpoint 尚无可用评测结果。
- 尚未识别到可供分析的失败模式。
- 当前项目尚未沉淀平台资产。

## 7.4 按钮 CTA

- 进入 Demo 项目
- 查看失败分析
- 生成 Patch Plan
- 批准并运行
- 查看运行状态
- 查看结果报告
- 查看平台沉淀
- 启动下一轮迭代
- 导出诊断摘要
- 返回项目概览

## 7.5 成功反馈

- Patch Plan 已生成，可进入审批与执行。
- 本轮迭代已完成，结果已同步至报告页。
- 新 checkpoint 已归档，平台资产已更新。
- 复用资产已写入平台记忆层。

## 7.6 错误反馈

- 当前评测结果不足，暂无法生成诊断。
- 本轮执行未完成，请检查运行日志后重试。
- 无法加载 artifact，请稍后刷新。
- 当前方案缺少必要输入，无法提交执行。

---

# 8. 视觉设计方向

## 8.1 整体风格

视觉风格建议延续现有 deck 的品牌调性：**深色科技感、蓝紫渐变、系统编排感、闭环与节点图语言**。页面整体应体现“智能调度系统”的高级感，而不是“工具后台”的操作感。

## 8.2 色彩建议

- 主背景：深紫黑 / 深石墨色
- 品牌高亮：蓝紫渐变、洋红渐变、冷光 cyan 辅助
- 语义色：
  - success：青绿或冷绿
  - warning：琥珀橙
  - failure：偏红紫，而非纯红
- 图表建议避免高饱和杂色，保持系统感

## 8.3 版式建议

- 采用左侧固定导航 + 顶部状态栏 + 中央主内容区
- 核心 KPI 和关键 CTA 保持首屏可见
- 每页只突出一个主叙事焦点，避免信息堆叠
- 模块间采用卡片与留白构建层级

## 8.4 卡片建议

- 使用微发光边框或渐变描边
- 关键卡片可使用轻微玻璃拟态，但不宜过重
- 卡片内部强调标题、数值、结论、动作四段式结构

## 8.5 图表建议

- 优先使用简洁条形图、热力图、雷达图、矩阵对比
- Failure cluster 建议卡片化 + 小图标 + 置信度
- Before/After 对比建议用并排卡片 + 明确 delta

## 8.6 节点 / 流程图建议

- 用 loop / graph 风格表达 intelligence loop
- 在 Patch Plan、Iteration Runner 中引入阶段节点
- 节点状态可配合动效展示运行进度

## 8.7 动效建议

- 页面切换：轻量淡入 + 渐变滑动
- Running 状态：节点脉冲、路径流动、进度高亮
- 成功状态：数值跃迁、卡片边框发光强化
- 动效应服务于“编排感”，而非炫技

## 避免事项

不要做成普通数据标注后台、任务工单系统或通用 MLOps 控制台；前端必须优先表达 **failure-driven orchestration**，而不是“管理数据/训练作业”。

---

# 9. 投资人 Demo Flow

## Step 1：从 Home / Project Hub 开始

**展示页面**：Home / Project Hub
**讲述内容**：Nvex 不是训练框架，而是将模型失败转化为可执行改进的 orchestration layer。
**希望投资人感知到的价值**：这是一个产品层，而不是研究代码或标注工具。

## Step 2：进入 Project Overview

**展示页面**：Project Overview
**讲述内容**：当前模型 checkpoint 已具备基础能力，但在特定场景下表现不稳定，Nvex 已识别出下一步最值得处理的问题。
**希望投资人感知到的价值**：系统不是泛泛而谈，而是在具体项目语境中做出判断。

## Step 3：展开 Failure Map

**展示页面**：Failure Map
**讲述内容**：Nvex 读懂 benchmark/eval 结果，定位模型失败的主要模式，并给出根因假设。
**希望投资人感知到的价值**：Nvex 的起点是 failure intelligence，而不是盲目加数据。

## Step 4：生成 Patch Plan

**展示页面**：Patch Plan
**讲述内容**：系统把 failure 转化为结构化 patch plan，包括补什么数据、用什么训练路径、如何验证。
**希望投资人感知到的价值**：Nvex 具备决策与编排能力，而不仅是分析能力。

## Step 5：展示 Iteration Runner

**展示页面**：Iteration Runner
**讲述内容**：Nvex 正在调度底层 runtime 执行一次真实迭代，AlphaBrain 负责实际训练、评测与 artifact 生成。
**希望投资人感知到的价值**：这是可执行闭环，不是静态报告。

## Step 6：收束到 Improvement Report + Platform Memory

**展示页面**：Improvement Report、Platform Memory
**讲述内容**：本轮迭代带来了更好的 checkpoint，并沉淀为 recipe、template、failure pattern 等可复用资产。
**希望投资人感知到的价值**：Nvex 将模型失败转化为更好的 checkpoint，并持续积累平台资产，形成复利式 moat。

---

# 10. 附录：页面到后端能力映射

| 页面               | 前端层角色      | 主要依赖的 AlphaBrain 能力                                   | 输出给前端的关键对象                             |
| ------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| Home / Project Hub | Nvex 展示层     | 项目元数据、已有 run artifact 汇总                           | Project、RunSummary、Activity                    |
| Project Overview   | Nvex 上下文层   | baseline eval 输出、benchmark snapshot                       | Checkpoint、EvalRun、RiskFlag                    |
| Failure Map        | Nvex 诊断层     | benchmark/eval 结果、world model eval artifact、episode 输出 | FailureCluster、MetricBreakdown、EpisodeArtifact |
| Patch Plan         | Nvex 编排层     | eval 结果、训练能力可用性、历史配置模板                      | PatchPlan、TrainingStrategy、VerificationSpec    |
| Iteration Runner   | Nvex 执行观察层 | baseline VLA、continual learning、re-eval、artifact 输出     | IterationRun、JobStage、RuntimeArtifact          |
| Improvement Report | Nvex 结果层     | before/after eval、artifact comparison、checkpoint 输出      | ImprovementReport、CheckpointDiff、MetricDelta   |
| Platform Memory    | Nvex 平台记忆层 | 历史 run、配置模板、结果归档                                 | ReusableAsset、Recipe、Template、FailurePattern  |

## 说明

- **Nvex 前端展示的是 orchestration layer**：负责理解失败、定义修复动作、组织迭代、沉淀资产
- **AlphaBrain 是 execution/runtime layer**：负责训练、评测、world model、artifact 生成等执行能力
- 前端不直接暴露底层训练复杂度，而是将其包装为 investor 可理解的 intelligence loop

---

# 11. 交付建议

本文件适合作为以下工作的直接输入：

1. **交给设计师**：用于产出高保真界面与视觉系统，建议优先设计 4 个核心页面：Project Overview、Failure Map、Patch Plan、Improvement Report
2. **交给前端工程师**：用于搭建路由结构、页面骨架、组件库与状态流转，建议先以静态假数据完成页面联调，再逐步接入真实后端
3. **交给产品与创始团队**：用于统一 investor demo narrative，确保页面顺序、字段命名、CTA 文案与现场讲解一致

建议落地顺序：

- 第一阶段：完成页面骨架与关键状态
- 第二阶段：接入真实 artifact 与结果数据
- 第三阶段：补充动效、对比组件与平台记忆层展示

最终目标不是做一个“功能齐全的后台”，而是做一个能在几分钟内清晰证明 Nvex 核心价值的 investor demo。
