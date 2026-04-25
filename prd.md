# Nvex Demo PRD》

## 一句话定位

**Nvex 是面向 Physical AI post-training 的 agent-in-the-loop orchestration layer：它以模型失败为起点，驱动 eval → gap analysis → data targeting → post-training → re-eval 的 intelligence loop；AlphaBrain 作为底层 execution/runtime layer，负责训练、评测与环境执行。**

---

## 1. 产品概述

### 1.1 产品定义

Nvex Demo 是一个面向投资人、同时可供产品/设计/工程团队执行的演示型产品。它的核心目标不是展示“又一个模型训练框架”，也不是展示“一个更高效的标注平台”，而是证明一个更高层的产品命题：

> **当一个 Physical AI policy 失败时，Nvex 可以识别失败模式、定位能力缺口、生成 targeted patch plan，并调度底层执行系统完成一次可验证的 checkpoint 改进。**

在系统分层上：

- **Nvex**：负责理解失败、制定修复策略、编排实验流程、沉淀可复用资产。
- **AlphaBrain**：负责执行 baseline VLA、continual learning、world model eval、benchmark/eval artifact 生成等底层能力。

### 1.2 Demo 要证明的核心价值

本次 demo 需要让投资人和内部团队在 3–5 分钟内理解三件事：

1. **Nvex 不是训练框架**它不与底层 VLA / RL / world model runtime 竞争，而是站在更高一层，定义“下一轮该做什么”。
2. **Nvex 不是普通标注工具**它不是以 annotation task 为起点，而是以 failure diagnosis 为起点。
3. **Nvex 的价值在 intelligence loop**
   它把 `eval → gap analysis → data targeting → post-training → re-eval` 串成一个闭环，并在每轮迭代中沉淀平台资产。

### 1.3 Demo 范围

本 PRD 面向的不是完整商用平台，而是一个 **investor-demo oriented MVP**。重点是：

- 有明确故事线
- 有可点击、可讲解的前端体验
- 至少有一条真实或半真实的执行闭环
- 能产出 before/after 改进结果
- 能体现“每个项目会沉淀为平台资产”的平台逻辑

---

## 2. 背景与机会

### 2.1 行业背景

Physical AI 正从研究演示走向落地部署，但其核心瓶颈并不只是“模型不够强”，而是模型迭代系统不够成熟。当前多数团队在出现失败后，通常依赖以下低效方式：

- 盲目追加更多数据
- 依赖人工经验判断问题来源
- 训练、评测、数据采集、环境验证彼此割裂
- 缺乏统一的 failure-to-fix 闭环

### 2.2 产品机会

在 Physical AI 场景中，真正有价值的不是一次性训练，而是持续提升模型部署 readiness 的能力。这个过程要求系统同时具备：

- **评测能力**：知道模型在哪些任务、场景、模态上失败
- **归因能力**：知道为什么失败
- **策略能力**：知道该补什么数据、怎么训练、如何验证
- **执行能力**：能快速触发一次小步迭代
- **记忆能力**：能把经验沉淀为 reusable recipe/template/ontology

这就是 Nvex 的产品机会：成为 **Physical AI 的 post-training intelligence layer**。

### 2.3 为什么采用 AlphaBrain 作为底座

AlphaBrain 适合作为 demo 的 execution/runtime layer，原因是它已经具备本项目最关键的底层能力：

- **Baseline VLA train/eval**
- **Continual learning**
- **World model eval**
- **Benchmark/eval artifacts 输出**
- **LIBERO 等 benchmark 支撑**

因此，Nvex 不需要从零搭建训练与评测引擎，而可以将重点放在：

- failure analysis
- patch planning
- experiment orchestration
- result presentation
- platform memory

这使得 demo 可以在较短周期内具备“可演示 + 可执行”的说服力。

---

## 3. 产品目标

## 3.1 核心目标

本次 Nvex Demo 的核心目标有三项：

### 目标 A：证明 Nvex 的产品定义

证明 Nvex 是一个 orchestration layer，而不是：

- 又一个训练框架
- 又一个 dashboard
- 又一个标注平台
- 又一个 LLM 助手壳子

### 目标 B：跑通一条 Failure-to-Fix 闭环

在一个选定的 benchmark/use case 上，完成以下最小链路：

1. 导入现有 checkpoint / eval run
2. 展示 failure diagnosis
3. 生成 patch plan
4. 调度底层 AlphaBrain 执行一轮迭代
5. 展示 before/after improvement

### 目标 C：建立平台化想象力

让投资人和潜在客户看到：

- 每轮迭代不只是输出一个新 checkpoint
- 还会沉淀 recipe、template、failure pattern、verification setup 等可复用资产
- 因此产品具备 compound platform 属性

## 3.2 商业目标

### 商业目标 A：支持融资叙事

帮助投资人快速理解公司从“数据交付/标注服务”走向“intelligence infrastructure”的路径。

### 商业目标 B：支持客户沟通

向潜在客户证明：

- Nvex 能帮助团队减少 blind data collection
- Nvex 能提高 post-training 决策效率
- Nvex 能将 failure diagnosis 与环境验证统一到一个系统中

### 商业目标 C：支持内部定位统一

为产品、设计、工程团队提供统一北极星，避免 demo 被做成“满屏图表但没有决策能力”的管理后台。

## 3.3 非目标

本期 demo **不追求**以下内容：

- 覆盖所有机器人类型与行业场景
- 完整实现数据采集与标注工作台
- 支持全量生产级多租户与权限体系
- 实现所有训练路径的自动编排
- 构建完整 MLOps 平台
- 用真实在线训练替代一切预生成结果

本期重点是 **讲清产品价值并跑通关键闭环**，而不是追求平台功能完备。

---

## 4. 目标用户

## 4.1 主要用户：投资人 / 董事会观众

关注点：

- 这是不是一个平台级产品
- 它的 moat 在哪里
- 为什么不是一个服务业务包装
- 为什么每一轮项目都会强化平台资产
- 为什么这层 orchestration 值得独立存在

他们需要看到的是 **产品逻辑与平台逻辑**，不是工程参数。

## 4.2 次要用户：潜在客户中的 AI / Robotics 负责人

关注点：

- 当前失败的 policy 进入 Nvex 后会发生什么
- Nvex 如何帮助团队判断下一轮该补什么数据
- Nvex 如何决定使用 CL、SFT、env verification 等路径
- Nvex 是否能提升试错效率

他们需要看到的是 **从失败到 checkpoint improvement 的实际价值**。

## 4.3 内部用户：产品 / 设计 / 工程 / 解决方案团队

关注点：

- Demo 的最小范围是什么
- 前后端各自承担什么职责
- 哪些数据必须真实接入，哪些可以先 mock
- 页面结构与叙事顺序如何安排
- 后续如何从 demo 进化成正式产品

---

## 5. 核心 Narrative / Demo 故事线

## 5.1 主故事线：Failure-to-Fix

推荐以一条清晰的失败修复链路作为主叙事：

> 一个在 benchmark 中表现不佳的 Physical AI policy 被导入 Nvex。Nvex 不从“多收一点数据”开始，而是先进行 failure diagnosis。它识别出问题主要集中在某几类场景中，然后生成针对性的 patch plan，决定该补哪些数据、采用什么训练策略、在哪个环境中进行验证。随后，Nvex 调用 AlphaBrain 完成一次增量迭代，并展示新 checkpoint 的 measurable uplift，以及本次迭代沉淀出的 reusable assets。

## 5.2 关键传达信息

Demo 需要自然传达以下观点：

1. **Eval 是起点，不是终点**
2. **模型失败不是“更多数据”问题，而是“正确的数据与正确的验证”问题**
3. **Nvex 的核心不是分析，而是分析后的决策与编排**
4. **AlphaBrain 负责执行，Nvex 负责决定**
5. **每轮项目会增强平台资产，而不是只完成一次性交付**

## 5.3 建议演示场景

优先选择一个小而典型的 benchmark/use case，例如：

- LIBERO 中的 manipulation task
- 某类受 occlusion、camera shift 或 recovery behavior 影响明显的任务

原因：

- 可复用 AlphaBrain 现有 benchmark/eval 路径
- 易于构造前后对比
- 容易映射到“failure cluster → patch plan → improvement”的闭环

---

## 6. 核心用户流程

## 6.1 Flow A：Project Intake

### 目标

建立项目上下文，让系统知道当前任务、模型和已有评测结果。

### 输入

- Project 名称
- Use case / domain
- Robot / environment 类型
- 当前 checkpoint
- benchmark/eval 结果
- 可用数据来源
- 期望 KPI

### 输出

- Project summary
- 初步风险提示
- 推荐 eval/verification 路径
- 当前模型状态快照

## 6.2 Flow B：Failure Diagnosis

### 目标

将原始 benchmark 结果转化为可解释的 failure intelligence。

### 输入

- Eval artifact
- Task/scene breakdown
- 视频 / rollout 样本
- 历史 run 结果

### 输出

- overall success 概览
- failure cluster 列表
- root-cause hypothesis
- top gaps to fix

## 6.3 Flow C：Patch Plan Generation

### 目标

把诊断结果转化为一份结构化修复计划。

### 输入

- failure diagnosis
- 可用数据源
- 底层可执行能力
- 历史 recipe/template

### 输出

- data targeting spec
- data recipe
- training strategy
- verification setup
- expected uplift
- confidence

## 6.4 Flow D：Iteration Execution

### 目标

调度底层执行引擎完成一次迭代。

### 输入

- patch plan
- chosen execution path
- job config

### 输出

- training/eval job 状态
- checkpoint artifact
- before/after report
- iteration trace

## 6.5 Flow E：Improvement Review & Memory

### 目标

证明这不是一次性修复，而是会沉淀平台资产的闭环。

### 输入

- iteration result
- previous baseline
- generated artifacts

### 输出

- KPI uplift summary
- reduced failure modes
- reusable recipe/template
- next recommended iteration

---

## 7. 功能需求

## 7.1 模块一：Project Intake

### 模块目标

作为 demo 的入口页，负责定义当前项目、场景和目标，不让用户陷入底层参数细节。

### MVP 范围

- 创建/选择 demo project
- 载入 checkpoint 与 eval result
- 展示项目摘要和推荐下一步动作

### 核心输入

- use case
- domain
- robot/environment
- current checkpoint
- eval result
- target KPI

### 核心输出

- 项目摘要
- 当前 success snapshot
- top risk summary
- recommended next action

### 页面职责

- 帮助观众快速进入上下文
- 让用户理解 Nvex 是“从失败开始”的系统
- 为后续 Failure Map 提供入口

### 非 MVP

- 全量项目管理
- 多租户权限
- 复杂配置编辑器

---

## 7.2 模块二：Failure Map

### 模块目标

将复杂的评测结果压缩成投资人和客户都能快速理解的 failure intelligence dashboard。

### MVP 范围

- overall success 展示
- task / scene / modality breakdown
- failure cluster 可视化
- representative episode / video
- root-cause hypothesis

### 输入

- benchmark result
- eval logs
- rollout artifacts
- checkpoint metadata

### 输出

- top failure modes
- top root-cause hypotheses
- prioritized gaps

### 页面职责

- 让观众看到“模型具体在哪里失败”
- 从“结果差”上升到“知道为什么差”
- 为 Patch Plan 提供依据

### 关键要求

- 页面组织应围绕 failure，而不是原始日志
- 必须给出结构化 root-cause，而非仅展示 charts
- 如果可能，应展示 before episode 片段

---

## 7.3 模块三：Patch Plan

### 模块目标

从 failure diagnosis 直接生成“下一步该怎么做”的结构化行动方案。

### MVP 范围

- gap summary
- target data recommendation
- training strategy recommendation
- verification setup
- expected uplift

### 输入

- failure clusters
- historical recipes/templates
- available execution paths
- user constraints（时间/资源）

### 输出

- PatchPlan 对象
- action cards
- run recommendation

### 结构化输出字段建议

- `root_causes`
- `target_data_spec`
- `annotation_schema`
- `source_ratio`
- `training_strategy`
- `verification_spec`
- `expected_uplift`
- `confidence`

### 页面职责

- 把 Nvex 从 dashboard 提升为 orchestration layer
- 让投资人看到系统具备“下一步决策能力”
- 让工程和解决方案团队明确下游执行输入

### 非 MVP

- 全自动生成真实数据采集任务
- 动态预算优化
- 多轮 agent self-play planning

---

## 7.4 模块四：Iteration Runner

### 模块目标

承接 Patch Plan，调用底层 AlphaBrain 执行路径，完成一次可感知的迭代。

### MVP 范围

优先支持：

- baseline fine-tune / eval
- continual learning update
- re-eval
- artifact 汇总

可选增强：

- world model predicted-vs-rollout 对比

### 输入

- plan_id
- execution config
- selected checkpoint
- chosen backend path

### 输出

- job status
- run timeline
- output checkpoint
- evaluation result
- artifacts

### 页面职责

- 让用户感知“系统真的在跑”
- 把底层复杂执行过程抽象成清晰阶段
- 为 Improvement Report 提供数据

### 与 AlphaBrain 的关系

本模块直接依赖 AlphaBrain 提供的：

- baseline VLA train/eval
- continual learning
- world model eval
- benchmark artifacts

Nvex 不重复实现这些 runtime，而是以 job 编排与结果消费为主。

---

## 7.5 模块五：Improvement Report

### 模块目标

将一次迭代的结果转化为对投资人、客户和内部团队都有价值的结果页面。

### MVP 范围

- before vs after
- KPI uplift
- failure reduction summary
- representative comparison
- assets created
- next recommendation

### 输入

- baseline eval
- new eval
- generated artifacts
- patch plan metadata

### 输出

- report summary
- uplift numbers
- reduced cluster list
- next-step suggestion

### 页面职责

- 完成“修复闭环”的最后一环
- 明确展示 Nvex 带来的 measurable improvement
- 连接到 Platform Memory，证明平台化沉淀

### 关键要求

- 必须用变化来组织信息，而不是静态展示新结果
- 需要同时展示结果改善和资产沉淀

---

## 7.6 模块六：Platform Memory

### 模块目标

展示每轮项目如何沉淀为平台能力，形成 compound platform 逻辑。

### MVP 范围

- recipe 列表
- template 列表
- failure ontology
- reuse summary

### 输入

- previous projects
- patch plans
- iteration results
- manually curated patterns

### 输出

- reusable assets
- reuse count
- project memory summary

### 页面职责

- 证明 Nvex 不是“一次性项目交付系统”
- 让投资人看到软件平台的复利效应
- 帮助内部团队建立知识库结构

### 关键要求

即便第一版内容较轻，也必须保留该页面或模块，因为它直接承载平台估值逻辑。

---

## 8. 建议的数据模型

以下数据模型以“足够支撑 demo + 可扩展到后续产品化”为原则设计。

## 8.1 Project

```json
{
  "id": "proj_001",
  "name": "LIBERO Kitchen Demo",
  "use_case": "tabletop manipulation",
  "domain": "robotics",
  "robot_type": "sim manipulator",
  "environment": "LIBERO",
  "current_checkpoint": "ckpt_v0.7",
  "status": "underperforming",
  "target_kpi": {
    "success_rate": 0.75
  }
}
```

### 说明

Project 是所有数据组织的根对象，承载项目上下文与业务目标。

---

## 8.2 EvalRun

```json
{
  "run_id": "eval_101",
  "project_id": "proj_001",
  "benchmark_suite": "LIBERO_goal",
  "overall_success": 0.62,
  "task_breakdown": [],
  "failure_clusters": [],
  "artifacts": {
    "videos": [],
    "logs": [],
    "metrics_json": ""
  },
  "created_at": "2026-04-26T10:00:00Z"
}
```

### 说明

EvalRun 代表一次评测结果，是 Failure Map 的核心输入。

---

## 8.3 PatchPlan

```json
{
  "plan_id": "plan_201",
  "project_id": "proj_001",
  "based_on_eval_run": "eval_101",
  "root_causes": [
    "occlusion-heavy scenes underrepresented",
    "recovery trajectories missing"
  ],
  "target_data_spec": {
    "patch_episodes": 120,
    "teleop_corrections": 40,
    "lighting_variants": 1
  },
  "annotation_schema": "recovery_fine_grained_v1",
  "source_ratio": {
    "real": 0.7,
    "synthetic": 0.3
  },
  "training_strategy": "continual_learning",
  "verification_spec": "robustness_subset_eval",
  "expected_uplift": 0.10,
  "confidence": 0.73
}
```

### 说明

PatchPlan 是 Nvex 的核心结构化输出，也是 orchestration 的核心输入。

---

## 8.4 IterationRun

```json
{
  "iteration_id": "iter_301",
  "project_id": "proj_001",
  "plan_id": "plan_201",
  "based_on_checkpoint": "ckpt_v0.7",
  "status": "completed",
  "execution_backend": "AlphaBrain_CL",
  "output_checkpoint": "ckpt_v0.8",
  "result_summary": {
    "success_before": 0.62,
    "success_after": 0.74
  },
  "artifacts": {
    "logs": [],
    "videos": [],
    "eval_runs": ["eval_101", "eval_102"]
  }
}
```

### 说明

IterationRun 连接 plan、execution 与结果，是 Improvement Report 的核心对象。

---

## 8.5 ReusableAsset

```json
{
  "asset_id": "asset_401",
  "type": "recipe",
  "name": "occlusion_recovery_v1",
  "source_project": "proj_001",
  "reuse_count": 0,
  "linked_iteration": "iter_301",
  "description": "Patch recipe for cluttered occlusion recovery tasks"
}
```

### 说明

ReusableAsset 用于表达“平台记忆”，建议至少支持以下类型：

- recipe
- template
- failure_pattern
- verification_setup

---

## 9. 技术架构建议

整体采用三层架构：

## 9.1 Execution Layer

### 角色

由 AlphaBrain 承担模型训练、评测与环境执行能力。

### 典型能力

- baseline VLA train/eval
- continual learning
- world model eval
- benchmark suite execution
- eval artifacts generation

### 设计原则

- 不在 Nvex 中重复实现训练 runtime
- 通过标准化 job 接口消费 AlphaBrain 能力
- 尽可能将结果转为统一 artifact schema

### 本期 MVP 接入优先级

1. baseline VLA eval
2. continual learning
3. re-eval
4. world model artifact（可选增强）

---

## 9.2 Orchestration Layer

### 角色

由 Nvex 自身承担，负责 intelligence loop 的核心价值。

### 典型能力

- failure analysis
- gap diagnosis
- patch plan generation
- experiment orchestration
- memory accumulation

### 设计原则

- 所有高层决策都应结构化输出
- 支持规则驱动 + 手工配置 + 后续 agent 增强
- 优先建设可解释性，而不是追求“黑盒自动化”

### 本期 MVP 重点

- Failure Map 生成
- Patch Plan 生成
- Iteration 触发与跟踪
- Asset 沉淀

---

## 9.3 Presentation Layer

### 角色

负责向投资人、客户和内部团队呈现系统价值。

### 典型能力

- project overview
- failure dashboard
- action plan cards
- iteration timeline
- before/after report
- platform memory view

### 设计原则

- 页面组织围绕 failure-to-fix，而不是围绕底层系统结构
- 强调“下一步行动”和“结果变化”
- 避免把界面做成通用后台或参数面板

---

## 10. 里程碑规划

## 10.1 Milestone 1：Narrative MVP

### 目标

先把产品故事线讲清楚，即便部分数据和执行结果为预生成。

### 范围

- Project Overview
- Failure Map
- Patch Plan
- Improvement Report
- 基础 Platform Memory

### 要求

- 页面可点击演示
- 故事线完整
- 数据结构基本成型
- 可用于内部评审和融资彩排

### 不要求

- 实时训练
- 完整后端编排
- 全量真实 artifact 接入

---

## 10.2 Milestone 2：Executable MVP

### 目标

接入至少一条真实的 AlphaBrain 执行闭环。

### 范围

- 导入真实 eval result
- 生成 Patch Plan
- 触发一次真实 continual learning 或 fine-tune run
- 产出真实 before/after 评测结果

### 要求

- 至少 1 个真实 use case
- 至少 1 次真实 improvement
- artifact 可回放
- Iteration Runner 可展示状态变化

---

## 10.3 Milestone 3：Investor-grade Demo

### 目标

形成可用于正式投资人会面的稳定版本。

### 范围

- 视觉强化
- 代表性视频/rollout
- 更清晰的平台资产展示
- 更完整的口播路径
- 可选 world model predicted-vs-rollout 对比

### 要求

- 3–5 分钟可讲清
- 支持现场操作与录屏演示
- 不依赖现场训练成功
- 有稳定的备份数据与预生成结果

---

## 11. 成功指标

成功指标同时分为 **demo 理解度指标** 与 **产品执行指标**。

## 11.1 Demo 理解度指标

### 指标 A：定位理解度

在内部/投资人试讲后，受众是否能准确复述以下内容：

- Nvex 不是训练框架
- Nvex 不是标注工具
- Nvex 的核心价值是闭环 orchestration

**目标值**：80% 以上受众可在 1 分钟内准确复述

### 指标 B：故事线理解度

受众是否能清晰说出 demo 的主流程：

`failure → diagnosis → patch plan → execution → improvement → platform memory`

**目标值**：80% 以上受众可复述 4 个以上步骤

### 指标 C：平台化认知

受众是否能理解“每轮项目都在积累平台资产，而不仅是交付一个项目结果”。

**目标值**：多数受众在反馈中自发提及“compound / reuse / platform memory”等概念

---

## 11.2 产品执行指标

### 指标 D：页面闭环完整度

首页到 Improvement Report 的 关键路径可无中断演示。

**目标值**：100% 覆盖以下页面
Project Intake → Failure Map → Patch Plan → Iteration Runner → Improvement Report

### 指标 E：真实数据接入度

至少一条主流程中的核心指标来自真实 AlphaBrain artifact。

**目标值**：

- Eval artifacts：真实
- Before/after KPI：真实或半真实
- Plan recommendation：可规则生成

### 指标 F：执行闭环时长

从 plan 触发到结果展示的时间应可控。

**目标值**：

- 现场演示模式：< 30 秒（通过预生成结果）
- 可执行模式：< 1 小时 完成一次小规模 run

### 指标 G：模块可扩展性

数据模型和前端结构能够支持后续加入更多项目类型和 execution path。

**目标值**：

- 至少支持 1 个 benchmark 场景
- 至少支持 2 种 training/eval path 扩展可能性

---

## 12. 风险与缓解策略

## 12.1 风险：被误解为“开源套壳 + UI”

### 表现

投资人可能认为 Nvex 只是把 AlphaBrain 外面包了一层前端。

### 缓解

- 强调 Nvex 新增的是 orchestration logic，不是 execution runtime
- 在产品结构中单独展示 Patch Plan、Failure Ontology、Platform Memory
- 突出“结构化决策对象”，而不是只显示日志与图表

---

## 12.2 风险：被误解为“普通 dashboard”

### 表现

如果页面只展示 metrics，没有明确推荐动作，就会被认为只是分析面板。

### 缓解

- Failure Map 页面必须连接到 Patch Plan
- 每个主要页面都应有“下一步动作”
- Patch Plan 必须结构化，不能只是一段 LLM 文本

---

## 12.3 风险：执行链路不稳定

### 表现

现场 demo 依赖实时训练/评测可能失败或耗时过长。

### 缓解

- 准备预生成结果与真实 artifact 回放
- 现场采用“可执行模拟 + 结果回放”模式
- 将实时执行作为附加能力而非唯一依赖

---

## 12.4 风险：故事过大、实现过散

### 表现

团队同时尝试做太多模块，导致主闭环无法打通。

### 缓解

- 严格锁定主叙事为 Failure-to-Fix
- 第一版只做 1 个项目、1 条执行路径
- world model、复杂 memory、全流程数据工作台作为增强项后置

---

## 12.5 风险：缺乏结果说服力

### 表现

如果没有 before/after measurable improvement，demo 会被认为停留在概念层。

### 缓解

- 至少准备一个真实 improvement case
- 使用真实 benchmark artifacts
- Improvement Report 明确展示 success uplift 和 failure shrinkage

---

## 13. Open Questions

以下问题需要产品、设计、工程在进入开发前进一步确认：

1. **首个 demo 用哪个 benchmark/use case 最稳妥？**是否统一采用 LIBERO 单任务场景作为首个展示对象。
2. **Patch Plan 的第一版推荐逻辑采用什么方式？**规则库、模板库、LLM 辅助还是混合式。
3. **Iteration Runner 的现场展示策略是什么？**实时跑一部分，还是完全采用预生成结果回放。
4. **Platform Memory 的第一版是否需要真实 reuse 数据？**还是先用 curated assets 展示结构与价值。
5. **前端是否需要区分 investor mode 与 operator mode？**投资人模式偏 narrative，内部模式偏细节。
6. **AlphaBrain 接入方式如何定义？**脚本调用、服务封装还是异步 job queue。
7. **是否需要在第一版支持 world model 视频对比？**若加入，是否值得增加 setup 复杂度。
8. **哪些指标必须真实，哪些可以先半真实？**需要对现场可讲解性与工程复杂度做平衡。
9. **Improvement Report 的“expected uplift”与“actual uplift”如何区分展示？**防止让观众误解模型推荐与真实结果的边界。
10. **后续产品化时，Nvex 与现有 data workbench 的关系如何组织？**
    是作为上层 orchestrator，还是作为核心智能入口。

---

## 14. 附录：相关来源链接

### AlphaBrain

- https://github.com/AlphaBrainGroup/AlphaBrain
- https://alphabraingroup.github.io/AlphaBrain/
- https://alphabraingroup.github.io/AlphaBrain/quickstart/baselineVLA/
- https://alphabraingroup.github.io/AlphaBrain/quickstart/continual_learning/
- https://alphabraingroup.github.io/AlphaBrain/quickstart/world_model/

### Deck 相关页面

- https://www.genspark.ai/api/files/s/ftlQpvqL
- https://www.genspark.ai/api/files/s/ayeSDvnK
- https://www.genspark.ai/api/files/s/8x8NPBQ1
- https://www.genspark.ai/api/files/s/KJkc8OQx
