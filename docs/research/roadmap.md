# Shinka Research：路线图

每个阶段结束时系统都必须端到端可用。阶段内的每一项对应一个 issue / PR。

## P0 · 设计与工程规范
- [x] 设计文档（[design.md](design.md)）
- [x] 路线图
- [x] fork 级 agent 规则（`AGENTS.md` / `CLAUDE.md`）
- [x] research 专用 CI（只跑 research 相关的 lint 和测试，带超时）

## P1 · 最小闭环：一个假设手动走通
- [x] `shinka/research/notebook.py`：方向 / 假设 / 实验 / 结论的数据模型与文件存储
- [x] `shinka/research/quant.py`：数据加载、样本内多折 + 样本外留存、成本模型、绩效指标、前视检测、`run_shinka_eval` 适配
- [x] `shinka/research/scaffold.py`：假设 → Shinka 任务目录，评估器与数据哈希记录与校验
- [x] `examples/quant_research/`：可直接运行的示例方向（合成数据），`shinka_run` 能跑
- [x] `shinka_research` CLI：`init` / `hypothesis` / `scaffold` / `run` / `status`

完成标准：在示例方向上 `init → hypothesis → scaffold → run` 能跑通，笔记本正确记录。

## P2 · 质疑者与结论
- [x] `shinka/research/skeptic.py`：DSR、子区间稳定性、样本内外衰减、累计试验数
- [x] CLI：`skeptic` / `verdict`，结论写回笔记本
- [x] 评估器篡改检测接入 `run` / `skeptic`

完成标准：对一次进化结果给出带证据的结论并入册。

## P3 · 研究循环 skill
- [x] `skills/shinka-research/SKILL.md`：方向 → 选题（文献检索）→ 假设树 → 人选择 → 脚手架 → 分阶段进化 → 质疑 → 结论 → `REPORT.md`
- [x] `program.md` 模板（方向、约束、预算、数据、阈值）
- [x] 假设模板（可证伪陈述、测试方法、判据、风险）与报告模板
- [x] `probe` 命令：用冻结评估器打分、只显示样本内指标（选题/消融时使用），每次计入试验数

完成标准：agent 只读 skill 就能从一个方向跑到结论报告。

## P3.5 · 留存集复用治理
- [x] lockbox：首次 scaffold 时物理切出数据尾段，`confirm` 整个方向只开一次
- [x] 质疑者加 PBO（CSCV），并纳入 `supported` 判据
- [x] skill 加入 lockbox 规则与前向测试关卡

## P4 · 持续研究与资料库
- [x] 工作区资料库（Karpathy LLM knowledge base）：`raw/` → agent 编译的 `wiki/`，`library add/pending/compiled/log`
- [x] 下一轮选题读取 `brief`（假设与证据、试验数、花费、lockbox、未编译资料、最近事件）
- [x] 方向级预算（`init --budget-usd`，`run` 超预算拒绝）与 `results.tsv` 实验流水
- [x] 跨方向知识沉淀：结论写回 wiki 的 finding 页，每轮 lint
- [x] skill：自主模式、简单性原则、编译/问答/写回/lint 步骤

## P5 · 研究层前端（只读）
- [x] 视觉方向与组件选型（design-pipeline）：ops-spec，Astro UXDS × Carbon，所有者确认
- [x] 设计系统包与 DESIGN.md 改到选定方向（tokens.json 是唯一值来源，界面直接读它）
- [x] `shinka_research ui <工作区目录>`：工作区方向列表、方向页和共享资料库；方向支持深链接、刷新与返回（[契约](workspace.md)，GitHub #12）
- [x] 可见性测试：未出结论不露验证段证据（含质疑者事件说明），未 confirm 不露 lockbox 数字
- [x] 页面按研究进度换内容：试验图（噪声线）、实验时间线、资料库页，跳转上游 viz_tree

完成标准：人不用 CLI 就能看清一个方向的全部状态，且看不到 agent 也不该看到的数字。

## 之后 · 通用化
- [ ] 把 `quant/` 抽象为可替换的"领域评估框架"接口，接入第二个领域验证
