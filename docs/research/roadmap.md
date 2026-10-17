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
- [ ] `skills/shinka-research/SKILL.md`：方向 → 选题（文献检索）→ 假设树 → 人选择 → 脚手架 → 分阶段进化 → 质疑 → 结论 → `REPORT.md`
- [ ] `program.md` 模板（方向、约束、预算、数据、阈值）
- [ ] 假设模板（可证伪陈述、测试方法、判据、风险）

完成标准：agent 只读 skill 就能从一个方向跑到结论报告。

## P4 · 持续研究
- [ ] 下一轮选题自动读取笔记本（避免重复、在已有结论上深挖）
- [ ] 方向级预算与进度汇总
- [ ] 跨方向的知识沉淀

## 之后 · 通用化
- [ ] 把 `quant/` 抽象为可替换的"领域评估框架"接口，接入第二个领域验证
