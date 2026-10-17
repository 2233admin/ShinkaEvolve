# Shinka Research：设计文档

> 状态：P0（设计定稿）· 范围：量化研究优先，通用化是下一阶段
> 本目录是 fork（2233admin/ShinkaEvolve）自有的内部文档，用中文；对外文档后续再出英文版。

## 1. 要解决的问题

ShinkaEvolve 能把"一个程序 + 一个评估器"进化到更高分，但它回答的是"怎么更高分"，不是"这个想法对不对"。
我们要的是：**人给一个研究方向，系统提出可证伪的假设、设计实验、动手尝试，最后给出有证据的结论**，并且结论会积累下来指导下一轮。

一句话：在 Shinka 之上加一层"研究"，把单次优化变成持续的研究循环。

## 2. 参考的成熟做法

| 来源 | 拿什么 | 不拿什么 |
|---|---|---|
| **Karpathy autoresearch** | 三方分工：人写 `program.md` 定方向和约束；agent 改实验代码；固定的评估器（`prepare.py`）谁都不能动。"keep or revert" 的棘轮循环。 | 单文件单指标的局限 |
| **AI-Scientist-v2** | 选题时结合文献查新；结构化的想法格式（假设 / 相关工作 / 实验 / 风险）；实验分阶段推进：`initial_implementation → baseline_tuning → creative_research → ablation_studies`；评审者挑毛病。 | **不搬代码**（其许可证为 RAIL 衍生，与 Apache-2.0 混用有问题）；不要论文/LaTeX 产线；不要它的 BFTS 搜索（与 Shinka 重叠） |
| **DeepEvolve** | "深度调研 + 进化"互相反馈：调研给方向，进化结果回流修正调研。消融显示无调研的纯进化很快停滞。 | 具体实现 |
| **ShinkaEvolve（本仓库）** | 整个进化引擎原样使用：`initial.py` 的 EVOLVE-BLOCK、`evaluate.py`、`public`/`private` 指标、岛屿、预算上限、SQLite 结果库、agent skills。 | — |
| **López de Prado 等的回测统计** | Deflated Sharpe Ratio（把试验次数算进显著性）、样本外留存、前视偏差检测。 | — |

## 3. 核心原则（不可妥协）

1. **研究的单位是假设，不是程序。** 每个假设必须可证伪：写明怎么测、什么结果算成立、什么算推翻。产出是结论（成立 / 推翻 / 证据不足）+ 证据，不是一个分数。
2. **评估器归人，不归 LLM。** 数据切分、成本假设、评估逻辑在实验开始前写死并记录哈希；进化只能改 EVOLVE-BLOCK。任何评估器改动都意味着开一个新实验。
3. **LLM 看不到留存集。** 样本内指标进 `public`（参与选择），样本外指标进 `private`（只给质疑者看）。
4. **所有试验都计数。** 一个方向下跑过的变体总数进入多重检验校正（DSR）。失败和证据不足的结论同样入册。
5. **人只在便宜的节点介入。** 选假设时一次，看结论时一次，其余自动。
6. **每层都有预算。** 方向、假设、单次进化各自有成本/代数上限。
7. **只加不改上游。** 新代码放在 `shinka/research/`、`skills/shinka-research/`、`examples/quant_*`、`docs/research/`；除非确有必要且单独提 PR，不改上游核心代码，方便持续同步上游。

## 4. 架构

```
人：方向（program.md）
        │
        ▼
┌──────────────────────────────┐
│ ① 选题（agent + 文献检索）       │  产出：假设树（3–8 个可证伪假设）
└──────────────┬───────────────┘
               │  ◀── 人选择要做的假设
               ▼
┌──────────────────────────────┐
│ ② 实验设计（脚手架）             │  每个假设 → 一个 Shinka 任务目录
│   initial.py + 固定 evaluate.py │  评估器哈希、数据切分写入笔记本
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ③ 尝试（ShinkaEvolve 原样）      │  按 v2 的四个阶段推进；public 指标驱动选择
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ④ 质疑者                       │  前视检测 / 样本外 / 子区间稳定性 /
│                               │  DSR（试验次数 = 该方向累计变体数）
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ⑤ 结论 → 研究笔记本              │  成立 / 推翻 / 证据不足 + 证据
└──────────────┬───────────────┘
               │  ◀── 人审阅结论
               └──────▶ 下一轮选题先读笔记本
```

### 职责划分

- **确定性的部分用 Python（`shinka/research/`）**：笔记本读写、任务脚手架与评估器哈希、量化评估框架、前视检测、质疑者统计、调用 Shinka、读结果库。可测试、可复现。
- **需要判断的部分交给 agent（`skills/shinka-research/`）**：选题与文献检索、假设撰写、阶段推进决策、结论解读、报告。这就是 autoresearch 里的 `program.md` 角色。
- **编排者**：Claude Code（或任意 coding agent）读 skill 后驱动整个循环，通过 `shinka_research` CLI 调用确定性部分。

### 模块

| 模块 | 职责 |
|---|---|
| `shinka/research/notebook.py` | 研究笔记本的数据模型与文件存储（方向 → 假设 → 实验 → 结论） |
| `shinka/research/scaffold.py` | 由假设生成 Shinka 任务目录；评估器是单个自包含的 `evaluate.py`（配置内嵌，因为 Shinka 会把它复制到结果目录再运行）；记录评估器与数据的哈希并校验 |
| `shinka/research/quant.py` | 量化评估框架：数据加载、样本内/样本外切分、样本内多折、成本模型、绩效指标、前视检测；产出符合 `run_shinka_eval` 约定的 `public`/`private` 指标 |
| `shinka/research/rundb.py` | 只读访问 Shinka 结果库（评估次数，排除岛屿迁移副本） |
| `shinka/research/skeptic.py` | DSR、PBO、子区间稳定性、样本内外衰减、基线；lockbox 一次性确认 |
| `shinka/research/cli.py` | `shinka_research` 命令：`init` / `hypothesis` / `scaffold` / `run` / `probe` / `skeptic` / `verdict` / `confirm` / `status` |
| `skills/shinka-research/` | 研究循环 program：每一步做什么、何时停下问人、产出格式 |

## 5. 研究笔记本（数据模型）

纯文件存储，git 友好、agent 可直接阅读：

```
research/<direction-slug>/
  program.md                 # 人写：方向、约束、预算、数据说明（autoresearch 的 program.md）
  notebook.json              # 机器可读索引：假设、状态、实验、结论、累计试验数
  hypotheses/<hid>.md        # 每个假设一份：可证伪陈述、测试方法、成立/推翻判据、风险
  experiments/<hid>/<eid>/   # Shinka 任务目录 + 结果目录 + 质疑报告
  REPORT.md                  # 该方向的阶段性研究报告（agent 生成）
```

假设状态机：`proposed → selected → running → challenged → {supported | refuted | inconclusive}`，也可 `proposed → dropped`。

## 6. 量化评估框架（第一版）

- **输入**：用户自备的 K 线数据（CSV/Parquet，带时间戳的 OHLCV）。不接数据源。
- **策略接口**（EVOLVE-BLOCK 内实现）：`generate_positions(bars: DataFrame) -> Series`，输出每根 K 线收盘后的目标仓位（-1…1）。框架负责按下一根 K 线收益结算，策略不接触收益计算。
- **切分（三段）**：方向的第一个实验 scaffold 时，把数据源的最后 20% 切成 lockbox（可配置），只把前 80% 写成 `data/dev.csv` 交给评估器；lockbox 段在物理上不进入任何实验。开发数据再按时间切成样本内（前 70%，再分 k 个连续折）和验证段（后 30%）。默认整体约为 56% / 24% / 20%。同一方向只能用同一份数据，换数据就开新方向。
- **成本**：按换手 × 每边成本（bps）扣除，默认值写在 `program.md` 里并计入评估器哈希。
- **public**（驱动进化）：样本内各折 Sharpe 的均值作为 `combined_score`，并报告各折 Sharpe 的标准差、最大回撤、换手、年化收益。
- **private**（只给质疑者）：样本外 Sharpe、收益、回撤；逐 K 线收益序列存入 `extra_data` 供 DSR 使用。
- **前视检测**：随机取若干截断点 t，比较"用全部数据算出的仓位"与"只用 t 之前数据算出的仓位"在 t 处是否一致；不一致即判为错误程序（`correct=False`）。实现上直接复用 `run_shinka_eval` 的多次运行：第 0 次传完整数据，第 1…k 次传截断数据，在聚合函数里比对。

- **执行隔离**：评估进程从不导入候选代码。每个数据视图（完整 + 各截断点）都在独立的新解释器（`python -I -m shinka.research.child`）里运行，只把浮点数组以 `.npy`（禁用 pickle）传回。这样候选代码无法篡改评分函数，也无法在不同视图之间缓存"未来"结果来骗过前视检测。

已知局限（第一版接受）：候选代码仍可读写磁盘（可借此跨进程传递信息）；不做撮合级仿真（滑点仅用 bps 近似）；单品种。

## 7. 质疑者（第一版）

- **DSR**：以样本外逐 K 线收益计算 Sharpe，试验次数 N = 该方向累计评估过的变体数，按 Bailey & López de Prado（2014）计算 Deflated Sharpe Ratio。试验间 Sharpe 的方差用该方向所有正确程序的样本内得分估计（近似：没有为每个试验单独算样本外 Sharpe）。
- **基线**：同一留存期上的买入持有、种子程序（`initial.py`）的样本外 Sharpe，写进证据供解读。
- **子区间稳定性**：样本外切成若干段，统计 Sharpe 为正的段数占比。
- **样本内外衰减**：样本外 Sharpe / 样本内 Sharpe。
- **结论规则**（默认阈值写在 `program.md`，可调）：
  - `supported`：DSR ≥ 0.95，子区间正比例 ≥ 2/3，且样本外 Sharpe 高于买入持有；
  - `refuted`：样本外 Sharpe ≤ 0（预先定义的反驳判据）；
  - 其余（包括未能胜过买入持有、DSR 不够）：`inconclusive`——没证明有效不等于证明无效。
  阈值只是默认值，agent 可在报告里给出不同解读，但笔记本里记录的是规则判定结果。

## 7.1 留存集复用：验证段 + lockbox + PBO + 前向测试

问题：每一轮的结论（来自验证段）会进入笔记本，下一轮选题会读到它，所以验证段会被自适应复用。DSR 按方向累计试验数做了部分校正，但校正不了"根据结论调整方向"。

做法（参照 train / validation / lockbox 的成熟实践，以及量化研究里"最终段只开一次、看过再修就降级为开发数据"的规则）：

- **验证段**：逐假设的质疑者和结论只用它，可多轮复用，结论算"开发级证据"（`supported`）。
- **Lockbox**：`shinka_research confirm` 在方向收尾时整个方向只开一次（需要 `--yes`，由人决定），对所有 `supported` 的假设做确认，结果为 `confirmed` / `unconfirmed`，写入 `CONFIRMATION.md`。确认同样要求：Sharpe > 0 且高于买入持有、DSR（试验数 = 全方向）、子区间稳定；并在 lockbox 内再做一次前视检测。开过之后不能再开，继续研究要用新数据开新方向。
- **PBO**（Bailey, Borwein, López de Prado & Zhu, 2014 的 CSCV）：质疑者取该实验得分最高的至多 64 个不同程序，在开发数据上算逐 K 线收益矩阵，切成 16 块，枚举所有"一半样本内、一半样本外"的组合，统计"样本内第一名在样本外落到中位数及以下"的比例。PBO > 0.5（可调）时不判 `supported`。
- **前向测试**（工具之外的最终关卡）：`confirmed` 的策略到模拟盘上用研究开始后才产生的新数据跑一段时间。
- **不采用** Thresholdout / Ladder 这类可复用留存集：查询预算约随留存样本数平方增长，几百到几千根 K 线太少，且其独立样本假设不适用于有自相关的行情数据。

## 8. 分阶段路线（渐进式，每一步都端到端可用）

见 [roadmap.md](roadmap.md)。

## 9. 不做的事（当前阶段）

- 不做论文 / LaTeX 产线（只出 Markdown 研究报告）。
- 不做多品种组合、撮合级回测、实时数据。
- 不做通用领域（下一阶段：把 `quant/` 换成其他领域的评估框架即可，笔记本、质疑者框架、skill 结构保持不变）。
- 不改上游 Shinka 核心。
