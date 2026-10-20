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
| **Karpathy autoresearch** | 三方分工：人写 `program.md` 定方向和约束；agent 改实验代码；固定的评估器（`prepare.py`）谁都不能动。"keep or revert" 的棘轮循环；每次尝试记一行 `results.tsv`；简单性原则；自主模式下不停下来问。 | 单文件单指标的局限 |
| **Karpathy LLM knowledge base** | 人把原始资料丢进 `raw/`，LLM 增量"编译"成互相链接的 markdown wiki（概念页、摘要、反向链接）；直接用 wiki 回答研究问题；问答结果写回 wiki；定期 lint 找矛盾、补缺失、发现新联系。 | Obsidian 等具体工具不是依赖（可选用来浏览） |
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
| `shinka/research/library.py` | 工作区资料库的记账：`raw/` 中哪些资料是新的/已编译（按哈希）、操作日志；编译本身由 agent 完成 |
| `shinka/research/cli.py` | `shinka_research` 命令：`init` / `hypothesis` / `scaffold` / `run` / `probe` / `skeptic` / `verdict` / `confirm` / `library` / `brief` / `status` |
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

已知局限（第一版接受）：候选代码仍可读写磁盘（可借此跨进程传递信息，或自己在磁盘上找到原始数据文件；真正封死需要文件系统沙箱）；不做撮合级仿真（滑点仅用 bps 近似）；单品种。

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
- **Lockbox**：`shinka_research confirm --data <源文件>` 在方向收尾时整个方向只开一次（需要 `--yes`，由人决定），一次性确认全部 `supported` 的假设：用的是结论里记录的那个实验、从结果库重新取出的那个程序，并校验评估器、开发数据和源文件哈希。笔记本里不存源文件路径（候选代码能读到笔记本），只存哈希。结果为 `confirmed` / `unconfirmed`，写入 `CONFIRMATION.md`。确认同样要求：Sharpe > 0 且高于买入持有、DSR（试验数 = 全方向）、子区间稳定；并在 lockbox 内再做一次前视检测。开过之后不能再开，继续研究要用新数据开新方向。不碰 lockbox 的检查全部通过、且方向里没有仍在增加试验的 `run`（按原则 4，试验数在开封时冻结；`probe` 一开始就计为一次试验）之后才开封：在锁内冻结每个假设的程序、评估器与开发数据的哈希、试验数、阈值和源文件哈希，并把 lockbox 记为已开，然后才运行候选程序；程序自身在 lockbox 上的失败（超时、出错退出、仓位无效、前视）只记类别并判 `unconfirmed`，不记它的输出（输出里可能带 lockbox 数字）；全部结果最后一起写入笔记本，后打分的程序读不到先打分的结果。中途中断时再执行同一条 `confirm --yes`：已开始打分的假设记为 `unconfirmed`（中断），不重跑，开封后被改动的程序也不运行，其余按冻结的规则补完；补完之前 `run`、`probe`、`skeptic`、`verdict`、`scaffold` 一律等待，同一时间只能有一个 `confirm`。
- **PBO**（Bailey, Borwein, López de Prado & Zhu, 2014 的 CSCV）：质疑者取该实验得分最高的至多 64 个不同程序，在开发数据上算逐 K 线收益矩阵，切成 16 块，枚举所有"一半样本内、一半样本外"的组合，统计"样本内第一名在样本外落到中位数及以下"的比例。PBO > 0.5（可调）时不判 `supported`。
- **前向测试**（工具之外的最终关卡）：`confirmed` 的策略到模拟盘上用研究开始后才产生的新数据跑一段时间。
- **不采用** Thresholdout / Ladder 这类可复用留存集：查询预算约随留存样本数平方增长，几百到几千根 K 线太少，且其独立样本假设不适用于有自相关的行情数据。

## 7.2 资料库与知识积累（P4）

外来资料的输入方式参照 Karpathy 的 LLM knowledge base：

```
<workspace>/library/
  raw/          人放资料：PDF、论文、网页剪藏、笔记；网址写进 links.md（`library add` 也行）
  wiki/         agent 编译的页面：paper / concept / finding，互相 [[链接]]，每条结论引用 raw 来源
  index.md      wiki 目录（agent 维护）
  log.md        编译、问答、lint、结论写回的流水
  manifest.json 已编译资料的哈希（命令行记账）
<workspace>/<direction>/   研究方向，同一工作区共用一个资料库
```

- **编译**：`library pending` 列出新的或改过的资料，agent 读完写成 wiki 页，再 `library compiled` 记账。agent 自己检索到的网络资料也先 `library add` 进 raw，保证所有结论都有来源可查。
- **结论写回**：每个结论出来后，agent 为该机制写/更新 finding 页（验证段证据、DSR 及试验数、PBO、消融结论），lockbox 数字在 `confirm` 之前不写。这就是跨方向的知识积累：下一个方向读 wiki 就知道"这个机制测过、结论是什么"。
- **Lint**：每轮检查 wiki 的矛盾、无来源的断言、过期结论、缺页的高频概念，结果记入 `log.md`。
- **简报**：`shinka_research brief` 汇总方向状态（假设与证据、试验数、花费与预算、lockbox 状态、未编译资料数、最近事件），agent 每轮选题前先读。
- **预算**：`init --budget-usd` 设定方向预算；`run` 前汇总各运行结果库里记录的模型费用，用完即拒绝运行。
- **实验流水**：方向目录下的 `results.tsv` 每个事件一行（scaffold / run / skeptic / verdict / confirm），对应 autoresearch 的 results.tsv。

## 7.3 研究层前端（只读）

上游 WebUI 只看一次进化运行（一个 `programs.sqlite`）。研究层缺的是方向级视图：假设与状态、结论证据、试验数 N、预算、lockbox、资料库、`results.tsv` 流水。

参考（2026-10 调研）：
- Karpathy autoresearch：`results.tsv` + 进度图（灰点丢弃、绿点保留、阶梯前沿线、保留点带描述）。照搬这个形式，补上它缺的：按假设分组、试验数门槛。
- Optuna Dashboard：视图少、单进程直接读库、每个 study 一份 markdown 笔记。照搬"少视图、直接读文件"。
- WorldQuant BRAIN：alpha 必须过一组硬门槛才算数。照搬成每条判定规则的 ✓/✗ 清单。
- MLflow 的阶段标签不带证据、W&B workspace 自动面板堆积、AI-Scientist-v2 的单次静态树 HTML：都是反例。状态旁必须挂证据；页面固定、不自动生成面板；跨实验在方向页汇总。
- 主流工具都不显式展示试验数、DSR、PBO，也没有"一次性不可逆"的留存集状态。这是本前端要补的。

决定：
- 入口 `shinka_research ui <工作区目录>`：`shinka/research/ui.py` 用标准库 HTTP 服务给出 `/api/workspace`（方向摘要与共享资料库）、`/api/snapshot?direction=<目录名>`（单方向快照）和构建好的前端，不改上游。方向以原生链接进入并返回工作区；每个实验的进化细节仍跳转到限定该方向的上游 viz_tree（隐藏 private 指标）。完整目录、身份与错误契约见 [workspace.md](workspace.md)。
- 前端在 `shinka/research/webui/app/`（Vite + React 19 + TypeScript），构建到 `webui/dist/`（不入库，`pnpm build` 生成）。照 Optuna Dashboard：Python 端只读文件、出 JSON，界面是单独打包的前端。
- 视觉方向 ops-spec（运维规格台，默认暗色、可切浅色），组件用现成库、不手搓：Astro UXDS 管「信号」（顶栏、状态符号、时间线、事件流），Carbon 管「内容」（表格、页签、术语提示、提示条、图表、代码）。两个库套同一份 token。选型过程与对比图见 `shinka/research/webui/design/changes/research-ui/component-selection/`。与 KatanaConsole 分开，不共用代码和组件。
- 只读。选中/放弃假设、出结论、confirm 仍走 CLI；confirm 永远不进 UI（不可逆操作不做成一次点击）。
- 页面：工作区（方向列表）、方向（头部状态条、假设表、前沿图、流水、报告）、资料库（index、wiki、`[[链接]]`、待编译、log）。
- 前沿图：方向内所有试验的样本内分数按时间排开，叠加"N 个纯噪声试验的期望最大值"门槛线（与质疑者同一近似：方差取试验样本内分数）。试得越多线越高，直观表达多重检验代价。
- 可见性纪律与 agent 相同：未出结论的假设只显示样本内指标，验证段证据在 verdict 之后才显示；lockbox 数字只在 `confirm` 之后显示。过滤在服务端做（`ui.snapshot`），浏览器拿不到不该显示的数；质疑者事件的说明里带验证段数字，未出结论时同样隐去。
- 视觉规范按 design-pipeline 走：方向与组件选型由所有者定，DESIGN.md 与设计系统包从 token 生成。

## 8. 分阶段路线（渐进式，每一步都端到端可用）

见 [roadmap.md](roadmap.md)。

## 9. 不做的事（当前阶段）

- 不做论文 / LaTeX 产线（只出 Markdown 研究报告）。
- 不做多品种组合、撮合级回测、实时数据。
- 不做通用领域（下一阶段：把 `quant/` 换成其他领域的评估框架即可，笔记本、质疑者框架、skill 结构保持不变）。
- 不改上游 Shinka 核心。
