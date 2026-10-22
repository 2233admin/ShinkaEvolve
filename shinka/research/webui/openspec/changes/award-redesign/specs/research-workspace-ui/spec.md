## Purpose

研究方向总览、方向与假设研究报告、共享资料库的只读呈现：研究者一屏读懂每个方向的状态与证据，并能从任一结论连续进入对应实验的分析。

## ADDED Requirements

### Requirement: 首页列出全部研究方向及其状态与证据

首页 SHALL 列出工作区直接子目录中的每个有效研究方向，并为每个方向同时显示：标题与目录 ID、假设状态分布、试验数（各运行库中非岛副本的已评估程序数加试探次数，即质疑检验计 DSR 的试验；运行中的实验随运行计入，不等运行结束才写入的 notebook 计数）、已评估程序数与其中评估失败的程序数、当前最佳样本内记录与当时的噪声线、花费与预算、封存段状态、最近活动时间。花费包含费用未知的运行时 MUST 以 `≥` 标为下限并注明费用未知的运行数。无法读取的方向 MUST 显示为不可进入的错误条目，不得伪造零花费、零试验或封存状态。首页 MUST NOT 为取得这些字段逐方向请求 `/api/snapshot`。

#### Scenario: 演示工作区的四种方向状态

- **WHEN** 打开演示工作区首页
- **THEN** 四个方向分别呈现进行中、已结论且封存段已开、新建无试验、超预算这四种可区分的状态，每个方向的假设状态分布、最佳记录与噪声线、花费与预算可在首屏读出
- **AND** 页面加载期间没有 `/api/snapshot` 请求

#### Scenario: 不可读方向

- **WHEN** 某方向的 notebook 损坏
- **THEN** 该方向以错误条目出现、不可进入，且不参与任何汇总数字

#### Scenario: 空工作区

- **WHEN** 工作区没有任何方向
- **THEN** 首页显示空状态，说明研究由对话或 CLI 发起，并仍可进入全部运行与资料库

### Requirement: 工作区接口提供方向证据摘要

`GET /api/workspace` 的每个成功方向条目 SHALL 增加只读字段 `unknown_cost_runs`（整数，方向内费用未知的实验运行数，口径见 `run-analysis-ui` › 运行费用只有一个口径）与 `evidence`：`hypothesis_status`（各状态计数）、`best_score`（全方向样本内最佳分数，无有效分数时为 `null`）、`noise_bar`（与方向试验图最后一点相同算法的噪声线，无试验时为 `null`）、`evaluated_programs`（方向各运行库中非岛副本的已评估程序数，即快照 `trial_points` 的条数）、`failed_programs`（其中评估失败的程序数，即快照中分数为空的点）、`last_activity`（最近事件或试验时间，无活动时为 `null`）。`spend_usd` 保持为已记录费用之和，`unknown_cost_runs` 大于 0 时它是下限；`trials` 是 `evaluated_programs` 加方向全部试探次数，与质疑检验计 DSR 的试验相同（空方向为 0，质疑检验在那里取下限 1）。`/api/snapshot` 的 `trials` 与每个实验的 `n_evaluated`、`best_is_score` 同样按运行库计数，同一响应中的计数、最佳分数与试验点来自对每个运行库的同一次读取。`/api/snapshot` 的方向同样带 `unknown_cost_runs`，每个实验带 `spend_known`。响应另含首页动态用的 `events`：各可读方向最近的事件按时间由新到旧合并（至多 30 条），每条带方向 ID，说明的隐藏规则与 `/api/snapshot` 相同；不可读方向不贡献事件。摘要 MUST NOT 包含验证段或封存段数字，也 MUST NOT 放宽现有路径限制。

#### Scenario: 与方向页数字一致

- **WHEN** 请求 `/api/workspace` 与同一方向的 `/api/snapshot`
- **THEN** `best_score` 等于快照中有效分数的最大值，`noise_bar` 等于快照 `trial_points` 最后一点的 `bar`，`evaluated_programs` 等于 `trial_points` 的条数，`failed_programs` 等于其中分数为空的条数，两处的 `unknown_cost_runs` 相等

#### Scenario: 运行中的实验按运行库计数

- **WHEN** 一个实验正在运行，notebook 记的已评估数仍是 0，它的运行库已有 150 个程序，方向另有 5 次试探
- **THEN** 该实验的 `n_evaluated` 为 150，两个接口的 `trials` 都等于全方向已评估程序数加 5，并与质疑检验计 DSR 的试验数相同
- **AND** 首页、方向页页首、假设表与实验卡片显示同一组数字

#### Scenario: 无试验方向

- **WHEN** 方向只有提出的假设、没有任何试验
- **THEN** `best_score`、`noise_bar` 与 `last_activity` 为 `null`，`evaluated_programs`、`failed_programs` 与 `unknown_cost_runs` 为 0，`spend_usd` 为 0

#### Scenario: 费用未知的运行使花费成为下限

- **WHEN** 方向的一个实验运行含没有价目的 headless 调用
- **THEN** 两个接口的 `unknown_cost_runs` 为 1，快照中该实验的 `spend_known` 为 false，`spend_usd` 是其余已记录费用之和
- **AND** 首页的该方向花费显示为 `≥` 下限，并注明 1 个运行费用未知

### Requirement: 方向研究报告按论证顺序呈现

方向页 SHALL 以“主张 → 试验与噪声线 → 检查台账 → 判词”的阅读顺序呈现每个假设。试验图的横轴 MUST 使用整数试验序号，噪声线 MUST 由图表本身绘制而非渲染后修改样式。验证段数字 MUST 只在出结论后显示，封存段数字 MUST 只在确认后显示；试探计入试验数但不画点。

#### Scenario: 出结论前隐藏验证段

- **WHEN** 假设尚无判词
- **THEN** 报告显示样本内证据与噪声线，验证段与封存段数字均不出现

#### Scenario: 整数试验序号

- **WHEN** 方向有 3 次试验
- **THEN** 试验图横轴刻度只出现整数序号

### Requirement: 从结论两步内进入实验分析并可返回

研究报告中的每个实验 SHALL 提供进入其分析的入口；从假设出发 MUST 在两次操作内到达该实验谱系中的最佳程序代码，返回时 MUST 恢复原方向与假设。

#### Scenario: 往返保留上下文

- **WHEN** 从假设进入实验分析、选中最佳程序查看代码后返回
- **THEN** 页面回到原方向与原假设，滚动位置与展开状态保持

### Requirement: 共享资料库只读阅读

资料库 SHALL 显示索引、日志、待处理条目与全部 wiki 页面；页面正文 MUST 以净化后的 Markdown 呈现。

#### Scenario: 恶意标记不执行

- **WHEN** wiki 页面含 `<svg onload=…>` 等标记
- **THEN** 标记按文本显示或被移除，不执行任何脚本

### Requirement: 研究页只读

研究页 MUST NOT 提供任何写入研究状态的控件，页面只发出 GET 请求。

#### Scenario: 无写请求

- **WHEN** 浏览首页、方向页与资料库的全部交互
- **THEN** 网络记录中没有 POST、PUT、PATCH 或 DELETE 请求

### Requirement: 研究页的响应式、主题与可读性

研究页 SHALL 在 1920、1440、1024 宽及深浅两套主题下无文字截断、元素遮挡与外层横向溢出；中文排版遵守项目 CJK 规则；所有交互键盘可达、焦点可见，正文对比度不低于 4.5。窄于 1024 的宽度不做专门设计与验收（所有者 2026-10-22：手机用不到）。

#### Scenario: 窄桌面宽度

- **WHEN** 以 1024 宽打开方向页
- **THEN** 导航栏收为带文字的窄栏，报告内容无截断，表格与图表在自身区域内滚动，外层无横向滚动

### Requirement: 中英切换

研究页 SHALL 提供中文与英文两种界面语言，切换覆盖导航、标题、按钮、表头、状态词、空与错误状态、日期说明与图表标注；所选语言 MUST 写入本地存储并在刷新后恢复。研究内容（方向与假设标题、notebook 正文、日志、wiki 页面）MUST 保持原文，不做翻译。Sharpe、DSR、PBO 在两种语言下都用英文缩写。

#### Scenario: 切换后刷新

- **WHEN** 在首页切换为英文后刷新页面
- **THEN** 界面仍为英文，状态词显示为 supported、refuted 等英文词，方向标题保持原文

#### Scenario: 空与错误状态

- **WHEN** 以英文打开空工作区与含不可读方向的工作区
- **THEN** 空状态说明与错误条目都以英文呈现，不出现未翻译的中文界面文字

### Requirement: 默认深色并可切换

研究页 SHALL 默认深色主题，并提供浅色主题；所选主题 MUST 写入本地存储并在刷新后恢复；两套主题都满足对比度要求。

#### Scenario: 首次打开

- **WHEN** 本地存储中没有主题记录时打开首页
- **THEN** 页面为深色主题

### Requirement: 试验点可打开对应程序

`/api/snapshot` 的每个 `trial_points` 条目 SHALL 增加只读的程序编号；方向报告中点按一个试验点，或以键盘聚焦试验图、用方向键选中试验点后按 Enter，MUST 打开该实验的分析并在检查器中显示该程序。未出结论的假设仍只显示样本内数字。

#### Scenario: 从试验点到检查器

- **WHEN** 在方向报告的试验图中选中一个试验点
- **THEN** 进入对应实验的分析，检查器显示该程序，返回时回到原方向与原假设
