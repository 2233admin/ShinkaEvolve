## Purpose

研究框架与领域无关：任何能作为程序反复执行的实验，都按同一套规则登记、冻结、隔离执行、质疑与判定；领域只提供数据准备、模板、用词与所支持的能力；检查器由 Agent 起草、所有者批准后冻结；判定规则由核心提供。

## ADDED Requirements

### Requirement: 方向的领域写定后不能改

`init` SHALL 写定方向的领域（缺省 `quant`）；方向中每个实验的计划 MUST 与方向的领域一致。没有领域字段的现有方向按 `quant` 读取。未登记的领域 MUST 明确报错，工作台只让该方向显示错误。

#### Scenario: 混入别的领域

- **WHEN** 在一个 `quant` 方向里为 `math` 领域脚手架实验
- **THEN** 命令拒绝，方向的试验数与方差估计不受影响

#### Scenario: 未知领域

- **WHEN** 一个方向的笔记本写着未登记的领域
- **THEN** 命令报出未知领域；工作台其余方向照常显示

### Requirement: 实验计划由方向批准与轮次授权共同冻结

有计划的实验 SHALL 冻结领域、检查器及其固定资源与假设数据的每个文件的哈希、生成的评估器包装器、任务提示、判定规则种类与版本及阈值、主张类型（证书型或经验型）、运行环境哈希、预算、种子得分、提案、批准与授权编号。所有者的认可分两步共同构成：方向批准（`v` 编号）冻结检查器、固定资源与运行环境，轮次授权（`g` 编号）冻结该假设的实验计划；两个编号都写进实验计划。经验型主张的计划 MUST 带隐藏实例或留存数据，并带最终评估。`run`、`probe`、`skeptic` 与 `verdict` MUST 在计划中任一文件的哈希不符时拒绝执行；判定 MUST 使用计划中的规则与阈值。没有计划的旧实验标为未冻结，照旧接受命令行阈值。

#### Scenario: 经验型主张没有隐藏数据

- **WHEN** 一份声明为经验型的计划既没有隐藏实例也没有留存数据
- **THEN** `hypothesis plan` 拒绝该计划

#### Scenario: 检查器被改动

- **WHEN** 实验冻结后，检查器目录中的一个文件被修改
- **THEN** `run` 与 `skeptic` 拒绝执行，并指出哪个文件与计划不符

#### Scenario: 有计划时不接受命令行阈值

- **WHEN** 对一个有计划的实验执行 `skeptic` 并传入不同的阈值
- **THEN** 命令拒绝该阈值，结论按计划给出

#### Scenario: 旧实验照旧

- **WHEN** 对一个迁移前的实验执行 `skeptic` 并按 `program.md` 传入阈值
- **THEN** 结论与迁移前相同，工作台标明该实验未冻结

### Requirement: 候选与检查器分进程执行

核心 SHALL 为每个计划生成评估器包装器：候选程序在独立子进程中运行，按约定只写出产物文件；候选退出后，冻结的检查器在另一个进程中只读该产物，给出 `correct`、`combined_score` 与指标。检查器 MUST NOT 导入候选代码，得分 MUST 由检查器从产物算出。确认 SHALL 先终止候选的整个进程树，把检查器与产物复制进以哈希命名的只读目录并核对哈希，只从该副本加载，产物只用不执行代码的格式解析，并在新的进程中运行，不运行候选代码。子进程不是操作系统沙箱，候选仍能读写文件与联网；经验型结论，以及做不到上述确认条件的任何结论，MUST 在工作台标明“未沙箱”。

#### Scenario: 候选篡改校验

- **WHEN** 一个候选程序试图替换检查器的函数或在产物里自报更高得分
- **THEN** 检查器的得分不受影响，篡改不改变结论

#### Scenario: 运行中改动检查器文件

- **WHEN** 候选运行期间改写了检查器目录中的文件
- **THEN** 确认因哈希不符被拒绝，并指出该文件

### Requirement: 外部已知结果随计划冻结

每个假设的实验计划 SHALL 带 `prior_art`，状态为 `found`（外部已知的最好结果、来源与检索词）、`none_found`（检索词与“没找到”）或 `unavailable`（没有联网条件、未检索）之一；`unavailable` MUST NOT 记成 `none_found`。`found` 时 SHALL 同时冻结比较方向与绝对容差。它随实验计划冻结，并在所有者授权该假设的轮次时与检查器同屏显示。`reach_target` 的结论与报告 MUST 按冻结的方向与容差写明结果差于已知、在容差内（复现已知）还是优于已知；没有 `found` 时写“没有外部参照”。存在外部已知的正确或错误产物时，方向提案或实验计划 SHALL 附上并记录检查器对它们的判定。

#### Scenario: 缺少外部参照状态

- **WHEN** 实验计划没有 `prior_art`
- **THEN** `hypothesis plan` 拒绝该计划

#### Scenario: 离线启动

- **WHEN** Agent 没有联网条件，实验计划记 `prior_art` 为 `unavailable`
- **THEN** 计划照常写入，授权页与报告标明“没有外部参照”

#### Scenario: 三档边界

- **WHEN** 较高为好、已知结果为 K、容差为 t，最好产物的得分分别为 K−2t、K+t/2、K+2t
- **THEN** 报告依次写“差于已知”“复现已知”“优于已知”

### Requirement: 资料来源由框架记录

`library add` SHALL 在只追加的资料库日志中按 Agent 执行环境标记记下资料由 Agent 还是人加入，不在 `raw/` 下新增文件；这个标记是协作标记，MUST NOT 被当作可信的出处证明。`brief` SHALL 显示原始资料数（按来源区分）、最近一次加入外部资料的时间与 wiki 中结论页的占比；这些只显示，不阻止 `hypothesis add`。

#### Scenario: Agent 加入网页资料

- **WHEN** 带 Agent 标记的环境执行 `library add <url>`
- **THEN** 日志记为 Agent 加入，`library pending` 的输出与以前的规则一致

### Requirement: 判定规则由核心提供

判定 SHALL 只使用核心提供的、带版本的规则种类（至少 `dsr_pbo`、`reach_target`、`counterexample`、`holdout_vs_baseline`）与随实验计划冻结的参数；领域与 Agent 只选种类、给参数。推翻标准在许愿后、所有者授权该假设的轮次前的对话中约定，落在规则种类与参数上；`reach_target` 的实验计划 MUST 带 `on_budget_exhausted`（`refuted` 或 `insufficient`），缺少时 `hypothesis plan` MUST 拒绝。对话原文 SHALL 作为 `rationale` 存进实验计划，只作审计，MUST NOT 参与判定。`reach_target` 的目标 MUST 高于 `hypothesis plan` 用已批准检查器测得的种子得分。Agent 执行环境中 `verdict --label` MUST NOT 把结论往“成立”方向改。

#### Scenario: 预算用完未达到目标

- **WHEN** `reach_target` 计划的 `on_budget_exhausted` 为 `refuted`，预算用完时最好得分低于目标
- **THEN** 假设判为推翻，结论写明推翻的是“该方法在该预算内能达到目标”；同样情况下 `insufficient` 判为证据不足

#### Scenario: 缺少预算用完时的判法

- **WHEN** `reach_target` 实验计划没有 `on_budget_exhausted`
- **THEN** `hypothesis plan` 拒绝该计划

#### Scenario: 目标不高于种子

- **WHEN** 实验计划的目标值不高于种子程序的得分
- **THEN** `hypothesis plan` 拒绝该计划

#### Scenario: 反例

- **WHEN** `counterexample` 规则下检查器认可一个反例产物
- **THEN** 假设判为推翻；没有找到反例时判为证据不足

### Requirement: 检查器按方向批准，假设的实验计划随轮次授权冻结

检查器 SHALL 每个方向批准一次：方向提案只包含可复用的检查器及其固定资源和运行环境（量化方向为 `QuantConfig` 与数据，封存段切分是方向级），所有者批准后冻结；换检查器 MUST 走新提案、新批准。每个假设的实验计划（任务提示、主张类型、预算、这个假设自己的数据与隐藏实例及其哈希、`prior_art`、种子程序与用已批准检查器测得的种子得分、规则种类与阈值、`on_budget_exhausted`、`rationale`）SHALL 在所有者授权该假设的轮次时写入授权记录的摘要并随之冻结；同一方向的不同假设 MAY 有不同的主张类型、预算、数据与外部参照。Agent 执行环境中，所有领域（包括 `quant`）的 `scaffold` SHALL 只接受已批准的方向提案与已授权的实验计划，并在逐个文件核对哈希后写入任务目录；量化的评估配置（含前视检查次数）MUST 来自已批准的方向提案，不接受命令行临时传入。提案与批准 SHALL 作为 `ledger.json` 中与轮次授权、封存段批准分开的记录种类保存，并与运行日志对照；检查器批准 MUST NOT 能用来打开封存段。`checker approve` 在 Agent 执行环境中 MUST 拒绝，批准前 MUST 重算提案副本的哈希。Agent 执行环境的界定与 research-agent 变更相同（`SHINKA_RESEARCH_AGENT` 标记）；它防止 Agent 按约定之外的路径操作，不是针对恶意进程的安全边界。手动终端的量化流程保持不变。

#### Scenario: 未批准的提案

- **WHEN** Agent 在方向只有待批提案时执行 `scaffold`
- **THEN** 命令拒绝，不留下任务目录，待决事项显示该提案等待批准

#### Scenario: 实验计划未授权

- **WHEN** 方向已批准，但某个假设的实验计划没有出现在任何轮次授权中，Agent 为它执行 `scaffold`
- **THEN** 命令拒绝

#### Scenario: 同一方向的两个假设

- **WHEN** 同一方向的两个假设分别以证书型、预算 5 美元和经验型、预算 20 美元获得授权
- **THEN** 两个实验共用方向批准的检查器与数据，各自冻结自己的主张类型、预算与外部参照

#### Scenario: Agent 关掉前视检查

- **WHEN** Agent 环境中，量化方向的 `scaffold` 在命令行传入 `--lookahead-checks 0`
- **THEN** 命令拒绝；前视检查次数只取自所有者批准的方向提案

#### Scenario: Agent 自己批准

- **WHEN** 带 Agent 标记的环境执行 `checker approve`
- **THEN** 命令拒绝，账本中没有新批准

#### Scenario: 副本被改

- **WHEN** 提案提交后，`proposals/` 中的副本被改动，所有者随后批准
- **THEN** 批准被拒绝并说明哈希不符

#### Scenario: 批准后再改

- **WHEN** 所有者批准后，Agent 想换检查器
- **THEN** 只能提交新提案并开新实验，已冻结的实验计划不变

### Requirement: 量化行为保持不变

量化 SHALL 成为第一个领域预设。现有方向的笔记本 MUST 不经手动迁移即可读取；结论、试验计数、封存段与工作台显示 MUST 与迁移前一致。

#### Scenario: 旧笔记本

- **WHEN** 一个没有领域字段的现有量化方向被读取、质疑并显示在工作台
- **THEN** 它按量化处理，质疑结果与工作台截图与迁移前相同

### Requirement: 外部系统的产出只经冻结检查器才算证据

Agent 调用的外部系统产出的程序或结果 SHALL 只作为候选；候选 MUST 放进实验任务目录、经冻结检查器在隔离执行下重新评估并计入试验数之后，才进入证据与结论。

#### Scenario: 外部系统自报的分数

- **WHEN** 一个外部系统报告某个候选得分很高
- **THEN** 结论台账不使用这个分数，直到同一候选经冻结检查器评估
