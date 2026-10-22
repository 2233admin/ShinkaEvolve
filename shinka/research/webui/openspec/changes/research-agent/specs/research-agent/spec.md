## Purpose

内置对话 Agent 的行为契约：所有者在工作台与终端都能和研究 Agent 对话、批准它的申请；研究运行不随窗口中断；授权边界由 `shinka_research` 执行。

## ADDED Requirements

### Requirement: 授权的这一轮做完就暂停

在 Agent 执行环境中（Agent 服务启动的 omp 与按说明配置的终端 omp 都带 `SHINKA_RESEARCH_AGENT` 标记）发起的 `run` SHALL 携带一条所有者批准的轮次授权编号（`--grant`），授权列明方向、假设、尝试次数、单次费用上限与允许的参数。此时 `run` MUST 在授权不存在或没有余量时拒绝启动，并在锁内扣减尝试次数；授权用完时这一轮结束，Agent 暂停并把下一轮列为待决事项。没有该标记的手动 `run` 保持现有行为，运行日志把它记为手动运行。

#### Scenario: 没有授权

- **WHEN** Agent 执行环境中的 `run` 没有带 `--grant`，或带的授权不存在
- **THEN** 命令拒绝启动并说明需要所有者授权，不产生运行、不扣预算

#### Scenario: 尝试次数用完

- **WHEN** 授权允许 2 次尝试，第 3 次 `run` 到来
- **THEN** 命令拒绝启动，运行日志记录本轮暂停，待决事项出现“是否开下一轮”

#### Scenario: Agent 自带费用上限

- **WHEN** Agent 执行环境中的 `run` 传入 `--set evo.max_api_costs=...`
- **THEN** 命令拒绝该参数，实际上限取方向剩余预算与授权单次上限中较小的一个

#### Scenario: 手动运行不受影响

- **WHEN** 所有者在没有 Agent 标记的终端里执行 `run`，并传入 `--set evo.max_api_costs=...`
- **THEN** 运行照旧启动并使用该上限，运行日志记为手动运行

### Requirement: 封存段只凭所有者批准打开

Agent 执行环境中的 `confirm` SHALL 携带一条未使用的封存段批准编号（`--approval`），并在开始打分之前于锁内消耗它；在该环境中 `--yes` 不足以打开封存段。批准可以在网页或终端给出，Agent 只能申请。所有者在没有 Agent 标记的终端里执行 `confirm --yes` 照旧打开封存段；开封与中断的规则两种环境相同，见下一条要求。

#### Scenario: 没有批准

- **WHEN** Agent 执行环境中的 `confirm --yes` 没有带有效的批准编号
- **THEN** 命令拒绝，封存段保持关闭

#### Scenario: 批准只用一次

- **WHEN** 两个带同一批准编号的 `confirm` 几乎同时开始
- **THEN** 只有一个消耗批准并开启封存段，另一个被拒绝

#### Scenario: 所有者手动确认

- **WHEN** 所有者在没有 Agent 标记的终端里执行 `confirm --yes`
- **THEN** 封存段照旧打开，且整个方向仍只能打开一次

### Requirement: 封存段开封可中断而不重看

`confirm` SHALL 在不碰封存段的检查（假设为 supported、结论所指的实验与程序存在、评估器与开发数据哈希、源文件哈希）全部通过、且方向里没有仍在增加试验的 `run`（`probe` 一开始就计为一次试验）之后，才在锁内冻结每个假设的程序、评估器与开发数据的哈希、试验数、阈值与源文件哈希，并把封存段记为已开，然后才运行候选程序。程序自身在封存段上的失败 MUST 只记类别（超时、出错退出、仓位无效、前视、开封后被改动、中断）并判为 unconfirmed，不记其输出；全部结果 MUST 一起写入笔记本，后打分的程序读不到先打分的结果。中断后再执行 `confirm --yes` MUST 只补完冻结时尚未开始打分的假设并沿用冻结的规则；已开始打分的假设记为 unconfirmed（中断），不得重跑。开封未完成期间 `run`、`probe`、`skeptic`、`verdict`、`scaffold` MUST 拒绝，同一方向同一时间只能有一个 `confirm`。

#### Scenario: 开封前出错

- **WHEN** `--data` 不是方向切分用的文件，或评估器被改动
- **THEN** `confirm` 拒绝，封存段保持封存，改正后可以再执行

#### Scenario: 打分中途中断

- **WHEN** 一个程序已经开始在封存段上运行时 `confirm` 被中断
- **THEN** 封存段显示已开封、方向仍为进行中，`run` 等命令拒绝；再次执行 `confirm --yes` 不重跑该程序，把它记为 unconfirmed（中断），其余假设按冻结的阈值补完

#### Scenario: 程序在封存段上前视

- **WHEN** 被确认的程序在封存段内用到未来数据
- **THEN** 该假设记为 unconfirmed，原因只写“前视”，其余假设照常打分

#### Scenario: 同时两个 confirm

- **WHEN** 第二个 `confirm` 在第一个运行时开始
- **THEN** 它立即被拒绝，不打分，封存段状态不变

#### Scenario: 还有试验在进行

- **WHEN** 一个 `run` 仍在增加试验时执行 `confirm --yes`
- **THEN** 命令拒绝，封存段保持封存；运行结束后再确认，冻结的试验数包含它的全部试验。`probe` 一开始就计为一次试验，进行中也不挡开封

### Requirement: 运行不随窗口中断

Agent 发起的运行 SHALL 脱离启动它的终端或网页继续执行，并留下运行租约。`watch` MUST 幂等：进程已结束时恢复假设状态与试验计数，并只写一条结束或失败事件。

#### Scenario: 关掉终端

- **WHEN** 运行开始后启动它的终端被关闭
- **THEN** 运行继续，研究界面显示“运行中”

#### Scenario: 运行被强行结束

- **WHEN** 运行进程被强杀后，两个 `watch` 同时运行
- **THEN** 假设状态恢复，试验计数按数据库重算，运行日志只多一条失败事件

### Requirement: 开着时才盯

终端或工作台开着时 SHALL 定时运行 `watch`，只在运行日志或授权账本有变化时唤醒模型；两者都关闭后不再盯，重新打开时补查一次。

#### Scenario: 重新打开

- **WHEN** 所有窗口关闭期间一轮运行结束，之后所有者打开工作台
- **THEN** `watch` 补查一次，Agent 页与顶栏显示本轮结果与待决事项

### Requirement: 两份对话，一份状态

工作台与终端的对话 SHALL 各自独立，共用运行日志、授权账本与待决事项；每一轮对话先读取 `brief` 与账本。同一实验已有存活运行时，再对该实验发起的 `run` MUST 被拒绝，因为两个进程会写同一个 `programs.sqlite`；同一方向的不同实验仍可并行。

#### Scenario: 两端同时发起

- **WHEN** 终端里的 Agent 正在运行实验 h002/e01，网页里的 Agent 也对 h002/e01 执行 `run`
- **THEN** 后者被拒绝并说明已有运行，账本与笔记本没有被覆盖

#### Scenario: 不同实验并行

- **WHEN** h002/e01 正在运行，另一端对 h003/e01 执行带有效授权的 `run`
- **THEN** 两个运行同时进行，各自记账

#### Scenario: 运行中的试验数

- **WHEN** 一个实验正在运行，notebook 记的已评估数还是运行开始前的值
- **THEN** `brief` 与 `status` 报的试验数等于各运行库非岛副本的已评估程序数加试探次数，与工作台和质疑检验一致

### Requirement: 网页写入只经本机 Agent 服务

研究数据服务 SHALL 只接受 GET，并只接受 Host 为 `127.0.0.1:<端口>` 或 `localhost:<端口>` 的请求。对话与批准 MUST 经独立的本机 Agent 服务：每次启动生成令牌，令牌经 URL 片段交给页面并只放在 `Authorization` 头中；每个写请求与 WebSocket 升级精确校验 Origin 与 Host；不发 CORS 头。

#### Scenario: 跨站请求

- **WHEN** 另一个网站的页面向 Agent 服务发起批准请求
- **THEN** 请求因 Origin 不符或缺少令牌被拒绝，账本不变

#### Scenario: DNS 重绑定

- **WHEN** 请求的 Host 是 `attacker.example:8772`
- **THEN** 研究数据服务与 Agent 服务都拒绝该请求
