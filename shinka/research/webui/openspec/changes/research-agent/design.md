# Design：内置对话 Agent

## 依据

所有者决定见 `proposal.md`。方案经 discuss:steady 与 discuss:divergent 各两轮审查、mentor:default 一轮审查（2026-10-20），三方一致的结论：授权规则必须落在每个入口都要经过的 `shinka_research` 里，而不是落在提示词或网页里；对话交给 omp 与现有 `shinka-research` 技能，不另建模型工具循环。

## 分层

| 层 | 归属 | 职责 |
|---|---|---|
| 状态层 | `shinka_research`（文件锁） | 笔记本原子保存；运行租约（pid、进程创建时间、假设与实验、轮次、开始时间）；轮次授权账本；封存段批准记录；尝试次数；运行日志 |
| 运行监管 | `shinka_research run --detach` 与 `watch` | 运行脱离启动它的窗口；`watch` 幂等对账：进程已死时按今天 `finally` 的逻辑恢复状态、计数并写一条事件，两个 `watch` 同时运行也只写一次 |
| 研究数据服务 | `ui.py`，仍只读 | 增加 Host 校验；新增只读接口提供运行日志、授权账本、待决事项与 Agent 状态 |
| 本机 Agent 服务 | 新的独立进程与端口 | 网页的全部写入：对话轮次与批准；以 RPC 驱动无界面 omp 并提供宿主工具；网页开着时定时运行 `watch` |
| 终端 | Tern 里的 omp | 同一技能；一个定时 shell 任务运行 `watch`，只在有变化时唤醒模型 |

## 决定

### 规则落在 CLI，提示词只作说明

两个 Agent 以同一个系统用户运行。终端 Agent 有 shell；网页 Agent 没有通用 shell，只能调用 Agent 服务提供的宿主工具（《对话适配》，4.1）。真正的控制点是 `shinka_research` 内部的检查，网页 Agent 另有服务端宿主工具的检查作为第二道防线。因此：

- **Agent 执行环境。** Agent 服务启动 omp 时设置 `SHINKA_RESEARCH_AGENT`；终端 omp 按使用说明在同一标记下运行。带标记时：
  - `run` 必须带 `--grant`，指向一条尚有尝试余量的轮次授权，并在锁内扣减尝试次数；授权用完即“本轮结束、暂停”。授权由 `grant` 命令记下（在 Agent 执行环境中拒绝），列明一个或多个假设（尝试次数合计，一份授权就是一轮）、尝试次数、单次费用上限、Agent 可设的 `--set` 键，并记下当时各实验 `shinka.yaml` 的哈希。所有拒绝都在扣减之前；先写尝试记录再登记租约，崩溃时最多损失一次尝试（已扣减却没有运行），不会有不记尝试的运行。取用最后一次尝试时运行日志记一次 `round-paused`，之后的请求被拒绝。Agent 发起的 `run` 一律转入后台（`--detach`）。
  - `run` 注入的单次费用上限取方向剩余预算与授权单次上限中较小的一个，拒绝自带的 `evo.max_api_costs`，其余 `--set` 键只放行授权列出的项。剩余预算 = 预算 − 已花 − 进行中的授权运行还能花的部分（每个停在自己的上限）；方向有预算、又有不带停止点的运行（如所有者的手动运行）在进行时，授权运行等它结束（2026-10-21 采用的保守默认，待所有者确认）。`shinka_run` 续跑实验的数据库时从库里已花的费用起算上限，这个起点与 `run_cost` 用同一条规则（《费用口径》），所以注入的是该实验此前的花费加上本次上限，原值传入不取整；此前的花费未知时授权运行拒绝启动，不论方向有无预算，因为从下限起算会让运行花过上限。这是软上限：`shinka_run` 在已承诺的费用达到上限后不再发起新提案，进行中的提案照样完成，实际花费可能多出一批。桩 LLM 驱动真实运行循环的实测（`tests/research/test_cost_cap_stop.py`，每次调用固定 0.125 美元，并发 4）：新库上限 1.0 时正好停在 1.0；续跑时多出至多（并发数 − 1）次调用，实测 0.25；动态岛屿复制出的有费用副本不再挤占续跑的额度（起点等于 `cost_before`）。透传参数只接受精确的 `--set KEY=VALUE` 或 `--set=KEY=VALUE`（`shinka_run` 的解析器接受 `--se` 之类的前缀，重复的选项以后者为准），键按 `shinka_run` 的规则校验；`grant` 拒绝列入会绕开冻结与计数的键（`job.eval_program_path`、`job.extra_cmd_args`（它的参数接在评估器的 `--program_path` 与 `--results_dir` 之后，可以替换它们，`scheduler.py:174-184`）、`db.db_path`、`evo.init_program_path`、解释器与激活脚本类键）。`shinka.yaml` 不在冻结哈希之内，手动运行照旧可改；授权运行要求它与授权时一致，授权之后搭建的实验须保持 scaffold 写出的样子（只有任务说明），文件缺失、读不了或形状不对都算改过。
  - `confirm` 必须带 `--approval`，指向一条未使用的封存段批准，在开始打分之前于锁内消耗它；`--yes` 在该环境中不够。消耗批准与写下开封记录（冻结的程序、评估器与开发数据哈希、试验数、阈值与源文件哈希，`DataSplit.opening`）在同一次加锁中完成；开封记录与中断规则手动 `confirm` 同样适用（spec《封存段开封可中断而不重看》）。批准由 `approve` 命令记下（在 Agent 执行环境中拒绝，封存段已开时拒绝），条款同样写入运行日志；消耗记录写在开封记录保存之前并记下开封时间，崩溃时最多损失一条批准（已消耗却没有开封），封存段不会在批准未消耗时打开。中断后补完与补写文件时，Agent 要出示开封所用的那条批准；任何环境里 `--approval` 都可代替 `--yes`，并按同样规则消耗。
- **手动用法不变。** 不带标记时，所有者手动执行的 `run`（含任意 `--set`）与 `confirm --yes` 保持现有行为，运行日志记为手动运行，让所有者能看出 Agent 会话期间是否有人绕开授权。手动运行、`finish_run` 与 `watch` 都不读账本，账本损坏不影响它们；运行的结束事件在运行编号之前写明 `manual run` 或所用授权与尝试（`agent run, grant g001 attempt 1 of 2, …`）。`--grant` 在没有标记的环境里同样按授权规则执行。`probe` 在计数的同一次加锁中写一条 `probe` 事件（程序名与 agent/manual，不含分数）。
- `authorized-round` 是新增的程序模式：Agent 在授权范围内跑完这一轮，不在每个检查点停下，授权用完即暂停。原 `autonomous` 与 `checkpoints` 模式及其对手动使用的含义保留（所有者 2026-10-20：原有功能一项不删）；技能说明补充一句：在 Agent 执行环境中，任何模式下的运行都受授权账本约束。

**残余风险，如实写明：** 同一系统用户下有 shell 的 Agent 能去掉标记、走手动路径，也能伪造任何文件，账本与记录不防篡改。能做到的是让绕开与篡改可被发现：冻结数据与评估器已有哈希校验（`cli.py:153`），授权与尝试记录只追加，`run` 启动前核对，手动运行单独记日志。核对的做法：账本 `ledger.json` 整文件原子写入；每条授权的条款同时写入 `results.tsv`（先写日志），授权运行的结束事件写明所用尝试，`run` 启动前把账本与运行日志、进行中的租约对照，删掉尝试记录或改动授权条款都会让授权运行拒绝启动；把两份文件一起改仍能伪造，去掉标记的 Agent 给出的授权与所有者给的无从区分。第二道防线只防误操作，不是隔离。网页 Agent 的宿主工具不经 shell 启动 `shinka_research`，标记与 PATH 由服务设置，子命令有白名单（没有 `grant`、`approve`），写文件只限白名单目录，所以去掉标记、直接调用 `shinka_run`、直接编辑 `notebook.json`、`results.tsv`、`ledger.json`、`.work.json`、`experiments/**/task/`、`library/raw/` 在网页 Agent 里做不到；但 `probe` 会执行 Agent 写的种子代码，代码本身仍能做任何事。终端 Agent 有 shell，而 omp 的 `bash.patterns` 只管内置的 `bash` 工具：所有者终端里的 shell 工具是扩展提供的 `exec_command`，实测不受它约束，`eval` 也不受（4.1），所以终端一侧只有技能说明与事后核对。原写法“去掉标记、直接调用 `shinka_run` 与 `grant`、`approve`、`confirm` 类命令需所有者审批”据此改为网页 Agent 中做不到（2026-10-21，待所有者确认）。

### 费用口径（2026-10-21 重开）

预算与授权上限只有在每一笔已付费用都算进去时才成立。用桩 LLM 自己的计费账本作基准实测（`tests/research/test_run_spend.py`），原来的总额对不上账：运行器续跑时把岛副本复制来的费用又算一遍；耗尽尝试的提案只写进 `attempt_log`，`run_cost` 与续跑都不读；失败的元分析已付的步骤哪里都不记；到达目标后丢弃的多余提案、空回复、存不下的提示词进化也都漏掉；数据库里只有第 0 代与失败提案时，续跑不载入已花费用。现在 `shinka/database/cost.py` 是唯一的规则，运行器续跑、`run_cost`（预算、授权、工作区、运行统计）都用它：非岛副本程序的四个费用字段，加上没有成为程序的已付费工作的 `attempt_log` 事件（耗尽尝试的提案、到达目标后丢弃的提案、没有存下提示词的提示词进化，每条各计一次），加上提示词进化；元分析无论成败都把已付的步骤记在触发它的程序上，结尾总结记在最佳程序（是岛副本时记在原程序）上。缺字段算记录的 0，值不是有限非负数、元数据或事件读不出时算未知；运行器写不进数据库或得不到金额的费用（目标清理时挂起被取消的提案、超时的结尾总结、多次重试仍写不进的程序）在结果目录留下 `cost-unrecorded.json`，合计从此未知，直到缺的费用补记进持久记录，只删除它不算。未知的合计让带上限的续跑、授权运行与有预算方向的新运行都拒绝；连标记都写不下时运行以 `ChargeNotRecordedError` 停下。到达目标时进行中的提案不再取消（取消会丢掉已付的调用），等它们做完、记下费用后丢弃，最多等 600 秒。研究分析页的总额改由服务端给出，逐程序之外的部分列为“其他费用”，比较页曲线终点落在这一总额上。经 mentor:default、discuss:steady 与 discuss:divergent 审查，规格见 award-redesign《运行费用只有一个口径》。

### 两份对话，一份状态

单一会话对接两端在现有条件下做不到：Tern 里的 omp 终端界面不能挂到另一个进程的会话上，omp 的协作功能默认经过外部中继 `my.omp.sh`。所有者因此选择两边对话各自独立：每一轮对话先读 `brief` 与账本，所以都知道另一边做过什么。两个写入者的正确性由文件锁与运行租约保证：同一实验已有存活的运行时，再对它发起的 `run` 拒绝，因为两个进程会写同一个 `programs.sqlite`；同一方向的不同实验仍可并行。

**轮次的只读视图**（`shinka/research/rounds.py`，3.2；经 discuss:steady 与 discuss:divergent 审查，2026-10-21）：工作台的 `/api/snapshot` 与 `/api/workspace`、终端的 `brief` 与之后的 Agent 服务都读同一个 `overview()`，Agent 服务直接调用，不经 HTTP。它不加锁，按租约 → 账本 → 运行日志的次序读，因为授权与批准先写日志再记账，后读的日志总与先读的账本一致；仍对不上时重读一次再报。每次尝试的运行是 `running`、`starting`、`unrecorded`（进程已无、结束未记）、`ended`（结束事件优先于租约）或 `lost`。运行先写尝试再登记租约，读者可能正好看到两者之间：尝试已有而租约还没有，60 秒内算 `starting`（不是结束，不出待决事项），超过才算 `lost`（尝试已取、运行一直没登记，只有崩溃会留下）。只读的一方不加锁，所以只能这样等，不能靠立刻重读。待决事项按授权计：一份授权的尝试都已取完、运行都已记下或已丢失，且之后没有涉及同一假设的新授权，就是 `round-over`；`lost` 单列一条，不挡本轮结束，因为它永远不会结束。Agent 用不了一份尚有余量的授权、只有所有者能解决的原因（预算没有余量或未知、有不带停止点的运行、`shinka.yaml` 改过）列为 `stuck`，用的就是 `run` 的那几项检查；这只是能从文件算出的部分，被拒绝的命令本身不记日志。状态是这一方向轮次的状态，不是对话的状态（对话在 Agent 服务里）：`unknown`、`blocked`、`running`（只算 Agent 的运行，含正在启动的）、`waiting`、`open`（有余量、没在跑）、`idle`。只给列出的字段，原因只给代码。**未决：Agent 的申请**（下一轮、开封存段）目前没有任何记录处，终端里提出的申请到不了网页；所有者决定“Agent 只能申请”“两边共用待决事项”意味着申请要进共享状态，记录方式随 4.2 与所有者确认后定，`pending` 届时加一类。

### 网页写入的安全边界

- Agent 服务不复用 `DatabaseRequestHandler`：它监听所有地址（`visualization.py:1766`）并发送 `Access-Control-Allow-Origin: *`（`visualization.py:1684`）。
- Host 只接受 `127.0.0.1:<端口>` 与 `localhost:<端口>`，防 DNS 重绑定；研究数据服务同样加这一条，因为它现在不校验 Host，重绑定页面能读到笔记本与代码。
- 每个写请求与 WebSocket 升级都精确校验 Origin；不发任何 CORS 头；令牌放在 `Authorization: Bearer` 头里，不用 cookie。
- 令牌每次启动用 `secrets.token_urlsafe(32)` 生成，只在内存里；通过打开页面的 URL 片段交给网页，网页存入 `sessionStorage` 并清掉片段；不写进工作区、不进 Agent 子进程环境。
- 批准卡片只能由所有者点击；Agent 的申请与所有者的批准分属两个接口。所有者选择网页也能批准封存段，因此渲染不可信文本的净化规则（award-redesign tasks.md:3）同样适用于对话页。

### 对话适配

Agent 服务以 RPC 启动无界面 omp：`omp --profile <服务自有 profile> --mode rpc --no-ui --no-extensions --no-rules --tools read,grep,glob,web_search --config <覆盖文件>`，子进程环境由服务显式给出，不带父进程的 `PI_*`、`OMP_*` 与模型密钥变量。服务自有的 profile 隔开的只是 omp 自己的配置、凭据与会话状态：所有者 `~/.omp/agent` 下的 `APPEND_SYSTEM.md`、`PERSONALITY.md`、设置、扩展、MCP 与登录都不进来。其他来源不按 profile 分开，要靠启动行与覆盖文件挡住：其他工具的用户目录（`~/.claude`、`~/.codex`、`~/.gemini`）靠外来用户来源默认不启用（实测时 `enabledProviders` 是默认的空列表，覆盖文件要显式写上），`~/.agents` 下的技能靠下面的 `skills.includeSkills`，项目里的文件靠下面的 `disabledExtensions` 与 `--no-rules`，扩展靠 `--no-extensions`，模型密钥靠显式的环境。代价是 Agent 用的模型凭据要在这个 profile 里另配一次（所有者在该 profile 下登录，或由服务传入密钥）。对话仍由 omp 承担；研究操作经 `set_host_tools` 注册的宿主工具回到服务执行：`shinka_research`（参数数组，服务用 venv 的 `python -m shinka.research.cli` 不经 shell 启动，子命令白名单，`--dir` 限在工作区内，拒绝含 NUL 或换行的参数）、`write_research_file`（Windows 路径规范化后只允许白名单目录与 `.md`、`.py`），再补 `shinka_models` 与查看最佳程序的只读工具，免得 Agent 借 `probe` 做分析而虚增试验数。覆盖文件由服务每次启动时写在工作区之外：`skills.customDirectories` 指向仓库的 `skills/`（默认不加载项目技能），`skills.includeSkills` 只留 `shinka-research`；`autolearn.enabled: false`（否则 Agent 会写所有者的全局记忆与技能）；显式给出 `tools.approvalMode: yolo`；`tools.approval.write: deny`；`personality: none`，不用 omp 的编码助手人设；`disabledExtensions` 去掉项目的 `AGENTS.md` 与 `CLAUDE.md`（仓库根目录现在只有这两个上下文文件），`--no-rules` 去掉规则文件，以后若加了别的约定文件（如 `.github/copilot-instructions.md`、`.gemini/GEMINI.md`）要在这里另加；会话目录由服务自有。不带 `find`：它要额外调用已配置的 judge 模型（`find.enabled: auto` 时须解析到原生判别模型），无凭据 profile 中不可用，本方案不启用，`grep`、`glob`、`read` 已够用。`--no-ui` 下需要审批的调用直接失败，不转给网页；网页的批准卡片只来自 Agent 的申请（4.2）。不引入新依赖：协议只用标准库。

握手实测（2026-10-21，不发提示词，零花费；判据、脚本与逐帧记录在 `E:/projects/ShinkaEvolve-runs/handshake-20261021/`）：RPC 与 ACP 都能作为 Windows 管道子进程在 0.2–2.8 秒内就绪，关 stdin 后退出码 0、不留进程；默认不加载项目技能，覆盖文件能加载；覆盖文件里的数组整体替换全局值（`bash.patterns` 只剩覆盖项，所有者的 14 条全局拒绝规则消失）；`--tools` 只筛内置工具，扩展工具全部保留；所有者的会话没有内置 `bash`，扩展的 `exec_command` 不受 `bash.patterns` 约束；`--no-extensions` 恢复 `bash`；4.1 当时的启动行（所有者的默认 profile，`--tools read,grep,glob,find,web_search`）加两件宿主工具后，工具只剩 `find`、`glob`、`grep`、`read`、`web_search`、宿主工具与只写设备的 `write`，技能仍在，没有界面帧。审批请求的形态只有文档依据：RPC 是按按钮文字识别的通用 `select`，ACP 只有 bash 是带类型的 `session/request_permission`。事先写下的决定性判据是审批请求的形态，但按最终配置两种方案都不把审批交给网页（ACP 方案显式 `yolo`、只留拒绝规则，RPC 方案用 `--no-ui` 让需要审批的调用直接失败；这是方案的选择，不是传输做不到），真正区分它们的是 RPC 有宿主工具：第二道防线因此落在服务的 Python 代码里，能用 pytest 测；ACP 方案只能靠命令文字规则，`python -m shinka.cli.run`、`env -u` 与脚本都能绕过。宿主工具在握手中只实测到注册被接受。方案经 discuss:steady 两轮与 discuss:divergent 审查。未比较的候选：Claude Agent SDK，文档写明宿主回调 `canUseTool` 逐次批准工具调用与进程内自定义工具，正是事先写下的审批形态；是否纳入待所有者定（tasks.md 4.1）。握手没有测模型发起的 `host_tool_call` 往返与带消息会话的重启恢复（空会话实测不能恢复：RPC 不落盘，ACP 的 `session/list` 为空、`session/load` 报找不到），两者在 4.2 第一道关实测（下段）。网页 Agent 是否需要执行任意脚本（pandas、git、装包）由所有者决定；需要的话它也能改写账本，只能事后发现。

4.2 第一道关实测（2026-10-21；判据在运行前写下，脚本、桩模型收到的每个请求与逐帧记录在 `E:/projects/ShinkaEvolve-runs/stub-gate-20261021/`；记录运行为 `summary-154649.json`，之前的 153748（启动失败，去掉 `find`）、153837 与 154315（发现下面两处问题）和跟进 154154 留作证据）：上面的启动行跑在无凭据的独立 profile 下，profile 里只有一个指向 127.0.0.1 的本地 Anthropic Messages 桩模型，桩按提示词里的场景标记发出工具调用并记下收到的一切，判定只看它收到的请求与 RPC 帧。零模型花费靠构造保证：子进程环境去掉了模型密钥变量，profile 新建、没有登录过，唯一的模型提供方是桩，桩收到的每个请求都带 profile 里的字面密钥。宿主工具往返与文档一致：`host_tool_call` 带着模型给的 `toolCallId`、工具名与参数，宿主的结果原样回到模型的 `tool_result`，`isError: true` 回去就是 `is_error`；宿主不答而服务发 `abort` 时，omp 发出 `targetId` 指向那次调用的 `host_tool_cancel`，这一轮以 `aborted` 结束，下一次请求带着 omp 补上的错误结果“Host tool "shinka_research" was aborted”，迟到的结果被丢弃，会话照常继续。内置 `write` 在这条启动行下根本不提供（模型调用得到“Tool write not found”）；显式加进 `--tools` 时，`yolo` 下 `tools.approval.write: deny` 拒绝它（“Tool "write" is blocked by user policy”），文件没有写出；`write_research_file` 不受这条拒绝规则影响。模型收到的工具正好是 `glob`、`grep`、`read`、`web_search` 与两件宿主工具，没有 MCP 工具，收尾时 omp 的子进程只有 `conhost`；技能只有 `shinka-research`；所有者的 `APPEND_SYSTEM.md`、`~/.claude/CLAUDE.md` 与 `~/.codex/AGENTS.md` 一行都没进来。最初命中的人设段落按推断是 omp 自带的默认人设，不是所有者的文件：它只有所有者 `PERSONALITY.md` 的前 18 行、没有第 19–40 行，换成 `pragmatic` 后命中归零，而用户级 `PERSONALITY.md` 会盖过任何预设；`personality: none` 去掉的是这个默认人设，不是在堵泄漏。真正的问题是项目的 `AGENTS.md` 经 `<repo-rules>` 块有 18 行进了提示词，现由覆盖文件去掉。重启恢复：一轮对话结束后把 omp 进程树强行结束，同一会话目录加 `--continue` 再启动，消息数仍是 2，模型在下一次请求里收到上一轮的回复。按事先写下的规则，4.1 的选择（RPC 加宿主工具）成立，不退回 ACP。这次实测带出的事实：父进程环境里有 `DEEPSEEK_API_KEY`、`CODE_INTEL_ANTHROPIC_API_KEY` 与 `ANTHROPIC_BASE_URL`，服务必须显式给子进程环境；零模型花费不等于零联网，新 profile 第一次使用时 omp 在线刷新了 35 家模型提供方的目录；omp 第一次使用时会用一次模型调用把每个技能的说明改写成不超过 12 个词的路由提示，并缓存在 profile 里，之后提示词里的技能说明就是这条改写（实测中变成了桩的回复）；剩下的系统提示仍是 omp 的编码助手模板（“You are omp's trusted coding assistant”与工程规则），网页 Agent 的人设在 4.2 替换它。没有实测：真实模型的调用形态（只用了桩）、`web_search`（提供了但没有调用过）、Claude Agent SDK（是否纳入待所有者定）。

### 终端定时查看（4.3）

终端 Agent 由 omp 的定时 shell 任务唤醒：`shinka_research watch <dir> --since terminal` 先照旧补记进程已无的运行，再列出这个读者在运行日志里还没读过的消息，有就以 3 退出；任务设 `wakeOn=failure`，所以只有新消息（或 `watch` 自身失败）才发起一次模型回合，其余时候不花模型费用。消息只取 Agent 等待时会发生、必须通知它的事件：所有者的 `grant`、`approval` 与任何运行的结束（`run`、`run-failed`，Agent 自己发起的运行也算，结束发生在它等待的时候）；Agent 自己的步骤（`scaffold`、`probe`、`skeptic`、`verdict`、`confirm`、`round-paused`）不算，启动运行不写事件。运行正常结束时由监管进程自己记下结束，`watch` 无可补记，所以只看 `watch` 的补记输出会漏掉最常见的消息。读者读到哪里（读过的完整行数与最后一行）存在 `<dir>/.watch/<名字>.json`，只由 `brief --since` 在打印之后推进，`watch` 从不推进：唤醒没送到 Agent（会话在 shell 任务与模型回合之间关闭、模型出错）时下一次照样报告；位置丢失或日志被手改时全部事件重新算作消息，只会重复、不会漏。网页 Agent（4.2）用同一个 `rounds.news()` 与自己的读者名。方案经 discuss:steady 与 discuss:divergent 审查。没有采用对 `overview()` 的结果做差异比较：读不出时的问题视图会清空授权与批准，Agent 自己的动作（启动授权运行占用预算余量、用掉最后一次尝试、消耗批准）也会改变结果，都造成误唤醒。也没有采用只按时间窗取近期事件：终端关着期间的消息在重开后就看不到了。

### 借鉴 harness2（2026-10-21）

所有者提出可借鉴 google-research/harness2（arXiv 2610.14224）的概念，并回复「可以copy一些他的概念进来」。只读评估：读了提交 `1cfe254` 的代码与论文 v1，不安装、不运行、不调用模型，证据在 `E:/projects/ShinkaEvolve-runs/harness2-study-20261021/findings.md`；方案经 mentor:default 前后两次、discuss:steady 与 discuss:divergent 审查。harness2 有两层：任务层在同一任务上试跑改过的 Agent 配置（系统提示、护栏、记忆、工具说明、子代理、技能、脚本、插件），由大模型对比各次执行的过程与产出组合出新配置，不看评分；跨任务层把采纳的配置与经验存进按领域分开的经验库，选出结果且有经验就自动存入（`core/schedule.py:555-556`），之后不验证、不淘汰。

采纳的只有经验条目的格式（适用情形、经验、证据、把握）与“没有经验就不写”，放进每轮报告的最后一节，作为给所有者看的候选（2.8）：它不进资料库，也不是规则。技能说明要求 Agent 不按旧报告里的经验行事，技能说明改不改由所有者决定；这条界线只是说明，代码不强制，与《规则落在 CLI，提示词只作说明》的划分一致。`REPORT.md` 每轮重写，所以候选不累积，也没有采纳记录；要跨轮保存与跟踪采纳，须在 4.2 之后另做有代码支撑的设计。写候选不做新的评估，所以不增加试验，但它写在这一轮的对话里，仍算这一轮的模型花费。

不采纳三样。一是在进行中的研究轮次里试跑：每个任务每种模式要执行 4k 次、开 3k 次改进会话（`core/ledger.py:26-33`），每次执行都是一次试验、都花模型费用，而我们有评分器，不需要大模型盲选。二是直接用它的代码：核心依赖 POSIX 的 `fcntl`（`core/schedule.py:294`），执行端只有 OpenCode 与 Codex 的提示词、换别的会报错（`core/prompts.py:667-671`），执行端花费不记账（`core/recursion.py:299` 等处记账不带金额），默认联网不受限且 Codex 关掉沙箱（README）。三是自动存入的经验库：它与“规则在代码与所有者决定里”冲突。论文没有同算力对照，对每组最强对照方法的提升大多在噪声范围内（研究子任务按论文表 1–2 计算，未逐项复核）。

以后可选：由 ShinkaEvolve 在已完成的研究轮次上离线进化 Agent 的工作方式，用我们的评分器打分，先做论文缺的同算力对照（相同花费下的 best-of-k）。前提是先有真实轮次、评分器与所有者批准的费用，单独立项，排在前端与 4.2 之后。

## 验证（不花模型费用）

- 用 PATH 上的假 `shinka_run`（写入数据库行、睡眠、非零退出、挂起）测试：关掉终端后后台运行继续；运行被强杀后 `watch` 恢复状态并只写一次失败事件，两个 `watch` 同时运行仍只写一次；没有授权时 `run` 拒绝；超过尝试次数的一次被拒绝；没有批准记录时 `confirm` 拒绝；并发 `save` 不丢更新；篡改冻结文件或账本后 `run` 拒绝启动。
- 终端：定时 shell 任务在关闭并重开后补查一次，期间不触发模型。
- Agent 服务：不发提示词的启动与恢复；宿主工具往返用本地桩模型；令牌、Origin、Host 与批准接口的真实 HTTP 回归测试。
- 网页：用演示工作区加夹具账本（暂停中、待决、卡住、封存段申请）做交互实测。
- 端到端：假 `shinka_run` 加脚本化的对话适配跑完整流程（两端发起与批准、封存段批准、关窗后继续、重开补查与提醒）。所有者授权模型与费用后的真实运行是可选的所有者验收，不是完成条件。

## 风险

- Windows 上关闭 Tern 会结束整个控制台进程组；后台运行需要以新进程组脱离启动。
- 宿主工具往返与带消息会话的重启恢复只在本地桩模型下实测（《对话适配》第一道关）；真实模型是否按参数格式调用宿主工具没有实测。
- 运行被强行结束时尚在进行的模型调用没有任何记录，合计会少算而看不出来；《费用口径》只能让写不进与得不到金额的费用变成“未知”。补记缺失费用的工具未做，由所有者决定。
- 上游运行器的两个问题，费用口径的实测中发现，未修：续跑后已承诺费用重复计入时会提前锁住“到达上限”，运行空转到卡住检测才再发起提案（22 次实测中 8 次）；上一次运行最后几代失败留下了代目录时，续跑认领这几代失败而卡住，授权运行白白用掉一次尝试。
- 所有者的全局系统提示与个人设置在这条启动行下没有进来（第一道关实测）：profile 只隔开 omp 自己的配置，其余来源靠启动行与覆盖文件逐项挡住，omp 新增发现来源时要重新核对；剩下的是 omp 的编码助手模板与 omp 用模型改写的技能说明，网页 Agent 的人设与上下文在 4.2 定。服务自有的 profile 需要另配一次模型凭据。
- omp 首次使用时为每个技能多一次模型调用来改写说明，生产 profile 里这次调用走 Agent 的模型凭据，改写出的路由提示没人审过，4.2 建 profile 时要核对；`web_search` 在启动行里但从没调用过，它的凭据、联网与花费没有实测（是否保留待所有者定）；新 profile 首次使用会联网刷新模型目录。
- `os.replace` 在 Windows 上遇到读者打开文件时可能失败，原子保存需要重试。
- 终端重开后的补查依赖 omp 定时任务的 `missedWindow: catch_up_one`（技能里显式写明）；omp 以初始提示启动时会话开始的补查会被跳过，之后是否由定时器补上没有实测。实测只到定时任务以 `--since` 跑一次无消息的 `watch`（退出码 0，不唤醒模型）；Tern 中关闭并重开后的补查要所有者实测一次，无界面 RPC 会话的启动与 Tern 不同，不能代替。以 3 退出的唤醒会发起真实模型回合，没有实测，依据是 discuss:steady 引用的定时任务源码（`pi-schedule/src/prompt.ts`：唤醒提示带退出码与输出），本仓库没有核对。
