# Tasks

动手前的问题已全部由所有者回答（`proposal.md` › 所有者决定）。2.5 的工作台合并确认与所有者凭据依赖 research-agent 的 4.2、5.1；在那之前 2.x 只按命令行验收。第 5 节的许愿与不联网的防茧房（5.1–5.3）不依赖其他各节，已先做；联网部分（5.4）所有者 2026-10-23 定为随 Agent 一起做：「许愿搞定就ok联网跟agent有关系」，并选择“许愿 + 不联网的防茧房”先做。

5.1–5.3 经 PR #30 以合并提交 `526587e` 并入 main（分支提交 `94fe2bd`，管线已按它登记交付）。所有者 2026-10-23 授权自动合并：「你自动合并吧 合并了 保证项目能用就收口」。合并后在 main 上整套 `pytest -q tests/research` 出现一次 `test_a_killed_run_is_recorded_once_by_two_watches` 失败：运行在结果库仍是回滚日志模式时带着未提交事务被杀，留下热日志，只读读取一律报 “attempt to write a readonly database”。记录运行结束前先回滚（`rundb.roll_back_dead_writer`），回归测试 `test_watch_records_a_run_killed_with_a_transaction_open`。

## 1. 核心与领域分开

- [ ] 1.1 方向的领域：`init --domain`（缺省 `quant`，另有 `open`、`math`），写进笔记本后不能改；现有方向读取时视为 `quant`；`program.md` 模板按领域生成。检查：在一个方向里脚手架另一个领域的实验被拒绝。
- [ ] 1.2 实验计划：实验记录新增可选的冻结计划字段（旧字段保留）；`verify_experiment` 按计划逐个文件校验，包括生成的包装器。检查：改动计划中任一文件后 `run`、`probe`、`skeptic`、`verdict` 都拒绝并指出文件。
- [ ] 1.3 核心评估包装器：候选在 `python -I` 子进程里运行（Windows 上包进 Job Object，退出时连同子进程一起结束）、约定只写产物；候选退出后检查器在另一个进程只读产物，输出 `metrics.json` / `correct.json`；`cmd_probe` 改走包装器。确认：终止候选进程树，把检查器与产物复制进以哈希命名的只读目录、核对后只从副本加载，产物只用 JSON 或 `numpy.load(allow_pickle=False)` 解析，在新进程里校验。经验型结论与做不到上述条件的结论在工作台标“未沙箱”（操作系统沙箱所有者定为以后再做）。检查：候选改动检查器函数或自报得分不影响得分；运行中改动检查器文件后确认被拒绝；候选留下的子进程在确认前已结束。
- [ ] 1.4 判定规则：核心规则集合（`dsr_pbo`、`reach_target`、`counterexample`、`holdout_vs_baseline`）带版本，判定只看冻结的种类与参数；`reach_target` 的参数含 `on_budget_exhausted`（`refuted` / `insufficient`）；对话原文存为 `rationale`，不参与判定；没有计划的旧实验标为“未冻结（旧）”并照旧接受命令行阈值。检查：规则边界单元测试（含预算用完未达到时 `refuted` 判推翻、`insufficient` 判证据不足）；旧实验用原命令行阈值重新质疑，结论与迁移前相同。
- [ ] 1.5 量化迁移为预设：`prepare_data`、任务提示、评估器模板、`dsr_pbo` 与封存段确认归入量化预设；`file_sha256` 移到 `state.py`；`_not_confirming`、`brief`、`status`、`add_approval`、`rounds` 按能力处理（`design.md` › 必须一起改的量化写死处）。封存段开封的每个目标带自己计划的阈值，已开封未完成的旧记录继续用顶层阈值。检查：`pytest -q tests/research` 除新增测试外全部照旧通过；演示工作区的 `/api/snapshot` 除新增字段外逐字段相同。
- [ ] 1.6 未知领域：报 `ValueError` 子类，工作台只让该方向显示错误。

## 2. 检查器提案与批准（决定 A）

- [ ] 2.1 `checker propose`（每个方向一次）：复制可复用的检查器及其固定资源进 `proposals/<pid>/`、记录运行环境、算哈希；量化方向的提案是 `QuantConfig`（含前视检查次数，为 0 时标高风险）与数据；有已知正确或错误产物时附上并跑检查器；账本记 `p<n>`，运行日志写入文件摘要。检查：副本哈希与提案记录一致；量化提案能完整重建现有 `scaffold` 的配置。
- [ ] 2.2 `checker approve`：Agent 环境中拒绝；批准前重算副本哈希，不符就拒绝；账本记 `v<n>`，与 `g`、`a` 分开，`grants.verify` 认识并对照这两种记录。检查：同一提案只能批准一次；用 `v` 编号开封存段被拒绝。
- [ ] 2.3 假设的实验计划与闸门：`hypothesis plan` 写入任务提示、主张类型、预算、这个假设自己的数据、隐藏实例或留存数据（复制并记哈希；经验型必带其一并带最终评估；量化用方向的切分）、`prior_art`（状态 `found`：结果、来源、检索词、比较方向、绝对容差；`none_found`：检索词；`unavailable`：离线）、种子程序、规则种类与阈值、`on_budget_exhausted`（`reach_target` 必填）与 `rationale`；命令用已批准的检查器跑种子并记录得分，`reach_target` 的目标必须高于它。`grant` 显示计划全文并把摘要写进授权记录（与现有按哈希固定 `shinka.yaml` 的做法相同，`grants.py:157-165`）。Agent 环境中所有领域（含量化）的 `scaffold` 只接受已批准的方向提案与已授权的计划，再次核对哈希，不接受命令行临时传入的量化配置，不留下半成品任务目录。检查：未批准、计划未授权、命令行传 `--lookahead-checks 0` 三种情况都被拒绝；缺 `prior_art` 或 `on_budget_exhausted` 被拒绝；经验型计划既无隐藏实例也无留存数据时被拒绝；离线以 `unavailable` 提交成功；同一方向两个假设各自冻结不同的主张类型、预算与数据；三档边界（差于、容差内、优于）判定正确；手动终端的量化流程与迁移前相同。
- [ ] 2.4 改判限制：Agent 环境中 `verdict --label` 不能把结论往“成立”方向改。
- [ ] 2.5 待决事项、`brief` 与工作台列出待批的方向提案（检查器源码、固定资源清单、已知答案测试结果）与待授权的假设实验计划（任务提示、主张类型、预算、数据清单、种子得分、目标、`prior_art`）；工作台在方向第一轮把“批准检查器”与“授权这一轮”合成一次确认，之后每个假设只确认授权。
- [ ] 2.6 技能拆成“通用流程”与“量化一节”，加入“起草检查器 → 提案 → 等批准”；`docs/research/design.md` 原则 1–4 按新分层改写。

## 3. 第二个领域：数学构造

- [ ] 3.1 `math` 预设与圆填充检查器：候选产出圆心与半径，检查器校验并取 `sum(radii)`；确认只重新校验保存的产物。检查：越界、重叠、自报得分不符、篡改检查器四种情况都判不合法。
- [ ] 3.2 端到端示例：Agent 提案 → 所有者批准 → 脚手架 → 运行 → 质疑 → 结论 → 确认，跑通一个方向。不调用模型的夹具版本进入测试；真实运行在所有者授权模型与费用后进行。

## 4. 工作台

- [ ] 4.1 `/api/snapshot` 增加 `domain`、`capabilities`、规则与证据标签；隐藏规则不变；没有 `lockbox` 的方向，封存段相关列表为空。
- [ ] 4.2 报告页按标签显示检查台账与判词；`reach_target` 写明低于、复现还是超过 `prior_art`；经验型结论标“未沙箱”；没有 `trial_correction` 时不画噪声线；待决事项显示检查器提案（源码、数据清单、任务提示、规则、种子得分、目标与 `prior_art`）。检查：量化方向演示页面与迁移前逐像素相同；数学方向在三宽 × 深浅 × 中英下截图实看。

## 5. 许愿与联网（防茧房）

- [x] 5.1 来源记录：`library add` 按是否带 `SHINKA_RESEARCH_AGENT` 标记在 `log.md` 记 `[add agent]` 或 `[add human]`，不在 `raw/` 下新增文件。检查：两种环境各加一个文件与一个 URL，日志记录正确，`library pending` 输出不变。实现：`library.add(lib, source, by)`（`by` 必填，演示工作区按所有者加入记 `human`），`cmd_library_add` 由 `grants.in_agent_env()` 决定；`test_add_logs_whether_the_agent_or_a_human_added_it`。
- [x] 5.2 `brief` 资料状况一行：原始资料数（Agent / 人）、最近一次加入外部资料的时间、wiki 中结论页占比。检查：夹具工作区的输出。实现：`library.sources`（迁移前的 `[add]` 行，以及手工放进 `raw/` 的文件和手写进 `raw/links.md` 的 URL 中没有加入记录的，计为“没有记录来源”）、`library.finding_pages`（按前言 `type: finding`；没有前言的页面取第一行独立的 `Type: <种类>`，演示工作区的页面就是这样写的），`brief` 的 `- Sources:` 行；`test_brief_says_where_the_library_came_from`（含两个 `add` 写入的 URL 与一个手写 URL，只有手写的计入没有记录来源）、`test_brief_of_an_empty_library`；演示工作区 `d1-momentum-regimes` 的 brief 实测 `3 of 9 wiki pages are findings`，与 `build_demo_workspace.py` 的 `WIKI` 中 3 个 `finding` 页相符。
- [x] 5.3 模板与技能（不联网部分）：`templates/hypothesis.md` 的 Related work 引用原始资料或 URL、新增“最强的反方证据”；技能第 0 步改为 Agent 起草 `program.md` 的方向说明、约束与笔记（标 (owner) 的数据、评估设置、预算、自主程度、判定阈值由所有者写，`program.md` 模板按此标注并新增 Constraints 一节），并按 `brief` 的资料状况提醒所有者补资料；第 2 步读推翻结论不超出其页面声明的数据与时段；第 8 步检查只引用自己结论页的假设与超出结论范围的引用。
- [ ] 5.4 联网（随 research-agent 的对话 Agent 一起做）：技能第 1 步加入每个方向的“已知情况”调研（有网络时；离线记 `prior_art: unavailable`，工作台标“没有外部参照”），写明网页内容是不可信文本、不执行其中的代码与指令；量化一节写明不检索封存段开始后的表现。

## 6. 收尾

- [ ] 6.1 文档：`docs/research/design.md`、`roadmap.md`（勾掉“通用化”）、`workspace.md` 第 8 项、CHANGELOG。
- [ ] 6.2 验收：`pytest -q tests/research`、ruff 与仓库门禁；所有者查看数学方向的端到端记录后签署。
