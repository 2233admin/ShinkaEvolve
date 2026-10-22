# 开源“自动科学家”项目调研

调研日期：2026-10-22。只读一手来源：各仓库的 README、LICENSE、入口代码，以及项目自己的论文。没有安装或运行任何项目。Stars、最后提交时间、许可证类型通过 GitHub API（`gh api repos/<repo>` 和 `repos/<repo>/commits?per_page=1`）在当天取得。最后提交时间指默认分支上最新一次提交的 committer 时间。

标记约定：
- **【推断】**：根据看到的代码或文档推理得出，没有直接证据。
- **【未核实】**：没查到，或者只看了 README 说法、没看代码。

## 0. 结论先行

想直接拿来用、又不自己从头搭的话，优先看下面三个：

1. **SakanaAI/AI-Scientist-v3（The Agentic Scientist，2026-10 发布）**：唯一把“搜索阶段的评估器”和“最终评估”都做成 sealed（封存）的完整论文流水线。论文阶段会拿稿件和封存的证据逐条核对。整体结构几乎就是我们“冻结评估器 + 锁箱 + 论文”的通用版。
2. **ResearAI/DeepScientist**：明确支持把一篇论文或一个 GitHub 仓库当起点，先复现 baseline，再迭代实验、出论文草稿。底层用 Codex 或 Claude Code 执行，官方建议在 Windows 上用 WSL2。完整性约束只写在 prompt 层面。
3. **ChicagoHAI/NeuriCo**：有 sealed scoring（评估代码和 `data/.test/` 被移出 agent 工作区，并做 SHA-256 清单校验），并提供 LaTeX 论文模板。设计与我们最接近，但每轮迭代都在同一份隐藏测试集上打分，存在“反复看测试集”的风险。

量化方向另有一条要特别注意：**microsoft/RD-Agent 的 Qlib 场景**拿测试期回测指标来决定是否替换 SOTA。这正是 2509.08713 所说的 post-hoc selection bias（事后选择偏差），见 §2.4。

## 1. 对比表

“现有仓库起步”一列指能否把一个已有的 GitHub 项目或参考实现直接当作实验起点。

| 项目 | 端到端链路 | 主要输入 | 现有仓库起步 | 输出 | 许可证 | 最后提交 | Stars | 运行需求 | Windows | 完整性要点 |
|---|---|---|---|---|---|---|---|---|---|---|
| SakanaAI/AI-Scientist-v3 | 任务→探索→消融→**封存最终评估**→事后分析→论文（可选 review） | 任务目录（`task.md` + evaluator + `program_0/`） | 可以：把参考代码包装成 `program_0` 并写 evaluator；仅凭 brief 自动起草属实验功能 | LaTeX→PDF + manifest | Apache-2.0 | 2026-10-18 | 463 | Docker + uv；Claude/Codex/Gemini 等 CLI；GPU 只在 GPU 预设下需要 | 【推断】需 WSL2 | 封存搜索/最终评估器；`results_tree_sha256`；任务快照不可变 |
| ResearAI/DeepScientist | 论文/仓库→复现 baseline→持续实验→论文草稿 | 研究目标、baseline 链接、论文链接、本地路径 | **可以（README 明写）** | 报告、图、论文草稿（本地 PDF/LaTeX） | Apache-2.0 | 2026-06-28 | 3387 | Node/npm + Python 3.11+；Codex/Claude Code/Kimi/OpenCode | 原生实验性，**官方推荐 WSL2** | 评估契约只是 markdown 模板，【推断】不强制 |
| ChicagoHAI/NeuriCo | 想法→找资源→实验→（AutoResearch 迭代）→论文 | YAML/Markdown 想法 | 部分：`--skip-resource-finder` 用已准备的工作区；可 bootstrap 现有工作区 | 代码、结果、日志、分数、LaTeX 论文 | Apache-2.0 | 2026-10-20 | 161 | Claude Code/Codex/Gemini CLI（OAuth）；Docker 或 uv；可选 Modal | 【推断】需 WSL2 | **sealed scorer + SHA-256 清单 + 隐藏 `data/.test/`** |
| microsoft/RD-Agent | 假设→因子/模型代码→Qlib 回测→反馈循环；或读论文/研报抽取因子与模型 | 场景命令；研报目录；论文 URL；Kaggle 赛题 | 可以读论文/研报；不能直接接任意仓库【推断】 | 代码 + 日志 + UI，**不写论文**【推断】 | MIT | 2026-09-23 | 15019 | **仅 Linux**，Docker（免 sudo），Python 3.10/3.11，Qlib 数据 | 只能 WSL2 | **SOTA 决策用测试期回测指标**（见 §2.4） |
| SamuelSchmidgall/AgentLaboratory | 文献→实验→报告 | 研究主题 + yaml 任务笔记 | 不直接支持【推断】 | LaTeX（可编译 PDF） | MIT | 2025-08-20 | 5908 | OpenAI/DeepSeek API；pdflatex 可选 | 【推断】WSL 可行 | **被 2509.08713 测过**；README 示例笔记直接要求在测试集上最大化 |
| SakanaAI/AI-Scientist（v1） | 模板→想法→实验（Aider 改代码）→LaTeX 论文→LLM 审稿 | 模板目录（experiment.py/plot.py/prompt.json/seed_ideas.json/template.tex） | 可以：自己写模板包住参考代码 | LaTeX/PDF + 审稿 JSON | AI Scientist Source Code License（RAIL 派生） | 2025-12-19 | 14772 | Linux + NVIDIA CUDA；texlive-full；多家 LLM API | 【推断】WSL2 + CUDA | 评估写在 agent 自己改的 experiment.py 里；失败 run 目录会被删除 |
| SakanaAI/AI-Scientist-v2（参照） | 主题→想法→BFTS 实验→论文 | 主题 md → 想法 JSON；可 `--load_code` | 部分（`--load_code`） | LaTeX/PDF | 同上 | 2025-12-19 | 7383 | Linux + NVIDIA CUDA | 【推断】WSL2 | **被 2509.08713 测过**；此前评审已知问题见正文 |
| JinyanSu1/Agentic-AI-Scientist | 辩论式选题→pilot→新颖性核查→开发循环（lock/revise/abandon）→LaTeX 论文 + 审稿 | 种子论文 PDF/检索词、topic md、或现成想法 JSON | 未提供【推断】 | tectonic 编译的 PDF + 审稿 | AI Scientist Source Code License（继承） | 2026-08-08 | 8 | Linux + NVIDIA；Codex/Claude Code；默认 Sakana 内部 **fugu** 网关 | 【推断】WSL2 | 每轮 verdict 落盘，`research_wiki.jsonl`；只有 fugu 路径真正跑通过 |
| ltjed/freephdlabor | Manager 调度 想法/实验/写作/审稿/校对 agent | `--task` 文字 | 可替换 RunExperimentTool 接自定义实验【推断】 | LaTeX 论文 | MIT | 2026-09-03 | 729 | Python 3.11、conda、建议 CUDA GPU；OpenAI/Anthropic/Google | 【推断】WSL2 | 实验引擎直接调用内置的 AI-Scientist-v2 BFTS；记录全部 LLM 调用 |
| Technion-Kishony-lab/data-to-paper | 数据→假设→文献→分析代码→解释→逐节写论文 | **带注释的数据集** | 不适用（输入是数据） | LaTeX/PDF，数字可回溯到代码行 | MIT | 2025-07-19 | 830 | pip；pandoc；LaTeX；OpenAI + Semantic Scholar key；无需 GPU | **原生 Windows（INSTALL 写了 MiKTeX）** | 数据链（data-chaining）可回溯；统计包加护栏；AI 水印 |
| HKUDS/AI-Researcher | 文献→想法→实现→验证→论文 | Level1 详细想法 / Level2 参考论文（benchmark JSON 格式） | 会自己挑选代码仓库；接自有仓库【未核实】 | 论文 | **无 LICENSE 文件** | 2025-10-16 | 5832 | Docker 镜像 + GPU；LiteLLM（默认 OpenRouter Gemini） | 【推断】WSL2 + Docker | 无专门机制【未核实】 |
| allenai/codescientist | 论文 + 代码块变异出想法→容器内实验（每想法约 5 次）→报告 + 元分析 | arXiv 链接 + codeblock 库 | 只能通过 codeblock 接入【推断】 | 报告 PDF | Apache-2.0 | 2026-03-25 | 350 | Ubuntu；实验跑在 **Modal.com 云容器**；OpenAI + Anthropic；LaTeX | “修改后可能可用” | 多次独立尝试 + 元分析；原始日志公开；成本硬上限 |
| liugangcode/deepevolve | Deep Research 出想法 + AlphaEvolve 式多文件进化 | `initial_code/` + `info.json` + 接口函数 | **可以**（把代码放进 `initial_code`） | 最优代码 + checkpoint，**不写论文** | **无 LICENSE 文件** | 2025-10-11 | 144 | Python 3.9、OpenAI Agents SDK | 【推断】可行 | 评估接口放在被进化的代码目录里【推断：可被改】 |
| algorithmicsuperintelligence/openevolve | AlphaEvolve 式程序进化 | `initial_program.py` + `evaluator.py` | 可以（evaluator 独立） | 程序 + checkpoint，不写论文 | Apache-2.0 | 2026-10-22 | 7623 | Python 3.10+；任意 OpenAI 兼容 API 或 Claude Code CLI | **README 写明支持 Windows** | 评估器是独立文件（与 ShinkaEvolve 同范式） |
| karpathy/autoresearch | agent 改 `train.py`→跑 5 分钟→按 val_bpb 保留或回滚，无限循环 | `program.md`（人写）+ 固定 `prepare.py` | 范式可迁移；本仓库固定为 nanochat | `results.tsv` + git 分支，不写论文 | 无 LICENSE 文件；README 写 MIT | 2026-03-26 | 98361 | 单张 NVIDIA GPU（测试用 H100）；uv；任意 coding agent | 有 Windows fork | 评估冻结靠指令，不靠强制；所有尝试（含丢弃、崩溃）都记账 |
| du-nlp-lab/MLR-Copilot | 论文→想法→实现→执行（人类反馈） | 研究想法文件 + `workspaces/<task>` 起始代码 | **可以**（起始代码目录） | 想法 + 执行结果，不写论文 | 无 LICENSE 文件 | 2025-03-30 | 71 | Docker；Claude/OpenAI key | 【推断】WSL2 + Docker | 全量 `full_log.jsonl` |
| zhu-minjun/Researcher（CycleResearcher） | 主题 + 参考文献→**生成整篇论文（实验结果是编造的）**；另有 Reviewer | 主题 + BibTeX | 不适用 | 论文文本 | CycleResearcher License（基于 Mistral AI Research License） | 2026-03-05 | 407 | 本地 12B/14B 模型【推断：需 GPU】 | 【推断】 | **论文自述实验结果为 fabricated** |

剔除（没有可运行的端到端代码，或不做实验）：
- **Agent4Science-UTokyo/Jr.AI-Scientist**：README 写明暂停开源，仓库里只有 PDF 和生成的论文。
- **JinheonBaek/ResearchAgent**：只生成想法。
- **jataware/open-coscientist** 等 AI co-scientist 复刻：只做假设的生成、排序和演化，不跑实验。

## 2. 分项

### 2.1 SakanaAI/AI-Scientist-v3（The Agentic Scientist）
来源：[README](https://github.com/SakanaAI/AI-Scientist-v3/blob/main/README.md)、[docs/evaluation_evidence.md](https://github.com/SakanaAI/AI-Scientist-v3/blob/main/docs/evaluation_evidence.md)、[docs/task_drafting.md](https://github.com/SakanaAI/AI-Scientist-v3/blob/main/docs/task_drafting.md)、[examples/grpo/evaluate.py](https://github.com/SakanaAI/AI-Scientist-v3/blob/main/examples/grpo/evaluate.py)、论文 [arXiv 2610.18933](https://arxiv.org/abs/2610.18933)。

- **链路**：从 `task.md` 出发，默认六个阶段：任务起草 → investigation（想法、实验、评估循环）→ 消融（与 baseline、消融变体比较）→ **sealed final evaluation**（在封存环境中重新打分）→ 事后分析 → 论文。之后可选 review 和 revision。每个阶段在独立容器里运行，只读取 run 本地的快照（README「How it works」）。
- **输入**：一个任务目录。官方示例都自带 evaluator，起草阶段只做快照；只给 brief、由系统自动起草 evaluator 和 rubric 属于实验功能，README 说明大任务上不稳定（README「8. Draft a task from a research brief」）。
- **能否以现有仓库起步**：可以，但要手工把参考代码整理成 `program_0/` 并写好 evaluator（参照 `examples/circpack/` 的结构）【推断：没有“输入 GitHub URL”的入口】。
- **输出**：`results/<task>/<agent>/<run_id>/` 下按阶段分目录，`task/` 是不可变的任务快照，`paper/` 里有 PDF、manifest 和可选的 AI 审稿（README「What you get」）。
- **运行**：Docker 镜像加 uv；通过 Headless CLI 驱动 claude/codex/cursor/opencode/gemini/pi。GPU 只在 GPU 预设下需要，用 `nvidia-smi` 校验。默认示例 circpack 只用 CPU（README「Install」「Recommended starting setup」）。安装步骤含 `source .venv/bin/activate`，【推断】Windows 上要走 WSL2。
- **完整性**：
  - 搜索阶段和最终阶段的评估器都是 sealed，两者共用一个只读的输出校验器。候选程序自己报告的诊断数字明确标为 **unverified**（evaluation_evidence.md）。
  - final manifest 通过 `results_tree_sha256` 封存证据，并记录主机侧测得的计算用量（同上）。
  - 候选子进程拿不到评估器的凭据和任务路径（task_drafting.md）。
- **2509.08713 对照**：v3 没被该文测过。
  - 结构上，它用独立的最终评估阶段把“选优”和“报告”分开，比 v2 好。
  - 但 `examples/grpo/evaluate.py` L72–84、L266–268 读的是候选程序**自己写入**的 `_results.json` 里的 `final_accuracy`。只有打开 `investigation.execution.harness_evaluated_attempts=true` 时才由 harness 打分（README 示例 7）。
  - 搜索阶段和最终评估是否使用不同的数据或种子：**【未核实】**。

### 2.2 ResearAI/DeepScientist
来源：[README](https://github.com/ResearAI/DeepScientist/blob/main/README.md)、[eval-contract-template.md](https://github.com/ResearAI/DeepScientist/blob/main/src/skills/scout/references/eval-contract-template.md)、[comparability-contract.md](https://github.com/ResearAI/DeepScientist/blob/main/src/skills/baseline/references/comparability-contract.md)、论文 [OpenReview（ICLR 2026）](https://openreview.net/forum?id=cZFgsLq8Gs)。

- **链路**：本地长期运行的“研究工作室”，覆盖 baseline 复现、多轮实验，直到产出可用于论文的结果。用 Findings Memory、贝叶斯优化和 Research Map 把每轮结果变成下一轮的起点（README 开头）。
- **输入、现有仓库**：README「Start a real project from a paper or a research question」原文写：“feed it a core paper, a GitHub repository, or a natural-language research objective”。会恢复仓库、准备环境、处理依赖，并记录哪些步骤可信。**每个 quest 都是一个真实的 git 仓库**。
- **输出**：图、报告、论文草稿；支持本地 PDF/LaTeX 编译。
- **运行**：`npm install -g @researai/deepscientist` 后运行 `ds --here`；runner 可选 Codex、Claude Code、Kimi Code、OpenCode；Python 3.11+。
- **Windows**：README 写明 Linux/macOS 完整支持，原生 Windows 为实验性，强烈推荐 WSL2。
- **完整性**：
  - scout 阶段生成一份 markdown 评估契约，内容包括 dataset、split_contract、官方评估路径、主指标、公平比较规则；baseline 阶段还有可比性契约，记录 split、评估脚本、源 commit。
  - 这些都是 skill/prompt 层面的约定，**没看到代码层的封存或哈希【推断】**。
  - 是否把测试集用于选优：**【未核实】**。该项目没被 2509.08713 测过。

### 2.3 ChicagoHAI/NeuriCo
来源：[README](https://github.com/ChicagoHAI/NeuriCo/blob/main/README.md)、[docs/AUTORESEARCH.md](https://github.com/ChicagoHAI/NeuriCo/blob/main/docs/AUTORESEARCH.md)、[src/core/scoring_seal.py](https://github.com/ChicagoHAI/NeuriCo/blob/main/src/core/scoring_seal.py)、[templates/agents/paper_writer.txt](https://github.com/ChicagoHAI/NeuriCo/blob/main/templates/agents/paper_writer.txt)。

- **链路**：有三种模式。
  - Standard：一次走完找资源、设计并执行实验、分析、写论文草稿。
  - AutoResearch：先建立已打分的 baseline，再循环“提一个改动 → 运行 → 用 sealed 评估协议打分 → 有提升才保留”。
  - HITL：在 AutoResearch 基础上加一个人工决策点。
- **输入**：YAML、Markdown 或 IdeaHub 想法（`./neurico submit idea.yaml`）。
- **现有仓库**：`--skip-resource-finder` 可以直接用已经准备好的工作区；`--bootstrap-autoresearch-baseline` 给已有工作区补上评分并建立 baseline。没有“给一个 GitHub URL”的入口【推断】。
- **输出**：代码、结果、日志、分数、可选论文。论文是模块化 LaTeX 工程，用 pdflatex 加 bibtex 编译，模板可选 neurips/icml/acl/ams（paper_writer.txt L113–157；README `--paper-style`）。
- **运行**：Claude Code、Codex 或 Gemini CLI（OAuth 登录，不需要 API key）；Docker 或本地 uv。文献检索和评分契约校验需要 S2 key 加 OpenRouter 或 OpenAI key。计算后端可选 local、Modal、Slurm。
- **完整性**：
  - `scoring_seal.py` L2–5：agent 修改实验期间，评估器内部文件会被移出工作区。L18–28 列出封存路径，包括 `scoring/eval.py`、`scoring/targets.json`、`data/.test/`。L36–47 计算 SHA-256 清单。L111–119 校验：清单不符就报 “Sealed evaluator payload changed”。L139–146 清除泄漏回工作区的评估文件。
  - 风险：AutoResearch 每轮都用 sealed evaluator 决定是否保留（AUTORESEARCH.md L4–6、L96）。如果打分用的是 `data/.test/`，就等于在同一份隐藏测试集上反复选优（adaptive overfitting）。代码里没看到试验计数或多重检验校正【推断】。选优是否另有验证集：**【未核实】**。

### 2.4 microsoft/RD-Agent（含 R&D-Agent-Quant）
来源：[README](https://github.com/microsoft/RD-Agent/blob/main/README.md)、[conf_baseline.yaml](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml)、[qlib/developer/feedback.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/developer/feedback.py)、论文 [R&D-Agent-Quant arXiv 2505.15155](https://arxiv.org/abs/2505.15155)（NeurIPS 2025）、[R&D-Agent arXiv 2505.14738](https://arxiv.org/abs/2505.14738)。

- **链路**：
  - 量化：`rdagent fin_quant`、`fin_factor`、`fin_model` 在 Qlib 上自循环地提出并实现因子或模型；`fin_factor_report --report-folder` 从研报中抽取因子。
  - 通用：`rdagent general_model <论文 URL>` 从论文中抽取模型并实现；`data_science --competition` 处理 Kaggle/MLE-bench 赛题。
- **输出**：代码、日志、Web UI（`rdagent server_ui`）。README 没提写论文【推断：不产出论文】。
- **运行**：README 写明 “currently only supports Linux”；Docker 必须能免 sudo 运行；Python 3.10/3.11；需要 Qlib 的 cn_data。Windows 上只能走 WSL2。
- **完整性（重点）**：
  - `conf_baseline.yaml` 切分为 train 2008–2014、valid 2015–2016、test 2017 年至今，而 `port_analysis_config.backtest.start_time` 用的正是 `test_start`。
  - `feedback.py` L17–21 定义的 `IMPORTANT_METRICS` 包括 IC、`1day.excess_return_with_cost.annualized_return` 和 `max_drawdown`；L70–73 拿当前结果和 SOTA 结果比较；L108 由 LLM 输出 “Replace Best Result” 作为决策。
  - 也就是说，**是否替换 SOTA 由测试期回测指标决定**。这和 2509.08713 第四条“内部奖励机制能看到测试集评估，并系统性偏好测试表现好的实验”属于同一类问题。这里是代码证据，不是该文对 RD-Agent 的测试结果。
- **对我们的意义**：领域最接近（A 股因子/模型），但如果接入我们的体系，要把它的反馈口改成只看样本内（我们的 `probe`），并把 test 段挪进锁箱。

### 2.5 SamuelSchmidgall/AgentLaboratory
来源：[README](https://github.com/SamuelSchmidgall/AgentLaboratory/blob/main/README.md)、论文 [arXiv 2501.04227](https://arxiv.org/abs/2501.04227)、AgentRxiv [arXiv 2503.18102](https://arxiv.org/abs/2503.18102)。

- **链路**：三个阶段：文献综述、实验、报告写作，用到 arXiv、Hugging Face、Python、LaTeX（README「How does Agent Laboratory work」）。
- **入口**：`python ai_lab_repo.py --yaml-location experiment_configs/MATH_agentlab.yaml`，yaml 里可以开 copilot 模式。
- **运行**：模型支持 o1、gpt-4o、o3-mini、deepseek-chat；pdflatex 可选（`--compile-latex false`）；进度存在 `state_saves` 里；许可证 MIT。
- **完整性**：
  - README Tip #1 的示例任务笔记原文是 “maximizing performance on the test set of MATH” 和 “You must evaluate on the entire 500 test questions of MATH”。用法上就是直接在测试集上优化。
  - 2509.08713 实测了该系统，发现：会自己合成数据集或对数据集降采样且不披露；会替换用户指定的指标；内部奖励机制能看到测试集评估，并偏好测试表现好的实验。
- 最后提交 2025-08-20，此后无更新。

### 2.6 SakanaAI/AI-Scientist（v1）
来源：[README](https://github.com/SakanaAI/AI-Scientist/blob/main/README.md)、[LICENSE](https://github.com/SakanaAI/AI-Scientist/blob/main/LICENSE)、[ai_scientist/perform_experiments.py](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_experiments.py)、论文 [arXiv 2408.06292](https://arxiv.org/abs/2408.06292)。

- **链路**：基于模板的想法生成 → Aider 修改 `experiment.py` 并运行（最多 5 次）→ 改 `plot.py` 出图 → 写 `notes.txt` → LaTeX 论文 → `perform_review` 给出 Overall 分和 Accept/Reject。
- **现有仓库**：可以。自己写一个模板（`experiment.py --out_dir`、`plot.py`、`prompt.json`、`seed_ideas.json`、`latex/template.tex`）把参考代码包进去，README「Making Your Own Template」有说明。
- **运行**：Linux + NVIDIA CUDA，texlive-full；支持 OpenAI、Anthropic（含 Bedrock、Vertex）、DeepSeek、OpenRouter、Gemini；README 要求容器化运行。
- **许可证**：LICENSE 文件首行是 “The AI Scientist Source Code License”，基于 Responsible AI Source Code License v1.1，其中第 3.2(e) 条要求在生成的论文中显著声明是机器生成的。
- **完整性（代码证据）**：
  - `perform_experiments.py` 里，评估逻辑写在 agent 要修改的同一个 `experiment.py` 中（`run_experiment` 读取它写出的 `final_info.json`），评估器与被改代码没有分离。
  - 失败或超时的 run 目录会被 `shutil.rmtree` 删除，只保留 `run_i.py` 代码副本，失败尝试的输出因此缺失。
  - 未被 2509.08713 测过。

### 2.7 SakanaAI/AI-Scientist-v2（参照，已评审过）
来源：[README](https://github.com/SakanaAI/AI-Scientist-v2/blob/main/README.md)、论文 [arXiv 2504.08066](https://arxiv.org/abs/2504.08066)。

- **链路**：先用 topic md 经 `perform_ideation_temp_free.py` 生成想法 JSON，再用 `launch_scientist_bfts.py`（BFTS，参数在 `bfts_config.yaml`）跑实验，可加 `--load_code` 用代码片段起步，最后写论文。
- **运行**：Linux + NVIDIA CUDA；单次实验约 15–20 美元，写作约 5 美元（README FAQ）。
- **已知问题**：许可证、清理进程时按关键字杀掉宿主进程、writeup 的输入约定，都见此前评审。2509.08713 实测：存在测试集选优、未披露的合成数据或降采样、指标替换。

### 2.8 JinyanSu1/Agentic-AI-Scientist
来源：[README](https://github.com/JinyanSu1/Agentic-AI-Scientist/blob/main/README.md)。

- **链路**：AI-Scientist-v2 的重构 fork，用一个开放式 research loop 取代 BFTS，步骤如下：
  1. Proposer、Challenger、Evaluator 三方辩论选题；
  2. 多个候选各跑一次低成本 pilot，按实验信号选胜者；
  3. 深度新颖性核查；
  4. 开发循环：每次尝试后 Evaluator 判 lock、revise 或 abandon，有安全上限；
  5. 写作：多源引用加 tectonic 编译 LaTeX，再做 LLM+VLM 审稿。
- **入口**：`python run_research_loop.py --seed-papers <PDF或检索词> --model fugu`；也可以用 `--start-idea-file` 从已有想法开始。
- **执行**：实验由 Codex CLI（`--dangerously-bypass-approvals-and-sandbox`）或 Claude Code（`--dangerously-skip-permissions`）执行。
- **可用性风险**：README「What has actually been tested」写明**只有 `--model fugu` + `--worker codex --codex-profile fugu` 跑通过**。fugu 是 Sakana 的内部网关（需 `SAKANA_API_KEY`），其他后端“never been run end to end”。
- **完整性**：
  - 每轮都把 `round_XX_outcome.json` 落盘；`development_decision.json` 记录论文来自干净的 lock 还是触发了安全上限；跨 run 的 `research_wiki.jsonl` 记录所有试过的想法（locked、abandoned、eliminated、capped）。这对试验计数有用。
  - 示例故意保留了 Reject 的自评和被推翻的结论，并对人工修改做了披露。
  - 许可证继承 AI Scientist Source Code License，并强制要求披露。
  - 2509.08713 未测过；写作部分来自 v2【推断：选优是否看测试集取决于 Evaluator 的输入，未核实】。
- Stars 8，属于个人项目。

### 2.9 ltjed/freephdlabor
来源：[README](https://github.com/ltjed/freephdlabor/blob/main/README.md)、[toolkits/run_experiment_tool.py](https://github.com/ltjed/freephdlabor/blob/main/freephdlabor/toolkits/run_experiment_tool.py)、论文 [arXiv 2510.15624](https://arxiv.org/abs/2510.15624)。

- **链路**：ManagerAgent 调度 Ideation、Experimentation、Writeup（LaTeX）、Reviewer、Proofreading 等 agent。入口是 `python launch_multiagent.py --task "<想法>"`，支持 `--resume` 和回调服务器，可以人工中途插话。
- **实验引擎**：`run_experiment_tool.py` L10–13、L74、L224–229、L264 表明 RunExperimentTool 直接调用仓库内置的 AI-Scientist-v2 `launch_scientist_bfts.py`，并读取 `logs/*-run/research_summary.json`。因此 **v2 的实验引擎问题它都继承**【推断：2509.08713 的结论适用】。
- **缓解措施**：L23–25 要求先用 IdeaStandardizationTool 把模型和数据集固定下来（工具描述里写着 “C4/MMLU not synthetic data”），这是针对“偷换数据集”的部分缓解。
- **其他**：所有 LLM 调用记入 `agent_llm_calls.jsonl`；README 建议在自己的领域把 RunExperimentTool 换成自定义工具。运行需要 Python 3.11+、conda，建议 CUDA GPU。许可证 MIT。

### 2.10 Technion-Kishony-lab/data-to-paper
来源：[README](https://github.com/Technion-Kishony-lab/data-to-paper/blob/main/README.md)、[INSTALL.md](https://github.com/Technion-Kishony-lab/data-to-paper/blob/main/INSTALL.md)、论文 [NEJM AI 10.1056/AIoa2400555](https://doi.org/10.1056/AIoa2400555)、[arXiv 2404.17605](https://arxiv.org/abs/2404.17605)。

- **链路**：从带注释的原始数据出发，依次做探索、文献检索、提出假设、写并调试分析代码、解释结果，最后逐节写完整论文。可全自动，也可在 Copilot App 里由人引导，支持回退、录制回放和 API 成本追踪。
- **完整性亮点**：
  - “data-chained” 稿件中**每个数字都能点击回溯到产生它的代码行**。
  - 用多重护栏覆盖标准统计包，减少常见的 LLM 编码错误。
  - 生成的论文带 AI 水印。
- **局限**：README 说明目前只适合简单研究目标、简单数据集上的统计假设检验。不做 ML 训练，也不接代码仓库。
- **运行**：`pip install data-to-paper`；需要 pandoc 和 LaTeX，INSTALL 分别给了 Debian、macOS 和 **Windows（MiKTeX）** 的做法；需要 OpenAI 和 Semantic Scholar key；无需 GPU。
- 最后提交 2025-07-19。未被 2509.08713 测过。

### 2.11 HKUDS/AI-Researcher
来源：[README](https://github.com/HKUDS/AI-Researcher/blob/main/README.md)、论文 [arXiv 2505.18705](https://arxiv.org/abs/2505.18705)（NeurIPS 2025）。

- **输入**：两个级别。Level 1 是详细的想法描述；Level 2 只给参考论文，由系统自己出想法（README 开头）。
- **入口**：命令行是 `run_infer_plan.py` 或 `run_infer_idea.py --instance_path ../benchmark/final/<category>/<id>.json`，输入是 benchmark 格式的 JSON；论文由 `paper_agent/writing.py --research_field --instance_id` 生成；另有 Web GUI `web_ai_researcher.py`。
- **资源选择**：README 说 Resource Filter 会挑选“well-maintained code implementations”和数据集，Idea Generator 也会利用代码仓库。能否指定自己的仓库：**【未核实】**。
- **运行**：必须用 Docker 镜像 `tjbtech1/airesearcher:v1`；`.env` 里配置 GPU（`GPUS='"device=0"'`）；通过 LiteLLM 调模型，默认 `openrouter/google/gemini-2.5-pro-preview-05-20`；还需要 `GITHUB_AI_TOKEN`。
- **许可证**：仓库根目录**没有 LICENSE 文件**，GitHub API 返回 null。
- 完整性机制：README 未提及【未核实】。最后提交 2025-10-16。

### 2.12 allenai/codescientist
来源：[README](https://github.com/allenai/codescientist/blob/main/README.md)、论文 [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.692/)。

- **链路**：用 LLM 作为变异算子，组合“论文 + 代码示例”产生想法；Experiment Builder 在容器里编写、运行、调试实验；写报告。通常每个想法做约 5 次独立尝试，并对这些尝试做元分析。论文主要报告的是人在环模式，全自动模式效率较低。
- **换领域**：粘贴 arXiv 链接，再往 `codeblocks/` 里加代码块。
- **运行**：实验跑在 **Modal.com 云容器**，需要账号；需要 OpenAI 和 Anthropic key；LLM 成本有硬上限 `LLM_COST_HARD_LIMIT`；写报告需要 pdflatex。开发环境是 Ubuntu，Windows “may work with modification”。
- **完整性**：多次独立尝试加元分析，能降低“挑一次好结果”的偏差；原始实验代码、日志和外部评审打分都已公开。许可证 Apache-2.0。

### 2.13 liugangcode/deepevolve
来源：[README](https://github.com/liugangcode/deepevolve/blob/main/README.md)、[coder.py](https://github.com/liugangcode/deepevolve/blob/main/coder.py)、论文 [arXiv 2510.06056](https://arxiv.org/abs/2510.06056)。

- **链路**：Deep Research（联网检索、出研究草案）加 AlphaEvolve 式进化，支持多文件代码，并自动调试。入口 `python deepevolve.py query="..." problem="<examples下的目录>"`；结果写到 `{workspace}/{problem}/{checkpoint}/best`。
- **现有仓库**：把起始代码放进 `initial_code/`，写好 `info.json` 和 `deepevolve_interface.py`（返回带 `combined_score` 的指标字典）即可。
- **完整性**：
  - 评估接口文件就放在被进化的 `initial_code/` 里。`coder.py` 的编辑规则只约定了 `DEEPEVOLVE-BLOCK` 标记格式，没有禁止改接口文件【推断：评估器可能被 agent 修改】。
  - 不写论文。根目录无 LICENSE 文件。最后提交 2025-10-11。

### 2.14 algorithmicsuperintelligence/openevolve
来源：[README](https://github.com/algorithmicsuperintelligence/openevolve/blob/main/README.md)。

- **链路**：AlphaEvolve 的开源实现。`openevolve-run.py initial_program.py evaluator.py --config ... --iterations N`，也可用 Python API `run_evolution(program, evaluator)`。
- **评估器**：独立文件，和 ShinkaEvolve 是同一种范式。
- **运行**：Python 3.10+；任意 OpenAI 兼容 API 或 Claude Code CLI；Docker 可选。README 写了在 macOS/Windows 下需要 `if __name__ == '__main__':` 保护，说明 Windows 可用。
- 不做文献检索，也不写论文。对我们价值有限，因为 ShinkaEvolve 已经覆盖这部分。许可证 Apache-2.0。

### 2.15 karpathy/autoresearch
来源：[README](https://github.com/karpathy/autoresearch/blob/master/README.md)、[program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)。

- **链路**：
  - 人只改 `program.md`，agent 只改 `train.py`。`prepare.py` 里放数据、tokenizer 和 `evaluate_bpb`，规定为只读。
  - 每次固定训练 5 分钟，看 val_bpb，变好就保留 commit，否则 `git reset`，“NEVER STOP”。
  - 所有尝试，包括 discard 和 crash，都写进 `results.tsv`（commit、val_bpb、显存、状态、描述）。
- **运行**：单张 NVIDIA GPU（测试用 H100），uv；agent 可以是 Claude 或 Codex 等任意 coding agent。README 列出 Windows fork `jsegov/autoresearch-win-rtx`。
- **完整性**：
  - 评估器“冻结”只靠 `program.md` 里的指令（“Do not modify”），**没有技术上的强制**。
  - 只有一个指标 val_bpb，既用于选优也用于报告，没有独立的测试集。
  - 好处是记账完整，试验数可直接统计。
- 不写论文。根目录无 LICENSE 文件，README 写 MIT。

### 2.16 du-nlp-lab/MLR-Copilot
来源：[README](https://github.com/du-nlp-lab/MLR-Copilot/blob/main/README.md)、论文 [arXiv 2408.14033](https://arxiv.org/abs/2408.14033)。

- **链路**：从论文出发，生成假设和实验计划，基于检索到的原型代码实现，然后执行，过程中有人类反馈。
- **现有仓库**：把起始代码放进 `workspaces/<task>`，想法放进 `problems/<task>`，然后运行 `bash run_demo.sh <task> <llm>`。
- **运行**：Docker 镜像 `tortcode/nlp-coresearcher`。
- **记录**：完整日志在 `logs/<task>/<ts>/agent_log/full_log.jsonl`。
- 不写论文。无 LICENSE 文件，README 说明部分组件来自 MLAgentBench（MIT）和 Prompt2Model（Apache-2.0）。最后提交 2025-03-30，已基本停更。

### 2.17 zhu-minjun/Researcher（CycleResearcher / CycleReviewer / DeepReviewer）
来源：[readme](https://github.com/zhu-minjun/Researcher/blob/main/readme.md)、[LICENSE.md](https://github.com/zhu-minjun/Researcher/blob/main/LICENSE.md)、论文 [arXiv 2411.00816](https://arxiv.org/abs/2411.00816)。

- **用法**：`pip install ai_researcher`，然后 `CycleResearcher(model_size="12B").generate_paper(topic, references)`；另有 CycleReviewer（8B）、DeepReviewer（14B）和 AIDetector。
- **关键事实**：论文脚注和局限性部分明确写着 “the experimental results generated by CycleResearcher in this work are fabricated and do not represent real experimental data”。附录 C.2 禁止把它用于真实审稿，也禁止不披露地投稿。**它不做真实实验，不能满足“跑真实实验”的需求。**
- **可取之处**：只有 Reviewer 可以当作论文的自动审稿或打分工具。
- **许可证**：CycleResearcher License，基于 Mistral AI Research License，使用前要提交身份信息。

## 3. 2509.08713 的四类问题，对照到各项目

来源：[arXiv 2509.08713](https://arxiv.org/abs/2509.08713)（Luo, Kasirzadeh, Shah）“Key findings”。该文只实测了 Agent Laboratory 和 AI Scientist v2，结论如下：
1. **基准选择**：倾向选 SOTA 已经很强的基准，或者默认选列表前几个。
2. **数据**：“Neither system peeks at the test data”，但两者都偶尔自己生成合成数据，或对给定数据降采样，而且不在论文中披露。
3. **指标**：没发现故意挑指标，但指标选择很随意，受任务描述里的排列顺序影响，有时会替换用户指定的指标。
4. **事后选择偏差**：内部奖励机制能看到测试集评估，并系统性偏好测试表现好的实验。

该文建议投稿时附上完整的 trace 日志和代码。

| 项目 | 证据性质 | 适用情况 |
|---|---|---|
| AgentLaboratory、AI-Scientist-v2 | 该文实测 | 第 2、3、4 条都发现了 |
| freephdlabor | 代码直接调用 v2 BFTS | 【推断】第 4 条随 v2 引擎继承；第 2 条有部分缓解（IdeaStandardizationTool） |
| Agentic-AI-Scientist | v2 fork，BFTS 已替换 | 【未核实】 |
| RD-Agent（Qlib） | 代码证据 | 第 4 条同类：SOTA 替换看的是测试期回测指标 |
| AI-Scientist v1 | 代码证据 | 评估和被改代码不分离；失败尝试的输出被删 |
| karpathy/autoresearch | 文档证据 | 单一验证指标同时用于选优和报告；冻结只靠指令 |
| NeuriCo | 代码证据 | 封存做得好；【推断】每轮迭代都在同一份隐藏测试集上打分 |
| AI-Scientist-v3 | 代码与文档证据 | 封存最终评估，结构上缓解第 4 条；搜索和最终是否不同数据【未核实】 |
| CycleResearcher | 论文自述 | 实验结果本来就是编造的 |
| data-to-paper | 文档证据 | 数字可追溯，是第 2、3 条的反向参照；未被测过 |

## 4. 排序推荐

### (a) 拿一个参考仓库开跑实验
1. **DeepScientist**：唯一在 README 里明确写“喂一个 GitHub 仓库”的成熟项目（3.4k stars，2026-06 仍在更新）。会先复现 baseline 并记录哪些步骤可信，每个 quest 是一个 git 仓库。runner 就是我们已经在用的 Codex/Claude Code，官方推荐 WSL2。
2. **NeuriCo**：可以把参考仓库预先放进工作区，再用 `--skip-resource-finder` 或 bootstrap 起步。好处是 sealed scorer 和 LaTeX 论文模板都是现成的；2026-10-20 仍在更新。
3. **AI-Scientist-v3**：需要手工把参考仓库整理成 `program_0/` 加 evaluator，前期成本最高，但之后的消融、最终评估、论文全自动完成。项目很新（2026-10-11 创建，463 stars），要预期会遇到早期问题。

量化专用的备选是 **RD-Agent**：`general_model <论文URL>` 和 `fin_factor_report` 能从论文或研报直接抽取模型和因子并在 Qlib 上跑。但只支持 Linux，且选优看测试期指标，要先改掉这一点。

### (b) 写论文
1. **AI-Scientist-v3**：论文阶段读取封存的证据并逐条核对稿件；可用 `execution.phases=paper` 对已有 run 目录单独重写论文；`agsci review` 能给任意 PDF 审稿。
2. **data-to-paper**：数字可回溯到代码行，这一点最接近我们 notebook/ledger 的要求；原生支持 Windows，无需 GPU。只适合统计型研究，可借鉴它的写作流程和 data-chaining 机制。
3. **NeuriCo 的 paper_writer**：模板化 LaTeX（neurips/icml/acl/ams）、模块化章节、强制编译通过，写作 prompt 可以单独抽出来用。

补充：Agentic-AI-Scientist 的写作模块（多源引用、写作前先读同领域论文定文风、tectonic 编译、VLM 审稿）质量不错，但默认依赖 fugu 网关。CycleResearcher 只适合当审稿器，不能当写作器。

### (c) 最适合挂到我们冻结评估器完整性层上
1. **AI-Scientist-v3**：它的结构可以和我们一一对应。

   | v3 | 我们 |
   |---|---|
   | 任务目录（evaluator + `program_0`） | scaffold |
   | sealed search evaluator | 冻结评估器 + `probe` |
   | sealed final evaluation | 锁箱 |
   | `results_tree_sha256` | 评估器和数据哈希 |

   要补的是：让 search evaluator 只返回样本内指标，final evaluation 一次性调用我们的锁箱，并把 investigation 的尝试数接入 DSR 校正。
2. **NeuriCo**：封存机制（移出工作区、SHA-256 清单、隐藏 `data/.test/`）和我们几乎同构，可以把 `scoring/eval.py` 换成我们的冻结评估器。要改的是 AutoResearch 的保留判据：从“看隐藏测试分”改成“看样本内分”，并加上试验计数。
3. **karpathy/autoresearch（作为范式）**：最简单。`program.md` 加只读的评估脚本，再加一张记录全部尝试的 `results.tsv`，指向我们的 `probe` 就能用，试验计数天然完整。缺点是没有强制、不写论文、只适合单指标场景。

**综合 top-3：AI-Scientist-v3、DeepScientist、NeuriCo。**

## 5. 未核实项汇总
- AI-Scientist-v3：搜索评估和最终评估是否使用不同的数据或种子；没有官方 Windows 说明。
- DeepScientist：是否把测试集用于选优；评估契约是否有代码层强制。
- NeuriCo：AutoResearch 选优是否另有验证集（目前只看到 `data/.test/` 被封存）。
- Agentic-AI-Scientist：Evaluator 判 lock 时看到的是哪组数据；fugu 以外的后端能否跑通（README 自述未测）。
- HKUDS AI-Researcher：能否接入指定仓库；有没有完整性机制。
- DeepEvolve：评估接口文件是否真的会被 agent 修改（只做了 grep，没读完整的 prompt 流程）。
- 所有“Windows 需 WSL2”的判断，除 data-to-paper、OpenEvolve、DeepScientist、RD-Agent、CodeScientist 之外，都是【推断】。
