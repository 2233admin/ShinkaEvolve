# 统一分析工作台：品质检查

本轮以自研工作台 `b31db504f9ca9616b5d4c8ca523cdcfdab26d8ab:shinka/webui/` 为恢复来源，保留当前研究工作区与受保护的只读后端。检查对象是实际用户路径，不是静态首页。技术自检不代表获奖，也不代替所有者视觉验收。

## 来源与适用范围

- [Awwwards Evaluation System](https://www.awwwards.com/about-evaluation/)：Design 40%、Usability 30%、Creativity 20%、Content 10%。官方同时明确优秀网站也未必获得 SOTD。这些权重不是本项目自评获奖公式。
- [Awwwards Developer Award](https://www.awwwards.com/developer-award/)：关注代码质量、现代设备互操作、移动端和无障碍。官方仍链接的开发指南标题与修订日期年代混杂，本项目不将它称为已核实的 2026 数值门槛。
- [Webby 2026/2027 Judging Criteria](https://www.webbyawards.com/judging-criteria/)：网站类别检查内容、结构导航、视觉、功能、交互、创新、整体体验。官方强调视觉应适合受众，不必追赶潮流；本项目不混用应用软件类别的额外维度。
- [FWA About](https://thefwa.com/about/about-fwa)：创新、进步、创意突破，不限定媒介或技术。[2026-10-17 制度切换公告](https://thefwa.com/news/who-will-win-the-last-ever-fwa-site-of-the-day)规定旧每日奖于 11 月 2 日退役；本项目不将旧队列分数沿用为内部获奖线。

## DO-CONFIRM：交付前暂停点

以下是项目验收要求，不冒充奖项官方评分标准。发现任一阻断项时继续修正并重验。

- [x] 自研首页、比较、13 个分析主视图和 10 个详情视图均保留；带数据检查及独立 Throughput 夹具见下表，不能用空状态代替能力。
- [x] 研究方向仍是首页，普通运行无需伪造方向；真实研究往返与普通运行比较/后退均保留身份和可复用 URL。
- [x] 研究与分析共用墨黑/Technical Paper、Inter/IBM Plex Mono、方角、分隔线及 1480px 上限；没有白色上游页面套黑壳、重复主题控制或非语义 emoji 操作。复制成功与复位标签也实际检查。
- [x] 1440/1024/390px 的填充页面无外层或 iframe 横向溢出；宽图局部滚动，窄屏纵向重排。
- [x] Chrome 原生页面缩放 200% 通过：任务自有 profile 的 Page zoom 从 1 改为 2，CSS zoom 仍为 1，innerWidth 1440→720、devicePixelRatio 1→2；选择/代码保留，外层和 iframe 均无横向溢出，检查后恢复 100%。
- [x] 键盘主路径、提示对话框、页签 Arrow/Home/End、程序选择及比较已操作回读；文字对比度抽查通过。检查范围不等于完整 WCAG 合规认证。
- [x] 加载、空库、无匹配、读取错误、越权拒绝和恢复均有真实反馈；未知数值不伪装为零，科学未定不改成成功。
- [x] 资源为本地打包；已记录冷加载、包体和页面/网络错误。完整 Plotly 带来的包体代价明确保留，不通过漏装能力缩小资源量。
- [x] 当前路径授权、程序库身份、跨目录/别名、派生文件与私有字段过滤通过真实 HTTP 回归；研究服务明确拒绝未隔离的 PDF 转换，前端说明该能力例外。

## 已观察的问题与修正

|迭代|观察|修正与复验|
|---|---|---|
|功能切片|当前白色上游分析页嵌入黑色工作区，且没有自研 Cost/Battle/Diversity/三合一 Path。|已定位自研来源；旧截图只作修改前证据，不作为最终品质通过。|
|安全评审|虚构数据库路径可读取同目录受控 meta 文本；伪造 .sqlite 文件可授权图片下载。|真实 HTTP 两项测试先失败、修复后 2 passed；发现列表的原始资料/无效库排除测试先失败、修复后连同前两项 3 passed。|
|数值边界|负分前沿被零截断，失败占位分数可能成为最佳值；缺失费用可能显示为免费。|有效分数只取正确且有限的值；失败仍计资源；明确零与未知分开。负数、失败、全空、缺失/零成本、单样本/重复嵌入均用保存的数据库检查。|
|切换归属|A 的延后渲染、全量嵌入、初始差异或图像可在 B 下完成；错误切换可留下 A 的旧详情。|运行、详情与视图请求分别校验所属版本；实际挂起 A→切 B→释放 A 后，B 的 ID、缓存、代码与图像保持不变；失败切换清空并显示错误。|
|尺寸变化|1440→1024 时数据库重载，epoch 从 1 变 2，程序选择从 a2 变空。|ResizeObserver 只重排现有数据；最终三种宽度均保持 epoch=1、a2、代码及三个 Path 表示。|
|选择与重绘|提示→程序链接留下 pending 选择；树重绘丢高亮，Path→树可能保留旧高亮。|移除延后选择定时器；同步选择当前树并复用高亮逻辑，重建/哈希跳过/不切页签的选择均重放视觉状态。两个保存的浏览器回归均通过。|
|直接链接|Path 直接进入时树尚未创建，D3 空选择的 `datum()` 抛错，中断代码页签恢复。|空选择先判断 `empty()`；Path→刷新→选择 a0→树后，URL、代码与唯一高亮均为 a0。|
|主题与可读性|旧浅色 Path、Cost、Ensembling、Battle、Diversity 标签出现 1.16–4.40 的文字对比度，提示标题保留紫色背景。|局部旧样式切到同一套 token；最终 Path/代码深色最低 5.65、浅色最低 4.51；13 个主视图抽查 1228 个可见叶文字实例，无低于 4.5 的样本。|
|错误真实性|损坏提示库可能断开统计响应或伪装成 `[]`；无转换器时下载正文却标成 PDF。|已有提示库损坏/缺表明确返回 500，删除损坏库后同服务恢复；首次 PDF 修复从假文件改为缺依赖 500，该历史观察保留但已被下项安全边界取代。|
|PDF 安全收口|合法 meta 路径不能约束转换器随后读取的本地文件与网络资源；仅缺依赖不等于安全。|研究处理器直接返回 501 JSON，不进入继承的转换路径。两种参数名×两种 meta 位置的 4 个真实 HTTP 用例先红后绿；转换器与子进程哨兵均未被调用，meta 正文仍可读取。双语禁用说明、390px 换行、两种别名的实际 501 和 1862B ZIP Blob 均经浏览器确认；上游独立查看器不改。|
|导出|浏览器 ZIP 路径未实际执行。|点击真实导出，输入 2，解码实际生成的 1862B Blob，得到 a2/a1 的完整代码及 README；没有执行候选代码，也不把 Blob 验证说成文件落盘验收。|
|检查口径复核|CSS 缩放代理不能替代原生 200%，其原截图也没有拍到代码；复制成功/复位仍有旧 emoji 文案。|保留原标准并补测 Chrome 原生 Page zoom=2，另拍可见代码的视口截图；源按钮和成功/复位文案改为双语纯文字，实际观察“复制→已复制→复制”，操作按钮 emoji 列表为空。|

## 带数据能力矩阵

普通运行夹具使用当前 `ProgramDatabase/Program` 正常保存，代码标为 `CONTROLLED ... NOT REAL SCIENCE`，不执行候选。主夹具有 6 个程序、3 代、2 个模型、2 个真实提示记录、代码/差异/日志/图像及 meta 文本。边界夹具分别覆盖负数、失败、全失败、空库、缺失、明确零、模型费用不完整、吞吐池与损坏提示库；这些不是 AU-5090 研究成果。

|主视图|实际观察|
|---|---|
|Tree|程序树 SVG；程序/代码选择与唯一高亮；全失败时无最佳节点。|
|Prompt Evolution|2 个提示，4 个 SVG；原生键盘选提示/程序、完整提示对话框、Escape 与焦点返回。|
|Programs|6 行程序；实际代码/详情选择和原生排序控件。|
|Metrics|4 个 Plotly 图。|
|Embeddings|3 个 SVG；另测全量嵌入迟到不污染 B 缓存。|
|Clusters|2 个 Plotly 图及 canvas。|
|Islands|3 个 SVG。|
|Ensembling|2 个模型行、2 个 SVG；模型成功率、费用缺失与明确零分别呈现。|
|Path|3 个步骤、趋势曲线与代码变更链联动；选择/刷新/重绘后的当前程序一致。|
|Cost|成本 SVG 与明细；未知费用累计不重新成为已知值。|
|MAPs|实际占用格；同格的有效 0 分不会被后来的 -1 分替代。|
|Battle|2 个模型行、2 个 SVG。|
|Diversity|3 行、2 个 SVG；单样本未知与两个相同嵌入的零距离不同。|

|详情视图|实际观察|
|---|---|
|Meta Info|6/6 正确、3 代、已记录总费用约 $0.35、最佳 a2/0.80。|
|Pareto|1 个 Plotly 图。|
|Meta/Scratchpad|实际读取并解析受控 individual/global/recommendation 标记；Global Insights 页签刷新后恢复。|
|Node|a2 的 ID、父节点、代数与元数据。|
|Code|a2 的完整保存代码；实际研究另核对 12-bar momentum 与 EMA span=20。|
|Diff|a1→a2；缓存选择后初始差异请求也可工作，迟到响应不覆盖新程序。|
|Evaluation|保存的标准输出与本地 PNG；迟到的 gen_2 图不替换当前 gen_1 图。|
|LLM|受控保存文本、模型、posterior/token 字段；恶意标记作为文字，未执行。|
|Analysis|11 个 Plotly 图。|
|Throughput|主夹具如实提示无池数据；另用真实 `pool_slots` 保存夹具显示 5 个图、6 行数据、16 秒窗口和 2 个任务。|

**2026-10-19 更正：**上表 Metrics、Clusters、Pareto 和 Analysis 四行只统计了 Plotly 图表容器的数量，不是渲染证明。2026-10-19 在桌面视口的 Chrome 中打开同一 6 程序主夹具，Metrics 页等待 4 秒后两个可见图表区域仍为空白（[`evidence/metrics-blank-20261019.png`](evidence/metrics-blank-20261019.png)），所以“Metrics｜4 个 Plotly 图”不成立；Clusters、Pareto、Analysis 的计数同样未经渲染证实，本记录不再作为这些图已经画出的证据。这四类视图在 `openspec/changes/award-redesign` 中以原生 SVG/D3 重做，并逐图实拍验收。

## 两条用户路径与状态证据

- **真实研究：**`btc-hourly-momentum/h001/e01` → 分析 → 选择 `377f7b78-d44c-4bb7-9cce-ad55f9a1ffb7` → 代码 → 刷新 → 返回 `/?direction=btc-hourly-momentum&hypothesis=h001`。重新启动的所有者预览 `http://127.0.0.1:8768/` 再次通过；代码仍为 `pct_change(12)`、`ewm(span=20)`，树唯一高亮和详情 ID 一致。只读取历史产物，没有重新运行研究或打开封存段。
- **普通运行：**键盘检索 `fixture`、按 `score` 排序、勾选 negative/failed-scores → 刷新仍保留检索/排序/两项选择 → 比较 → 进入 negative（最高有效分 -0.10）→ 浏览器后退。后退后的比较 URL 与两行数据完全相同；两组不同比较选择的原生历史也分别保留。
- **空库恢复：**正常保存一条惰性程序，选中并显示代码后删除夹具记录，再点击刷新；最终选中为 null、原始数据为 0、摘要为空、详情回到选择提示，两个空态与刷新控件保留，不停留在“正在读取”状态。
- **可复用回归：**`shinka/research/webui/browser-automation/unified-analysis.mjs` 的 `verifyPromptProgramSelection` 与 `verifyPathDeepLink` 在最终构建上执行通过；前者 resize 后仍选中并高亮 a2，后者直接链接/刷新/Path 选择/树切换后选中并高亮 a0。脚本迁入研究代码边界后已从新路径重新导入并跑通两项，不保留旧路径兼容层。调用参数取自已授权保存的运行，不执行候选。

结构化浏览器记录为 [`evidence/browser-acceptance.json`](evidence/browser-acceptance.json)，包含能力、交互、乱序、数值、导出、对比度与所有者预览的实际返回值。它是实现者的运行记录，不冒充独立视觉评审或原生管线认证。

## 最终运行与审查

|检查|实际结果|
|---|---|
|原生 Windows `pytest -q tests/research`|108 passed，59.67s；包含 Windows junction 和 4 个不进入 PDF 转换器的 HTTP 回归。|
|独立 WSL 全量 `pytest -q -m 'not requires_secrets'`|1017 passed、1 skipped、1 deselected、5 warnings，90.19s；跳过项是仅 Windows 的 junction 回归。|
|独立 Ruff check / format --check|两条命令 exit 0；`All checks passed!` / `26 files already formatted`。|
|前端 `pnpm run build`|整合时 TypeScript、主前端与分析构建 exit 0；安全收口后以 `pnpm exec vite build --config vite.analysis.config.ts` 重建分析资产，exit 0，15.16s。|
|`node --test src/analysis/actions.test.mjs`|3 passed；有限操作语法、完整拒绝可执行后缀、同源图像行为。|
|Standards / Spec|整合的有效发现已修正并经实际路径复验。安全收口另由两名独立审查者检查处理器拒绝、路径校验先后、相邻能力和研究代码边界；两轴均无可操作发现。审查不冒充运行证明。|
|本地运行资源|冷加载稳定 959ms，分析文档 DOMContentLoaded 182.6ms；记录的资源均来自 127.0.0.1，无页面/资源错误。这是本机 Chrome 测量，不是公网性能保证。|
|Git whitespace|整合提交 `bd7e353` 的完整 staged 检查曾 exit 1，恢复的历史 HTML 有行尾空白；排除恢复源目录后的检查 exit 0。安全更正的完整 `git diff --check bd7e353` exit 0，不反推历史整合检查已通过，也不批量改写脚本字符串或许可文本。|

安全收口后的独立后端证据保存在 `\\wsl.localhost\archlinux\tmp\shinka-pdf-isolation-gate-20261019-YS2pRrIL\stable-snapshot\`，含 exact argv、环境、JUnit、前后清单和完整日志；研究集为 107 passed、1 skipped，53.57s。最终四个门禁 exit 0，选定的 1506 个文件在门禁期间内容与成员保持一致；首次门禁期间 QA/截图仍变动，因此保留首轮证据并完整重跑，以稳定快照一轮为准。随后只更新交付记录，不改变下列已验代码哈希。五个警告来自未改的 bandit persistence 数值运算。原生 JUnit 为 `E:/projects/ShinkaEvolve-runs/ui-pdf-isolation-native.xml`。

整合提交的旧后端证据仍保存在 `\\wsl.localhost\archlinux\tmp\shinka-final-pdf-release-20261019-7d3c\evidence\`；旧 500 PDF 观察及其门禁是历史记录，不代替本次 501 拒绝契约的证据。结构化浏览器记录的 `safetyCorrection` 保存双语提示、真实请求、ZIP 解码、迁移后回归和重启后的 8768 实际候选检查。

最终关键 SHA-256：

```text
shinka/research/analysis.py
96691fb75c78d91924448a9281751faa0c221485bb595ac8dd3538ad49202cf7
tests/research/test_workspace_analysis.py
1b13d9fa0574d21b308b569e9f480b84078ebf91a2b4d276476c742319d0cadd
```

## 最终画面与适用限制

|证据|实际范围|
|---|---|
|[桌面](evidence/desktop-path.png)|1440px，Path 与 a2 代码同屏。|
|[平板](evidence/tablet-path.png)|1024px，保留 a2、代码与同一运行 epoch。|
|[手机](evidence/mobile-path.png)|390px，面板纵向重排，宽 Path/代码在局部滚动；外层和 iframe 均无横向溢出。|
|[浅色](evidence/light-path.png)|同一填充状态的 Technical Paper 主题。|
|[200% 代理](evidence/zoom-200.png)|仅证明 CSS 代理下已填充 Path 和保留的选择；代码状态由结构化记录确认，原图不作为可见代码证据。|
|[原生 200%](evidence/native-zoom-200.png)、[原生 200% 代码](evidence/native-zoom-200-code.png)|Chrome 的真实页面缩放设置；第二张视口截图实际显示 a2 的完整代码。100%→200%→100% 的控制值、DPR、宽度与恢复均留有记录。|
|[PDF 停用说明](evidence/pdf-disabled-mobile.png)|390px 的实际草稿页面；禁用按钮旁的中文说明完整换行可见，草稿正文仍在。|

- **实际成本：**分析 runtime JS 约 5.56MB（gzip 约 1.74MB），包含原功能需要的完整 Plotly 等本地依赖。该资源取舍与测得本机时间并列保留，不声称轻量或公网性能已通过。
- **无障碍范围：**测过键盘主路径、可见焦点、页签、提示对话框、减少动态效果和可见文字对比度；未进行全平台读屏、所有图表对象、Safari/Firefox 或完整 WCAG 认证。
- **导出范围：**ZIP 真实生成并解码通过；研究统一界面的服务端 PDF 导出当前停用，合法请求返回 501 JSON。仅安装 `wkhtmltopdf`、`pandoc` 或 PDF 引擎不能安全启用；必须先验证转换实现对本地文件和网络资源的限制。旧 500 依赖错误仅为历史观察，不是当前契约，也没有被当作资源隔离证明。本次不声称完整 PDF 能力对等，草稿阅读与程序 ZIP 不受影响；未安装转换器或执行实际转换。
- **许可：**构建随附 `/analysis/THIRD_PARTY_NOTICES.txt`；原归属、完整许可和 Plotly 内嵌组件补充清单保存在 `analysis/licenses/`。`native-promise-only` 的历史发布物只含作者 MIT 链接，补充文本明确保留来源年代限制，不冒充完整法律审计。
- **工具门禁：**文件绑定的静态 interaction audit 不能替代本 HTTP/API 应用的实际浏览器验收；旧 Sentrux 包装器要求的 PowerShell 7.2 不在本机，不记为通过。原 mentor 的 live 消息入口不可用，已保留其建议并用具备具体结果的独立 mentor debrief 复核交付边界；KB registry 不可用，没有虚构同步成功。
- **所有者与奖项：**本文件记录技术完成；所有者视觉验收仍未签署。Awwwards/Webby/FWA 是自检参考与品质目标，不存在本项目自授的“获奖分数”或获奖保证。
