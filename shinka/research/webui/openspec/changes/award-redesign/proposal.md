## Why

2026-10-19 所有者判定统一前端仍不合格，并把目标定为可角逐 Awwwards、Webby、FWA 的作品水准，通过自检反复提升直到达到。真实 Chrome 截图（`E:/projects/ShinkaEvolve-runs/frontend-diagnosis-20261019/`）显示：首页是黑底中的单行表格；方向页试验图横轴出现 0.2、0.4 等小数序号，时间线标签被挤成逐词换行，事件流表头是英文且行被截断；分析页是 iframe 里的 18,626 行旧页面，双层页头、13 个页签溢出截断、图例浮层遮挡画布，指标页四个图表为空白灰块；390px 下浮层遮住页签、表格被截。上一轮设计管线只记录了 intake 与交付（gates、drafts 均为空），没有方向样稿、门禁和评审点，这些缺陷因此没有被拦下。

## What Changes

- 用 3 个在同一份演示数据上真实渲染的方向样稿重新比较视觉方向，由所有者选择；其中一个是运维规格台的升级版。选定后重写 `DESIGN.md`、`MOTION.md` 与 `design-system/`。
- **BREAKING（设计系统）**：移除 Astro UXDS 与 Carbon；交互与无障碍行为由 react-aria-components 提供（所有者 2026-10-22 选定），外观、图表与数据呈现由新设计系统负责。
- 研究页原生重做：研究方向总览首页（状态、证据、预算与封存段一屏可读）、方向与假设研究报告（主张 → 试验与噪声线 → 检查台账 → 判词）、共享资料库。
- **BREAKING（分析入口）**：分析从 iframe 旧页面迁为原生视图，按要回答的问题重组为谱系、进展、解空间、对比四个联动视图和一个常驻程序检查器；运行列表与多运行比较同样原生重做。旧 `/viz_tree.html`、`/analysis/*` 页面与 `?analysis=` 投影在对应能力组通过对照后删除，不保留兼容跳转（AGENTS.md：不维护向后兼容）。
- 合并旧视图前由所有者审阅能力对照表；旧视图能回答的每个问题在新界面中都有去处并经实测。
- 新增确定性、明确标注“演示数据”的演示工作区生成器，只用于设计评审与验收，不运行研究、不调用模型。
- 只读 HTTP 契约保持；确需新增的只读聚合接口沿用同一路径约束、私有指标隐藏与错误语义。

## Capabilities

### New Capabilities

- `research-workspace-ui`：研究方向总览、方向与假设研究报告、共享资料库的呈现、导航、空与错误状态及只读边界。
- `run-analysis-ui`：运行列表、多运行比较、四个联动分析视图与程序检查器；能力对照、选择联动、URL 恢复、不可信文本净化与导出。
- `demo-workspace`：确定性、明确标注的演示工作区与普通运行生成，不触碰真实研究数据。

### Modified Capabilities

（无。`openspec/specs/` 目前没有既有规格。）

## Impact

- 前端：`shinka/research/webui/app/src/**` 重写；`shinka/research/webui/analysis/**`、`app/src/analysis/**`、`vite.analysis.config.ts` 在各能力组通过对照后删除；`app/package.json` 依赖按工具链计划调整（移除 `@astrouxds/astro-web-components`；Carbon 的三个包已换成 `react-aria-components`；图表与 3D 依赖由工具链计划决定）。
- 服务：`shinka/research/ui.py` 提供新路由；`shinka/research/analysis.py` 只在需要时增加只读聚合接口。
- 测试与文档：`tests/research/` 中与旧页面路由相关的回归随迁移更新；`docs/research/workspace.md`、`DESIGN.md`、`MOTION.md`、`design-system/` 更新。
- 不变：研究引擎、评估器、notebook 格式、CLI 语义、读取权限与 PDF 停用边界。
