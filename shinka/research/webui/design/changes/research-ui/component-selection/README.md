# 组件选型：Astro UXDS × Carbon（2026-10-18，所有者确认）

所有者选了方向 B（ops-spec，运维规格台），并要求「不手搓组件，用现成组件库」，研究界面与 KatanaConsole 分开、组件库也分开。

做法：研究界面要用的每个件，两个库各放一版，套同一份 token（Operations Dark / Technical Paper），并排比较。所有者看图后定：「现在样子挺好的，就这样」。

- 暗色：[dark.png](dark.png)
- 浅色：[light.png](light.png)

## 分工

Astro UXDS 8.0 管「信号」：一眼要看到的状态、时间、运行情况。Carbon 1.118 管「内容」：要读、要查的东西。

| 件 | 用哪个 | 理由 |
|---|---|---|
| 顶栏 | Astro `rux-global-status-bar` + `rux-monitoring-icon` | 应用名有分量，监控图标 + 时钟就是运控台；Carbon Header 薄且普通 |
| 状态符号 | Astro `rux-status` | 形状带意思（三角 = 否定、方块 = 未定），色弱可分；Carbon 圆角标签是 SaaS 样 |
| 实验时间线 | Astro `rux-timeline` | 独有；日期旁的年内第几天要处理掉 |
| 事件流 | Astro `rux-log` | 自带过滤与状态符号；时间列要补日期 |
| 术语提示 | Carbon `DefinitionTooltip` | 就是为术语解释做的 |
| 页签、折叠 | Carbon `Tabs`、`Accordion` | Astro 的未选中页签与折叠标题全用强调色，太吵 |
| 表格 | Carbon `DataTable`，状态列放 Astro `rux-status` | 紧凑、可排序、可展开 |
| 提示条 | Carbon `InlineNotification` | Astro 提示条的粗黄边压不住 |
| 试验图 | Carbon Charts | Astro 没有图表 |
| 代码 | Carbon `CodeSnippet` | Astro 没有；真看 diff 另选 diff 组件 |
| 规格卡 | Carbon `StructuredList` 打底 + Astro `rux-status` + 自写的键值规格块 | 两边单独都不够 |
| 开关 | Carbon `Toggle` | |

不用：Astro `rux-classification-marking`（军用密级语义，「UNCLASSIFIED」前缀去不掉）。封存状态放进顶栏的监控图标。

## 接入时踩到的

- `@astrouxds/react` 8.0.0 调用 Stencil 4 已删除的 `applyPolyfills`，不打补丁起不来。改为 React 19 直接用 `rux-*` 自定义元素，少一层包装。
- Astro 在组件内部（`:host`）重新声明颜色变量，token 要设到元素本身上；仍有个别颜色（PASS 标签底色）改不动，用到时换件。
- Astro 浅色主题下顶栏文字是白色，要我们补变量。
- Carbon Charts 用自己的配色，图表要显式给颜色映射。
