# Grill 决策记录（research-ui）

来源：所有者在对话中的回答（2026-10-17），以及 `docs/research/design.md` §7.3。

## 已定（所有者）
- 范围：只读。选中/放弃假设、出结论、confirm 走 CLI；confirm 永不进 UI。
- 可见性：未出结论的假设只显示样本内指标；验证段证据在 verdict 之后显示；lockbox 数字只在 confirm 之后显示。
- 受众：所有者在桌面浏览器使用，也会分享给别人看；手机要能读。
- 新人引导：术语悬停解释 + 页顶一句话说明；不做引导流程。
- 界面语言：中文（Sharpe、DSR、PBO 等术语保留英文缩写）。
- 风格：独立于上游 Shinka WebUI，跳转过去时不违和。
- 分工：agent 出方向预览与实现；所有者看每一步（含 DESIGN.md 初稿），选方向并最终验收。
- 视觉规范按 design-pipeline（ui / standard）。

## 产品语言（术语表）
- 方向 direction：一个研究问题，含多个假设，共享数据、lockbox、预算、试验数。
- 假设 hypothesis：可证伪陈述；状态 proposed / selected / running / challenged / supported / refuted / inconclusive / confirmed / unconfirmed / dropped。
- 试验数 N：方向内所有评估过的变体（含 probe），DSR 的多重检验基数。
- 门槛线：N 个纯噪声试验的期望最大 Sharpe（与质疑者同一近似）。
- 验证段：样本外、agent 看不到的数据段；lockbox：方向首次 scaffold 时切出的尾段，只开一次。
- 资料库：工作区共享的 raw/ → wiki/。

## 设计相关决定
- 压力与任务：打开就要回答"这个方向现在到哪了、哪个假设站得住、还剩多少预算、lockbox 开没开"。扫读优先，重复使用。
- 失败与边界状态：空工作区、方向无假设、假设未跑、实验运行中、数据未 scaffold、预算未设、lockbox 已开、wiki 为空、文件损坏/读不到。
- 密度：数据密集的表格式布局；数字 tabular、右对齐。
- 禁止的视觉语言：营销落地页式 hero、渐变装饰、玻璃拟态、无意义图标、emoji 当状态、只靠颜色区分状态。
- 平台与性能：Python stdlib 服务 + 静态 HTML/JS，无构建步骤，不依赖外网 CDN（沙箱和离线机器都要能用）。
- 无障碍：WCAG AA 对比度，键盘可达，状态用文字+形状而非只靠颜色，尊重 reduced motion。
- 组件体系：项目内自写少量组件（状态标签、证据格、清单、图表），不引入组件库。
