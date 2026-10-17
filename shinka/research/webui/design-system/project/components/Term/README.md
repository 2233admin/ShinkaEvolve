# Term

A statistical or research term with a one-sentence Chinese explanation on hover or keyboard focus.

- Dotted `ink-muted` underline; the tooltip is `ink` ground with `on-ink` text, 280px wide, square.
- Use it the first time a term appears on a page (lede, facts labels, chart caption), not on every repeat.
- The fixed explanations:
  - N：这个方向里评估过的所有变体（含 probe）。试得越多，门槛越高。
  - DSR：考虑试验次数 N 后，真实 Sharpe 超过噪声门槛的概率。
  - PBO：样本内最优的策略，在样本外排到中位数以下的比例。
  - 噪声门槛：N 个纯噪声试验的期望最高分。
  - 验证段：agent 看不到的样本外数据，用来出结论。
  - Lockbox：研究期间谁都看不到的数据尾段；收尾时 confirm 一次性打开。
- Never put figures or advice in a tooltip; it explains a word.
