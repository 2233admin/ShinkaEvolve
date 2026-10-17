# FactsRow

The state of a direction (or experiment) in one line: label over value, between a 1.5px `ink` rule above and a 1px `rule-strong` rule below.

- Direction page items, in order: 开发数据 · 预算 · 试验数 N · Lockbox · 资料库.
- Experiment page items: 评估器 / 数据 (`HashChip`) · 设置 · 试验数 · 花费.
- Values in `data-lg` (tabular); the unit or context after it in `small`, `ink-muted`.
- Budget carries a 3px meter; at or over the limit the fill turns `refuted` and the value says 已用完.
- A missing value reads 未设置 in `ink-muted`, never a dash or 0.
- N always shows its noise bar beside it (门槛 1.36): a best score is never shown without N.
