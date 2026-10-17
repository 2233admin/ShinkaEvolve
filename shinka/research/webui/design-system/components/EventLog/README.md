# EventLog

The direction's `results.tsv`, newest first: one row per scaffold, run, probe, skeptic, verdict or confirm.

- Columns: time (`ink-muted`, `MM-DD HH:MM`) · hypothesis id (`ink`, 600) · event word · note (`ink-secondary`).
- Event words: scaffold 建立实验 · run 进化 · probe 试探 · skeptic 质疑 · verdict 结论 · confirm 启封.
- Ten rows by default, then a `Toggle` 显示全部事件.
- Notes are copied from the log, not rewritten; a note never shows validation figures for a hypothesis without a verdict (the CLI already guarantees this).
- Below 760px each row stacks.
