# HypothesisEntry

One hypothesis as a notebook entry: id and trial count in the margin column; title, stamp, one sentence of evidence and the check ledger in the main column.

- Provide: id, title, status, trial count, and the evidence for that status.
- Evidence sentence by status:
  - verdict (supported / refuted / inconclusive): 验证段 Sharpe 1.21，DSR 0.97（N = 412），PBO 0.22，子区间为正 3/3；买入持有 0.78。 then `CheckLedger`.
  - running / challenged: in-sample only — 进化中，第 23/40 代，样本内最优 Sharpe 0.93，118 次试验。验证段证据出结论后才显示。
  - proposed / selected: the claim's first sentence, then 等待人选择 / 已选，待建立实验.
  - confirmed / unconfirmed: the verdict sentence, then a second sentence with the lockbox figures.
- End with 查看实验 → linking to the experiment page.
- Entries sit in three groups under a section heading: 在研 (proposed, selected, running, challenged), 已有结论 (supported, inconclusive, confirmed, unconfirmed), 已关闭 (refuted, dropped).
- In 已关闭, entries collapse to one line (id · title · stamp · shortest evidence) that is a `Toggle`; expanding shows the full entry in place.
- Below 760px the margin folds into a line above the title.
