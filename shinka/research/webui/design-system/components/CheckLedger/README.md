# CheckLedger

The five decision rules behind every verdict, written inline in `body-sm`: ✓ in `supported`, ✗ in `refuted`.

- Order is the decision order: 验证段 Sharpe > 0 · 跑赢买入持有 · DSR ≥ 0.95 · 子区间为正 ≥ 2/3 · PBO ≤ 0.5. Thresholds come from the direction's `program.md`, not from this component.
- The first failing rule is the one that decided the verdict: underline it in `decider` and give it the tooltip 这一条决定了结论.
- When a rule could not be computed (PBO with too few programs) show — 不适用 in `ink-muted`; it counts as passing, as the skeptic does.
- Shown only after a verdict. Before that the entry says validation evidence appears after the verdict.
