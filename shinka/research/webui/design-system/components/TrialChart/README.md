# TrialChart

Every trial of a direction against the noise bar: the picture of what trying more costs.

- Provide: the trials (number, hypothesis, generation, in-sample score, kept or not, failed or not) and the noise bar per N (`expected_max_sharpe` from the skeptic).
- X is the trial number across the direction (on the experiment page: generation). Y is the in-sample score.
- Discarded trials: 3.6px `trial-dot` dots. Each new best: a 5.6px `frontier` dot, joined per hypothesis by a 1.6px step line.
- Alternate `paper-deep` bands per hypothesis, labelled `h001`… under the axis in `chart` type.
- The noise bar: dashed 5/4 `noise-bar` line labelled 噪声门槛 <value> at its right end.
- Gridlines every 0.5 in `rule`; no chart border, no legend box.
- Caption in the margin column: 虚线：N 个纯噪声试验的期望最高分。实线：每个假设的最优前沿。 plus 失败 <k> 次 when trials failed — failed trials are counted, never plotted as 0.
- Hover or focus on a kept dot shows 第 212 次 · h002 · 第 14 代 · 样本内 Sharpe 1.48.
- Fewer than two trials: `Notice` 还没有可画的试验.
