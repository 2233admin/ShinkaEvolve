# CheckLedger

The check ledger lists the five rules behind a verdict, one line each with its pass mark and the measured figure, followed by the in-sample context and the reason sentence.

## What the consumer provides

- The verdict: `evidence` (`oos_sharpe`, `baseline_buy_hold_oos_sharpe`, `dsr`, `oos_positive_fraction`, `pbo`, `is_sharpe`, `expected_max_sharpe_annual`, `n_trials`), the `thresholds` recorded when the verdict was made (`dsr_supported`, `min_positive_fraction`, `max_pbo`) and the `reason` sentence.
- It renders only inside the SpecCard, and only when a verdict exists.

## Built from

- `section` with an `h3` section label (`label-section`, "检查台账"), then `ul.ledger`: rows of a 20px column for Astro `rux-status`, the rule text, and the figure in `.mono.ledger-value` (`data-sm`, right-aligned).
- The five rules, in this order, with the thresholds from the verdict: 验证段 Sharpe > 0 · 跑赢买入持有 (`oos > buy-and-hold`, value `a vs b`) · DSR ≥ `dsr_supported` · 子区间为正 ≥ `min_positive_fraction` (percent) · PBO ≤ `max_pbo`. Terms in the rule text are Carbon `DefinitionTooltip` (`Term`).
- Below the list, a spec sheet (`dl.spec`): 样本内 Sharpe, 噪声线, 试验数 (the number of trials counted into the noise line).
- `p.reason`: the verdict's own reason sentence, unedited.

## States

| Mark | Meaning |
| --- | --- |
| `rux-status normal` | the rule passes |
| `rux-status critical` | the rule fails (the deciding failures are the red marks) |
| `rux-status off` | evidence is missing; value `—` |

PBO that cannot be computed counts as a pass and says so in words: 无法计算，视为通过.

## Don'ts

- Never show a ledger for a hypothesis without a verdict.
- Never reorder, hide or merge rules, or invent a sixth one; they mirror `skeptic.decide()`.
- Never show a pass as green alone: the mark's shape and the figure are always there.
- Never edit the reason sentence or round figures differently from the spec: Sharpe two decimals, DSR and PBO per `fmt` (DSR three), fractions as whole percents.
