# Term

A term is a word the page explains in place: a dotted underline that shows a plain-language definition on hover or focus.

## What the consumer provides

- A term key and the word as it reads in the sentence (`<Term k="insample">样本内</Term>`). The definitions live in one table, `terms.ts`; never write a definition inline.

## Built from

- Carbon `DefinitionTooltip` (`openOnHover`) wrapping the word: a `button.cds--definition-term` inside a `cds--popover-container` with a high-contrast popover (`cds--popover--high-contrast`, caret, bottom). It is keyboard-focusable and opens on focus as well.
- bundle.css makes terms inside table headers and spec-sheet keys inherit their parent's font, so a term never changes the type of the label it sits in; `.cds--popover-container` is capped at 100% so a term cannot widen its cell.

## Definitions

| Term | Definition |
| --- | --- |
| Sharpe | 每承担一份波动换来多少收益，越大越好，0 以下说明不如不做。 |
| 样本内 | 进化过程中程序能看到、拿来打分的那段历史数据。 |
| 验证段 | 进化时看不到的一段数据，质疑者用它检验结果；只在出结论后才显示。 |
| 封存段 | 数据最末尾单独封起来的一段，整个方向只在最终确认时打开一次。 |
| 试验数 | 整个方向里一共试过多少个程序（含探测），试得越多，碰巧撞上好结果的机会越大。 |
| 噪声线 | 试了这么多次，纯靠运气也能拿到的最高 Sharpe；真实结果要明显高于它才可信。 |
| DSR | 扣掉「试得多总会撞上好运」的部分后，Sharpe 真的大于 0 的概率。 |
| PBO | 样本内最好的那个程序，换一段数据后排名掉到中位数以下的可能性，越小越好。 |
| 子区间 | 把验证段切成几等份，看每一份是否都赚钱，避免靠一两段行情撑起整体结果。 |
| 买入持有 | 一直满仓拿着不动的基准做法，策略至少要比它强才有意义。 |
| 出结论后显示 | The reason a validation cell is `—` before a verdict (`tbd`). |

Each definition starts with the term and a full-width colon, then one plain sentence. A new term is added to the table once and used by key everywhere.

## States

- Closed: dotted underline in the text colour.
- Open (hover, focus): the popover above or below with a caret.
- Inside a header or key: same, and the cell keeps its width.

## Don'ts

- Never use a tooltip for anything but a definition; actions, errors and status belong elsewhere.
- Never put essential information only in a definition: the sentence must still read without it.
- Never explain a term twice on one screen with different words.
- Never style a term as a link or in orange.
