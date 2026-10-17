# StatusStamp

A hypothesis status written as a stamp: glyph + Chinese word in `stamp` type, 1.5px border in the status ink, never filled.

| status | stamp | ink |
|---|---|---|
| proposed | ○ 待选 | `ink-muted` |
| selected | ● 已选 | `ink-secondary` |
| running | ▸ 运行中 | `running` |
| challenged | ◇ 质疑中 | `running` |
| supported | ✓ 支持 | `supported` |
| refuted | ✗ 证伪 | `refuted` |
| inconclusive | ? 不确定 | `inconclusive` |
| confirmed | ✓ 已确认 | `supported` |
| unconfirmed | ✗ 未确认 | `refuted` |
| dropped | – 已放弃 | `ink-muted` |

- Wrap the glyph in `<span class="g">` so it sets in the sans: serif CJK faces draw ✓ like a root sign.
- Place it after the entry title, `space-2`–`space-3` away, baseline-aligned.
- The word and glyph carry the meaning; colour only reinforces it (grey-scale prints must read).
