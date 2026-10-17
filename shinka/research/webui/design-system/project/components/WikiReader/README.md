# WikiReader

A library wiki page in `reading` type at `reading-measure`; metadata in the margin column.

- Margin: type (概念 / 论文 / 发现), sources (links to `raw/…`), updated date, pages that link here.
- Headings `reading-h2`; quotes with a 2px `rule-strong` rule; tables with `rule` hairlines; code as `CodeBlock`.
- `[[page]]` becomes a link in `link` showing the page name without brackets; a page that does not exist yet is `ink-muted` with a dotted underline and the tooltip 这个页面还不存在.
- Finding pages never show lockbox figures before `confirm` — the agent writes them that way; the reader shows what is on disk.
