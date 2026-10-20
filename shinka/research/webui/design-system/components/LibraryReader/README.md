# LibraryReader

The library reader shows the researcher's wiki: a page list on the left and one rendered Markdown page on the right, with a notice when raw material is waiting to be compiled. It lives in `app/src/library/Library.tsx` (award-redesign 4.3).

## What the consumer provides

- `lib`: `index` and `log` Markdown, `pages` (`name`, `text`) and `pending` (raw file names), or `null` when there is no library.
- `page`: the URL's page — `null` for the index, `log`, or `wiki/<name>`.
- Markdown is rendered with `markdown-it` (HTML disabled, so output is safe to inject); `[[page]]` links are app links to `/?view=library&page=wiki/<name>` (marked `data-wiki`) that the shell's router follows, so they also work from hypothesis text and in a new tab; other links open in a new tab with `rel="noopener noreferrer"`; leading YAML front matter is not shown.

## Built from

- A native `nav` of links: 目录, 日志, then the heading 页面（N） and one link per page. The open page carries `aria-current="page"`. Names wrap; they are never cut. From 768px the list sits 240px wide beside the page and stays in view under the top bar; on phones it is one row that scrolls on its own and starts at the open page.
- The header reuses the report's `rhead`: 共享资料库, then 所有研究方向共享同一份资料库 · N 个 wiki 页面 · N 份资料待整理.
- `.md`: the reading style (13px / 1.75) with headings (20, 16, 14), lists, `code` underlined by a `line` rule, `pre` between two rules, blockquotes with a `line-strong` left rule in `muted`, tables as ruled text, links in the text colour with a `stroke` underline. No filled blocks anywhere.
- Carbon `InlineNotification` (`kind="info"`, `lowContrast`, `hideCloseButton`) when `pending` is not empty: title 有 N 份资料待整理：, then the first five file names joined with 、 (等 when there are more) and the command that marks them compiled.

## States

- 目录 and 日志 show the library's `index` and `log`.
- A page the library does not have shows the index under an info notice 资料库里没有这一页： with the name.
- No library (`null`): 还没有资料库——用 `shinka_research library add` 加资料。 under the header and nothing else.
- No pages: the list shows only 目录 and 日志 plus the muted note wiki 里还没有页面，研究员整理资料后会写到这里。
- Empty page text: 这一页是空的。

## Don'ts

- Never inject HTML that did not come through `markdown-it` with `html: false`.
- Never fill code, quote or table blocks with a background; rules only.
- Never use the signal orange here: pending sources are work to do, not a fault.
- Never add editing: the library is written by the researcher agent, not from this page.
