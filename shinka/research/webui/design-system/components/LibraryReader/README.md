# LibraryReader

The library reader shows the researcher's wiki: a page tree on the left and one rendered Markdown page on the right, with a notice when raw material is waiting to be filed.

## What the consumer provides

- `lib`: `index` and `log` Markdown, `pages` (`name`, `text`) and `pending` (raw file names), or `null` when there is no library.
- Markdown is rendered with `markdown-it` (HTML disabled, so output is safe to inject); `[[page]]` links become in-library links (`data-wiki`), other links open in a new tab with `rel="noopener noreferrer"`; leading YAML front matter is not shown.

## Built from

- Carbon `TreeView` (`size="sm"`, `hideLabel`) with `TreeNode`: 目录, 日志, then an expanded 页面（N） parent with one leaf per page. The node ids are the page names; selecting a leaf shows it. The view is re-keyed by the selected page because Carbon reads `active` and `selected` only on mount; this keeps the tree in step when a `[[link]]` jumps.
- `.lib`: a two-column grid (`--library-nav-width` and the page, 32px apart), one column below 900px. `.lib-page` stops at `78ch`.
- `.md`: the reading style (`text-reading`, 13px / 1.75) with `heading-reading` headings (20, 16, 14), lists, `code` underlined by a `line` rule, `pre` between two rules, blockquotes with a `line-strong` left rule in `muted`, tables as ruled text with `label-spec` headers, links in `accent-text`, underlined. No filled blocks anywhere.
- Carbon `InlineNotification` (`kind="warning"`, `lowContrast`, `hideCloseButton`) above the grid when `pending` is not empty: title 有 N 份资料待整理：, subtitle the first five file names joined with 、 and 等 when there are more.
- Carbon `Tabs` with a `Tag` (`size="sm"`, `type="gray"`) showing the pending count on the 资料库 tab.

## States

- Page selected: tree and article as above; the selected node has Carbon's active mark mapped to `accent`.
- 目录 and 日志 show the library's `index` and `log`.
- No library (`null`): 还没有资料库——用 `shinka_research library add` 加资料。 and nothing else.
- No pages: the tree shows only 目录 and 日志 plus the muted note wiki 里还没有页面，研究员整理资料后会写到这里。
- Empty page text: 这一页是空的。
- A `[[link]]` to a page that does not exist does nothing.

## Don'ts

- Never inject HTML that did not come through `markdown-it` with `html: false`.
- Never fill code, quote or table blocks with a background; rules only.
- Never use orange for anything here except links, the selected tree mark and focus.
- Never add editing: the library is written by the researcher agent, not from this page.
