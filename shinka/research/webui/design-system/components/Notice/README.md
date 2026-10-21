# Notice

A notice tells the reader what is wrong, missing or withheld, in one line, with the command that fixes it where there is one.

## What the consumer provides

- A kind, a title (what happened, ending in a full-width colon) and a subtitle (what to do or what is affected).
- Empty states are plain sentences, not notices.

## Built from

- `app/src/Notice.tsx`: a `div.notice` with `role="status"`, the icon named by `label` (Carbon's InformationFilled or ErrorFilled at 20px, kept as paths), the title, then the subtitle or the children. `kind` is `info` or `error`.
- `app/src/design/base.css` keeps the metrics of Carbon's low-contrast `InlineNotification`, which the app used until 2026-10-22: a 3px left edge, a 1px frame at 40% opacity, at least 48px tall, title 600 and subtitle 400 at 14px, and Carbon's width caps (18rem; 38rem from 42rem wide, 46rem from 66rem, 52rem from 99rem). Info has a `hover` ground with a `muted` edge and icon; error has a `notice-error-bg` ground (Carbon's `#262626` dark and `#fff1f1` light, kept as a token) with a `signal` edge and icon. The library page widens notices to 1000px.
- Empty states: `p.muted` (13px, `muted`); commands in `<code class="mono">`.

## Copy in use

| Where | Kind | Title · subtitle |
| --- | --- | --- |
| Snapshot cannot be read | `error` | 读取失败： · `{message}`。确认 shinka_research ui 还在运行，然后点刷新。 |
| No hypotheses | `info` | 还没有假设： · 用 shinka_research hypothesis add 添加第一个假设。 |
| Library has raw material to compile | `info` | 有 N 份资料待整理： · the first five file names joined with 、 (等 when there are more), then the command that marks them compiled |
| Library page that does not exist | `info` | 资料库里没有这一页： · the page name |

Empty-state sentences: 还没有试验——用 `shinka_research run` 跑实验。 · 还没有进化出的程序（只有 N 次试探，试探不画点）—— · 还没有试验，画不出时间线—— · 还没有事件。 · 还没有资料库——用 `shinka_research library add` 加资料。 · wiki 里还没有页面，研究员整理资料后会写到这里。 · 这一页是空的。

## Don'ts

- Never use a notice for success; success is a status symbol.
- Never add a close button, a call-to-action button or an illustration.
- Never use exclamation marks, apologies or blame in the copy; say what is missing and the next command.
- Never use Astro's `rux-notification` (its thick yellow edge fights the rules-only look).
- Never show validation or lockbox numbers in a notice.
