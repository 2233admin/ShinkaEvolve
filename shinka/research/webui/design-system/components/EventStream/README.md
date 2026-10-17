# EventStream

The event stream is the direction's chronological log: one line per run, skeptic report, verdict or confirmation, with a status symbol and a search box.

## What the consumer provides

- The snapshot's `events`: `time` (ISO), `hypothesis`, `experiment`, `event`, `note`, oldest first as recorded.

## Built from

- Astro `rux-log` with its `data` property set to `[{ timestamp: Date, status, message }]`; `message` is `` `${hypothesis}/${experiment} · ${event} · ${note}` ``. Astro supplies the table, the time cell (`rux-datetime`, UTC, time of day), the status symbol (`rux-status`) and the search input.
- Status per event: `verdict` and `confirm` `normal`; `skeptic` `caution`; `run` `standby`; `run-failed` `critical`; anything else `off`.
- bundle.css sets the block to `display: block` and the text sizes to 13px (12px for small text), so the log reads at the same size as the rest of the page; status fills map to `ok`, `warn`, `err`, `muted`.
- The preview adds a plain table that shows only when the Astro element is not defined.

## States

- Events present: the log in the order given, filterable by the search box (Astro shows the active filter and how many records are hidden).
- No events: no log, one muted sentence: 还没有事件。
- The time column shows the time of day only; the date is not shown.

## Don'ts

- Never add live updating, auto-scroll or counters; the log is read from disk and refreshed with the page's refresh button.
- Never colour the message text by status; the symbol carries it.
- Never show validation or lockbox figures in the `note` before the verdict or confirmation that releases them.
