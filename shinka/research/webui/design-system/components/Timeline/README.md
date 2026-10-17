# Timeline

The experiment timeline draws one track per hypothesis that has trials, so the order and overlap of work is visible at a glance; clicking a region selects that hypothesis.

## What the consumer provides

- The hypotheses (id, title, status, experiments with `first_trial` and `last_trial`) and the selected id with `onSelect(id)`.
- The container width: the component measures it and picks the unit.

## Built from

- Astro `rux-timeline` with `start`, `end`, `interval` (`minute`, `hour` or `day`), `zoom`, `timezone="UTC"` and `hideJDay` (no day-of-year beside the date). The unit is the finest one that keeps the whole range from scrolling sideways at 0.6 × its column width (60px for minute and hour, 120px for day); `start` and `end` are aligned to unit boundaries because Astro drops a last partial column; `zoom` is clamped to 0.3–4. The 200px label column is fixed by Astro.
- One `rux-track` per hypothesis; its `slot="label"` holds `.track-label` with the hypothesis id in `.mono`.
- One `rux-time-region` per track: `start` and `end` of the hypothesis's trials (at least one minute wide), `status` from the hypothesis status (same map as HypothesisTable), `selected` for the current one, the hypothesis title as content. Clicking selects.
- `.timeline-wrap`: horizontal scroll, `line` rules above and below. bundle.css pins the region fill to `selected` (never an accent fill). The status dot inside a region keeps Astro's own status colours: the library does not expose them to tokens.

## States

- Tracks: only hypotheses with at least one experiment that has trials.
- Selected: the region carries a 1px `ink` border; others have none.
- No trials anywhere: no timeline, one muted sentence: 还没有试验，画不出时间线——用 `shinka_research run` 跑实验。
- Times are UTC.

## Don'ts

- Never fill a region with orange or a large inverted block; selection is the border.
- Never show future or planned work: regions are recorded trials only.
- Never let the timeline scroll the page sideways; it scrolls inside `.timeline-wrap`.
- Never add drag, resize or playhead: the timeline is read-only.
