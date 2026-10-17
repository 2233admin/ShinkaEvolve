# TopBar

The top bar names the product, shows three monitoring icons (trial count, budget, lockbox) and holds the theme switch; an orange index-stripe line closes it.

## What the consumer provides

- The snapshot's `trials`, `spend_usd`, `budget_usd` and `lockbox` (`null`, or `{opened}`); `null` snapshot while loading renders the bar with the switch only.
- The theme state; the switch is a controlled toggle that sets `data-theme` on `<html>` and the Carbon `Theme` (`g100` or `white`).
- The viewport: below 900px the bar is `compact` and the icons move into the strip row below it.

## Built from

- Astro `rux-global-status-bar` with `app-domain="SHINKA"` `app-name="RESEARCH"` `menu-icon=""` (no menu) and `slot="right-side"` holding `.bar-right`. In narrow mode add the `compact` attribute.
- Astro `rux-monitoring-icon` × 3, props `icon`, `label`, `sublabel`, `status`, and `title` carrying the term definition where the label is a term: `assessment` / 试验数 / the count; `attach-money` / 预算; `lock` or `lock-open` / 封存段.
- Carbon `Toggle` (`size="sm"`, `hideLabel`, `labelA="暗"` `labelB="亮"`) for the theme.
- `.stripes` (a `div`, `aria-hidden`): `repeating-linear-gradient` of `accent`, 2px on and 4px off, `stripe-height` tall. `.status-strip` is the narrow-screen row.
- bundle.css maps the icon fills: `normal` uses `ink`, `serious` uses `accent`, `caution` `warn`, `critical` `err`, `standby` and `off` `muted`.

## States

| Part | Value | Icon · sublabel · status |
| --- | --- | --- |
| 预算 | spend below 80% of budget, or no budget | `attach-money` · `$6.00 / $20.00` (or `$6.00`) · `normal` |
| 预算 | 80% or more | same · `caution` |
| 预算 | 100% or more | same · `critical` |
| 封存段 | lockbox not set | `lock-open` · 未设 · `off` |
| 封存段 | sealed | `lock` · 封存 · `serious` (the only orange in the bar) |
| 封存段 | opened | `lock-open` · 已开封 · `standby` |
| 试验数 | always | `assessment` · the count · `normal` |

Loading: no icons, only the title and the switch. Error: the bar is unchanged; the page body shows the error Notice.

## Don'ts

- Never add a menu, user name, clock or classification marking (`rux-classification-marking` carries military classification wording and is not used).
- Never colour an icon orange for anything but the sealed lockbox; budget warnings use `warn` and `err`.
- Never put a control here that writes research state; the theme switch is the only control.
- Never rebuild the icons by hand; set the Astro props.
