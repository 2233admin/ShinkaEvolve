# Toggle

The only control in the console: a text button that shows or hides content in place.

- Provide: the label and the region it controls (`aria-expanded`, `aria-controls`).
- Collapsed: ▸ + label in `link`. Expanded: ▾ + label in `ink`.
- Labels say what appears: 展开改动 · 最优程序 · 显示全部事件. A whole collapsed entry row can be the toggle.
- 28px tall on desktop, 44px touch target below 760px; `focus` ring on keyboard focus.
- Expanding is instant (MOTION.md: static).
