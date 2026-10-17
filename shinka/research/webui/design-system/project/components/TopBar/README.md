# TopBar

The 48px strip above every page: breadcrumbs on the left, the two destinations (工作区 · 资料库) on the right.

- Provide: the crumb trail (every crumb but the last is a link) and which destination is current (`aria-current="page"`).
- Sits on `paper` with no rule under it; the page's own spacing (`space-12`) separates it from the kicker.
- Below 760px only the breadcrumbs remain.
- Don't add actions, search or user menus: the console is read-only and has nothing to act on.
