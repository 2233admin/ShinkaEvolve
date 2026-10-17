---
schema: design-pipeline.motion-foundation.v0.1
name: Shinka Research 运维规格台 motion language
posture: static
primitiveRegistry: design-pipeline.motion-primitives.v1
---

# Shinka Research 运维规格台 Motion Foundation

## Motion Thesis

The console is an ops sheet people read. State changes in the research happen through the CLI, not on
the page, so there is nothing to choreograph. The page ships no animation of its own; the only motion
is the small transitions the Carbon and Astro components already have, left at their defaults.

## Motion Principles

- Nothing animates on load; content is never hidden behind an entrance.
- Selecting a row, switching a tab or the theme, expanding a tree node and reading a definition are
  state changes; the library's own micro-transition may play, ours add nothing.
- Hover and focus feedback is a colour or outline change.

## Motion Vocabulary

No moving primitive is selected for this static posture. The libraries' own defaults, left as they ship, are Carbon's tooltip (popover) fade,
tab underline, toggle knob and tree-node expansion, and Astro's own hover states.

## Procedural Motion

Disabled. Charts are drawn once from the files on disk and are redrawn only when the page is
refreshed.

## Runtime Policy

No animation libraries, custom keyframes, timers or `transition` rules are shipped by this project;
`bundle.css` and `app.css` contain none. The library components' built-in transitions are not
overridden.

## Reduced Motion

Substitute: none is needed for our own surfaces. Fallback: the Carbon and Astro micro-transitions follow
the browser's `prefers-reduced-motion` setting as the libraries ship it; when it is set, state changes
are instant and charts are drawn once.

## Source Decisions

Adopted: the static posture (the checker's `minimal` posture requires a registered moving primitive, and we select none), with only the Carbon and Astro defaults, from the owner's choice of
the ops-spec direction and ready-made components (`design/changes/research-ui/component-selection/README.md`).

Rejected: custom animation, animated chart entrances, live-updating counters and a pulsing status
indicator, which would suggest the page controls or streams the research.
