---
schema: design-pipeline.motion-foundation.v0.1
name: Shinka Research 暗场仪器 motion language
posture: static
primitiveRegistry: design-pipeline.motion-primitives.v1
---

# Shinka Research 暗场仪器 Motion Foundation

## Motion Thesis

The workbench is an instrument people read. Research state changes through the CLI, not on the page, so nothing is choreographed. Every state change on the page is instant. The only time-based behaviour is generation replay, which the reader starts and which steps one generation at a time.

## Motion Principles

- Nothing animates on load; content is never hidden behind an entrance.
- State changes are instant: selecting a program or row, switching a tab, view, theme or language, opening a panel, folding a group.
- Hover and focus change a colour, an outline or an underline, never a position or a size.
- Replay is discrete: one generation per step, never an interpolation between them.

## Motion Vocabulary

No moving primitive is selected for this static posture. The page ships no CSS transitions, keyframes or animation libraries in the native views. react-aria-components adds no animation of its own here, because its styles are not loaded.

## Procedural Motion

Disabled for charts. Charts are drawn from the data the page last read and are redrawn when the automatic refresh brings new data (every 10 seconds on All runs, 30 seconds elsewhere, paused on request). Points do not tween between redraws.

Generation replay is the one stepped sequence. The Lineage and Progress scrubber (`app/src/run/Scrubber.tsx`) steps one generation every 450 ms after Replay is pressed. It stops at the last generation, or when the reader presses Replay again, drags or uses the keys.

## Runtime Policy

- The native views (`app/src/**`, except `analysis/`) contain no `transition`, `@keyframes` or `animation` rules.
- The only timers are the automatic refresh, the replay step, and the 1.6 s after which the inspector's "copied" note reverts to its button label (an instant swap, not a fade).
- The restored analysis page, which serves Solution space and Contest (owner decision A), keeps its own chart libraries. Its stylesheet `app/src/analysis/workbench.css` stops the live-indicator pulse. It also removes every animation and transition when reduced motion is requested.

## Reduced Motion

Substitute: when `prefers-reduced-motion: reduce` is set, each press of Replay advances exactly one generation, and no timer runs. Fallback: everything else is already instant. The restored page drops its remaining transitions under the same setting.

## Source Decisions

Adopted: the static posture from the owner's choice of direction A, 暗场仪器 (`openspec/changes/award-redesign/design.md` › 动效). The chevron of the run groups lost its 0.15 s turn on 2026-10-22, so that state changes are instant everywhere.

Rejected: animated chart entrances, tweened redraws, live-updating counters and a pulsing status indicator, which would suggest the page controls or streams the research.
