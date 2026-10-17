---
schema: design-pipeline.motion-foundation.v0.1
name: Shinka Research 研究记录本 motion language
posture: static
primitiveRegistry: design-pipeline.motion-primitives.v1
---

# Shinka Research 研究记录本 Motion Foundation

## Motion Thesis

The console is a notebook people read and print. State changes in the research happen through the
CLI, not on the page, so there is nothing to choreograph. Pages appear complete.

## Motion Principles

- Nothing animates on load; content is never hidden behind an entrance.
- Expanding an entry, a diff or the best program is an immediate state change.
- Hover and focus feedback is an instant color or outline change.

## Motion Vocabulary

No moving primitive is selected for this static posture.

## Procedural Motion

Disabled. Charts are drawn once from the files on disk and are redrawn only when the page is
reloaded.

## Runtime Policy

No animation libraries, timers or transitions are shipped. Pages are server-rendered data plus a
small script for expanding entries and drawing charts.

## Reduced Motion

Substitute: none is needed because nothing moves. Fallback: when `prefers-reduced-motion: reduce` is set, behavior is identical — expanding content and hover/focus feedback stay instant state changes, and charts are drawn once.

## Source Decisions

Adopted: the static posture of the upstream design-pipeline foundation and the owner's choice of
the notebook direction (`design/changes/research-ui/direction-preview.json`).

Rejected: animated chart entrances and live-updating counters, which would suggest the page
controls or streams the research.
