# Diff

A unified diff of one evolution step, as Shinka stored it.

- One line per row on `paper-raised`; added lines on `diff-add` starting with +, removed on `diff-del` starting with −, hunk headers on `diff-hunk` in `ink-muted`.
- No syntax colouring: the ground and the sign carry the change, so grey-scale prints read.
- Never wraps; scrolls horizontally inside its block.
- Shown inside `EvolutionLedger` behind a `Toggle` 展开改动.
