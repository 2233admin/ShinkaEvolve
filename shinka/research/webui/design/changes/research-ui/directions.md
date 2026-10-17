# Directions

Preview: `direction-preview.json`, `direction-previews/index.html` (fixture `direction-previews/fixture.json`).

## A · notebook 研究记录本 — selected (owner, 2026-10-17)
- Thesis: research is a notebook kept entry by entry; verdicts and evidence are written as sentences.
- Fit: easiest for people the owner shares with; prints and screenshots well; light ground matches the upstream WebUI it links to.
- Risks found in preview: low density with many hypotheses; chart below the fold; CJK serif falls back badly on Windows.
- Fixes carried into DESIGN.md: status groups with one-line closed entries; chart directly under the facts row; serif only for display and wiki reading, system sans for prose and data.
- Owner addition: combine with existing Shinka WebUI features → experiment page with the evolution ledger (best path + diffs), best program and per-generation chart; heavy views linked to upstream.

## B · desk 交易台终端 — not selected
- Fastest scan, best numeric alignment. Rejected: unreadable to newcomers (rule pips), dark screenshots, hard contrast with the upstream light UI. Adopted: N paired with the noise bar in the facts row.

## C · audit 审计档案 — not selected
- Best traceability of the deciding rule. Rejected: generic card-grid look. Adopted: marking the deciding (first failing) rule.
