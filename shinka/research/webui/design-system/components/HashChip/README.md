# HashChip

The fingerprint of an experiment's frozen evaluator or data: the first 12 hex characters in `code` type; the full SHA-256 in its tooltip.

- Provide: label (评估器 / 数据) and the full hash.
- When the file on disk no longer matches the hash recorded at scaffold, show `Notice` (error) 评估器已被改动，结果不可信 under the facts row instead of the experiment's figures.
