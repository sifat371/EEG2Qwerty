# EEG2Qwerty public package

This package contains reusable, non-sensitive infrastructure intended to become the stable public EEG2Qwerty framework.

Current public components:

- `data.sentence_batching` — recording-aware whole-sentence batching utilities
- `metrics.cer` — dependency-free character-error-rate utilities
- `metrics.participant` — pooled and participant-level CER aggregation

Historical architectures remain under `experiments/historical/` until they are reproduced and intentionally refactored under the standardized benchmark.

Unpublished research models are not developed in this public package.
