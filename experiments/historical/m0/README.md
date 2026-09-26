# M0 — Average-Pooling Baseline

Historical experiment imported from the original local Brain2Qwerty EEG development checkout.

## Public summary

- compact EEGNet-style spatial-temporal encoder
- average pooling inside each keypress window
- subject embedding
- sentence Transformer or BiGRU option
- 29-character output space

Recorded historical result:

| Metric | Value |
|---|---:|
| Parameters | ~2.679M |
| Validation CER | 75.30% |
| Test CER | 74.74% |

## Provenance warning

This is a historical source snapshot. The training script retains original package/import assumptions from the development checkout and is not yet part of the standardized EEG2Qwerty benchmark runner.

The CER values above were produced with the earlier internal evaluation pipeline.


Only the original machine-specific output directory was replaced with a repository-relative path during public migration; the model/training logic is otherwise preserved as the historical snapshot.
