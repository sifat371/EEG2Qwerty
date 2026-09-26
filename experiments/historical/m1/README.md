# M1 — Temporal-Attention Variant

Historical experiment imported from the original local Brain2Qwerty EEG development checkout.

## Public summary

M1 extends the compact M0 family with learned temporal attention over the neural samples within each keypress window.

Recorded historical result:

| Metric | Value |
|---|---:|
| Parameters | ~2.684M |
| Validation CER | 75.10% |
| Test CER | 74.56% |

## Provenance warning

This is a historical source snapshot. The training script retains original package/import assumptions from the development checkout and is not yet part of the standardized EEG2Qwerty benchmark runner.

The CER values above were produced with the earlier internal evaluation pipeline.


Only the original machine-specific output directory was replaced with a repository-relative path during public migration; the model/training logic is otherwise preserved as the historical snapshot.
