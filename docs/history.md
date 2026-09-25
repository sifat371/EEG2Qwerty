# Historical Development Lineage

This page records **completed public-facing experiments** only. It intentionally does not document unpublished hypotheses or planned model designs.

## M0 — compact baseline

A compact EEGNet-style spatial-temporal encoder followed by a sentence sequence model.

Historical result:

- validation CER: 75.30%
- test CER: 74.74%
- approximately 2.679M trainable parameters

## M1 — temporal attention

Extended the compact baseline with learned temporal attention within each EEG keypress window.

Historical result:

- validation CER: 75.10%
- test CER: 74.56%
- approximately 2.684M trainable parameters

## M2 — balanced compact encoder

Combined per-channel temporal filtering, learned spatial projection, subject-conditioned FiLM, dilated residual temporal blocks, learned temporal pooling, and a compact sentence Transformer.

Historical result:

- validation CER: 72.39%
- test CER: 72.20%
- approximately 1.640M trainable parameters

M2 is the strongest completed compact historical result currently recorded in this repository.

## M3 — geometry-aware spatial fusion

Explored electrode-position-aware spatial fusion with a higher-capacity temporal/contextual model.

Historical result:

- validation CER: 76.78%
- test CER: 76.58%
- approximately 13.69M trainable parameters

## M4 — residual geometry variant

Explored a residual geometry-aware path while retaining the direct EEG signal pathway.

Historical result:

- validation CER: 73.15%
- test CER: 72.81%
- approximately 6.642M trainable parameters

## G1 — graph-temporal ablation

Explored graph-based electrode interactions and temporal sequence modeling.

Historical result:

- best recorded validation CER: 72.19%
- substantially higher activation-memory cost than the compact M2 line

## Interpretation policy

These numbers are maintained as development provenance. They were obtained before the current standardized EEG2Qwerty benchmark protocol was finalized.

The public repository therefore does not use them to claim superiority over the published Brain2Qwerty system.

As historical architectures are reproduced under the standardized pipeline, corrected results should be added to the benchmark registry without deleting the original records.
