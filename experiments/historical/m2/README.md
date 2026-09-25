# M2 — Balanced Compact Encoder

Historical experiment imported from the original local Brain2Qwerty EEG development checkout.

## Public summary

M2 combines:

- per-electrode temporal filtering
- learned signed spatial channel projection
- subject-conditioned FiLM
- depthwise-separable dilated residual temporal blocks
- learned temporal pooling
- compact sentence Transformer

Recorded historical result:

| Metric | Value |
|---|---:|
| Parameters | ~1.640M |
| Validation CER | 72.39% |
| Test CER | 72.20% |

This is the strongest completed compact result currently recorded in the historical experiment table.

## Provenance warning

This is a historical source snapshot. The training script retains original package/import assumptions from the development checkout and is not yet part of the standardized EEG2Qwerty benchmark runner.

The CER values above were produced with the earlier internal evaluation pipeline and are not presented as a direct comparison with the final published Brain2Qwerty result.
