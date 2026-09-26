# M3 — GeoFusion

Historical experiment imported from the original local Brain2Qwerty EEG development checkout.

## Public summary

M3 explored a higher-capacity geometry-aware EEG encoder with:

- electrode-position-aware virtual-channel fusion
- participant-specific spatial adaptation
- multi-scale temporal filtering
- subject-conditioned FiLM
- gated dilated residual temporal blocks
- ALiBi sentence encoding

Recorded historical result:

| Metric | Value |
|---|---:|
| Parameters | ~13.69M |
| Validation CER | 76.78% |
| Test CER | 76.58% |

## Provenance warning

This is a historical source snapshot. It retains the original development-checkout imports and evaluation logic and has not yet been reproduced under the standardized EEG2Qwerty benchmark.
