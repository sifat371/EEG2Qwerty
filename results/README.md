# Results

This directory contains **curated research results**, not raw training outputs.

## Historical results

`historical_results.csv` records results obtained during the exploratory development phase. Those runs used an earlier internal training/evaluation pipeline and are retained for provenance.

They must not be interpreted as standardized comparisons against the final published Brain2Qwerty EEG pipeline.

## Standardized results

Future reproduced experiments should be added separately and include:

- Git commit SHA
- model/config identifier
- random seed
- target definition
- sentence batching protocol
- evaluator version
- neural-only CER
- language-model-assisted CER when applicable
- participant-level mean, median, standard deviation, and/or IQR
- parameter count
- peak VRAM
- training time
- inference latency

Raw checkpoints and logs belong under ignored local directories such as `runs/`, `checkpoints/`, or `results/raw/`.
