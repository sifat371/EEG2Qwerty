# Results

This directory contains **curated research results**, not raw training outputs.

## Historical results

`historical_results.csv` records measurements obtained during the exploratory development phase. Those runs used an earlier internal training/evaluation pipeline and are retained for provenance.

They must not be interpreted as standardized comparisons against the final published Brain2Qwerty EEG pipeline.

## Standardized experiment registry

`experiment_registry.csv` defines the public schema for future reproduced runs.

A standardized row should include:

- Git commit SHA
- protocol and target
- evaluator version
- random seed
- parameter count
- peak VRAM
- training time
- neural-only CER
- participant-level CER statistics
- LM-assisted CER when applicable
- status/notes

Raw checkpoints and logs belong under ignored local directories such as `runs/`, `checkpoints/`, or `results/raw/`.

The repository deliberately keeps **historical** and **standardized** measurements separate.
