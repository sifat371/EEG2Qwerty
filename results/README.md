# Results

This directory contains **curated research results**, not raw checkpoints or logs.

## Historical results

`historical_results.csv` records measurements obtained during the exploratory development phase. Those runs used an earlier internal training/evaluation pipeline and are retained for provenance.

They must not be interpreted as standardized comparisons against the final published Brain2Qwerty EEG pipeline.

## Standardized experiment registry

`experiment_registry.csv` is intentionally empty until a full standardized run is audited and promoted.

The registry stores:

- EEG2Qwerty commit SHA
- upstream Brain2Qwerty commit SHA
- protocol/target/evaluator
- random seed
- parameter count
- hardware and peak VRAM
- training time
- neural-only CER
- participant mean/median/SD CER
- LM-assisted CER when applicable
- result status and notes

The training runner first writes a **candidate** `registry_row.csv` inside the ignored run directory. That row should be copied into the curated registry only after the protocol audit passes.

Raw checkpoints, logs, prediction dumps, and run manifests belong under ignored directories such as `results/raw/`.


## Promote an audited candidate

After a full non-debug run and a passing loader-aware protocol audit:

```bash
python scripts/promote_registry_row.py \
  --candidate results/raw/m2_typed_seed33/registry_row.csv \
  --audit results/raw/protocol_audit_with_loaders.json
```

The promotion command refuses to append a row unless train/validation/test are present, typed reconstruction meets the validation threshold, and every loader reports zero sentences split across batches.
