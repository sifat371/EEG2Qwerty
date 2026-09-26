# Public EEG data and evaluation utilities

These scripts are path-agnostic versions of utilities used during the Brain2Qwerty EEG reproduction work.

They intentionally depend on the upstream Brain2Qwerty/NeuralSet environment rather than copying the entire upstream repository into EEG2Qwerty.

## Build aligned EEG events

```bash
python scripts/build_all_eeg_events.py \
  --data-root /path/to/SpanishBCBL
```

## Audit training events and split integrity

```bash
python scripts/audit_eeg_training_events.py \
  --data-root /path/to/SpanishBCBL \
  --cache-root /path/to/cache
```

## Evaluate sentence predictions

For a CSV with `subject`, `reference`, and `prediction` columns:

```bash
python scripts/evaluate_predictions.py predictions.csv
```

Use `--output` to save the JSON summary.

No dataset files, checkpoints, or raw training outputs are committed to this repository.
