# Public EEG data utilities

These scripts are cleaned, path-agnostic versions of utilities used during the local Brain2Qwerty EEG reproduction work.

They intentionally depend on the upstream Brain2Qwerty/NeuralSet environment rather than copying the entire upstream repository into EEG2Qwerty.

Example:

```bash
python scripts/build_all_eeg_events.py \
  --data-root /path/to/Brain2Qwerty_EEG

python scripts/audit_eeg_training_events.py \
  --data-root /path/to/Brain2Qwerty_EEG \
  --cache-root /path/to/Brain2Qwerty_cache
```

No dataset files are committed to this repository.
