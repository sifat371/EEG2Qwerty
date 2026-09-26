# Contributing

EEG2Qwerty is currently an early-stage research repository.

Public contributions are welcome when they improve:

- reproducibility
- dataset/pipeline validation
- evaluation correctness
- documentation
- tests
- efficiency measurement
- completed public historical experiment reproduction

## Before opening a pull request

Run:

```bash
python -m pip install -e .
python -m pip install pytest
pytest
python -m compileall -q eeg2qwerty scripts experiments/historical
```

## Research boundary

Please do not use public issues or pull requests to speculate about, reconstruct, or solicit unpublished EEG2Qwerty model directions.

The public repository intentionally contains completed public-facing work and reproducibility infrastructure, while unpublished research designs are developed separately.

## Data and artifacts

Do not commit:

- SpanishBCBL dataset files
- participant recordings
- checkpoints
- raw experiment logs
- credentials/tokens
- machine-specific local paths

Curated aggregate metrics may be added under `results/` when their protocol and provenance are documented.

## Upstream attribution

Some workflows and historical code build upon Brain2Qwerty. Preserve applicable upstream copyright/license notices and follow the repository's CC BY-NC 4.0 licensing terms.
