# Reproduction Notes

## Installation

The reusable public package can be installed directly from a clone:

```bash
git clone https://github.com/sifat371/EEG2Qwerty.git
cd EEG2Qwerty
python -m pip install -e .
```

For development/tests:

```bash
python -m pip install pytest
pytest
```

## Upstream Brain2Qwerty environment

The EEG reproduction utilities intentionally depend on the public upstream Brain2Qwerty stack rather than copying that repository into EEG2Qwerty.

The current upstream repository already contains the public `Pinet2024Eeg` SpanishBCBL study implementation. EEG2Qwerty therefore does not duplicate that study loader.

Install upstream Brain2Qwerty separately when running the EEG data/reproduction scripts:

```bash
python -m pip install "eeg2qwerty[brain2qwerty]"
python -m pip install "git+https://github.com/facebookresearch/brain2qwerty.git"
```

For strict reproduction work, record the exact upstream Brain2Qwerty commit SHA alongside the EEG2Qwerty commit SHA.

## Historical experiments

The source under `experiments/historical/` preserves completed development snapshots separately from the standardized framework.

Historical scripts were originally run inside a local Brain2Qwerty checkout and may retain imports such as `brain2qwerty_v1` or the former `liteqwerty_eeg` package namespace. They are intentionally not rewritten to masquerade as new standardized implementations.

Historical results should be interpreted as development records, not final benchmark numbers.

## Public EEG utilities

The path-agnostic utilities under `scripts/` can be run once the upstream Brain2Qwerty environment and SpanishBCBL data are available.

For example:

```bash
python scripts/build_all_eeg_events.py \
  --data-root /path/to/SpanishBCBL

python scripts/audit_eeg_training_events.py \
  --data-root /path/to/SpanishBCBL \
  --cache-root /path/to/cache
```

## Prediction evaluation

A sentence-prediction CSV containing `subject`, `reference`, and `prediction` columns can be summarized with:

```bash
python scripts/evaluate_predictions.py predictions.csv \
  --output results/raw/prediction_metrics.json
```

The evaluator reports pooled CER plus participant-level CER distribution statistics.

## Reproduction policy

A result should be promoted from **historical** to **standardized** only after:

1. the dataset/split and target protocol are documented,
2. complete-sentence batching/evaluation is used,
3. the exact EEG2Qwerty and upstream Brain2Qwerty commit SHAs are recorded,
4. config and random seed are recorded,
5. participant-level metrics are retained,
6. compute/resource metadata are recorded,
7. the run can be reproduced from the public code.

Candidate public configurations live under `configs/`; a config does not by itself imply a validated result.
