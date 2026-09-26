# Reproduction Notes

## 1. Install EEG2Qwerty

```bash
git clone https://github.com/sifat371/EEG2Qwerty.git
cd EEG2Qwerty
python -m pip install -e .
```

For tests:

```bash
python -m pip install pytest
pytest
```

## 2. Install the upstream Brain2Qwerty stack

The EEG data utilities and standardized training runner intentionally depend on the public upstream Brain2Qwerty/NeuralSet stack rather than copying that repository into EEG2Qwerty.

```bash
python -m pip install -e ".[brain2qwerty]"

git clone https://github.com/facebookresearch/brain2qwerty.git
cd brain2qwerty
git checkout 5f9889621d0df391c5aab37c996683d308e6e926
UPSTREAM_SHA=$(git rev-parse HEAD)
python -m pip install -e .
cd ../EEG2Qwerty
echo "$UPSTREAM_SHA"
```

The baseline configs already pin this reviewed upstream revision. Use `--upstream-commit` only when deliberately testing a different upstream revision. See [../UPSTREAM.md](../UPSTREAM.md).

The upstream repository already includes the public `Pinet2024Eeg` study implementation. EEG2Qwerty does not duplicate it.

## 3. Prepare SpanishBCBL

EEG2Qwerty does not redistribute the dataset.

After obtaining the public SpanishBCBL data, keep it outside the Git repository, for example:

```text
/data/SpanishBCBL
/data/Brain2Qwerty_cache
```

## 4. Audit the final sentence/target protocol

Metadata-only audit:

```bash
python scripts/audit_standardized_protocol.py \
  --data-root /data/SpanishBCBL \
  --cache-root /data/Brain2Qwerty_cache \
  --max-typographical-errors 10 \
  --output results/raw/protocol_audit.json
```

Full loader-integrity audit:

```bash
python scripts/audit_standardized_protocol.py \
  --data-root /data/SpanishBCBL \
  --cache-root /data/Brain2Qwerty_cache \
  --max-typographical-errors 10 \
  --check-loaders \
  --output results/raw/protocol_audit_with_loaders.json
```

The full audit raises an error if any sentence appears across multiple batches.

## 5. Debug the standardized typed-key baseline

```bash
python scripts/train_standardized_m2.py \
  --config configs/m2_whole_sentence_typed.yaml \
  --data-root /data/SpanishBCBL \
  --cache-root /data/Brain2Qwerty_cache \
  --output-dir results/raw/m2_typed_debug \
  --num-workers 4 \
  --debug
```

A debug run is recorded with status `debug` and must not be copied into the curated result registry.

## 6. Run the standardized typed-key baseline

```bash
python scripts/train_standardized_m2.py \
  --config configs/m2_whole_sentence_typed.yaml \
  --data-root /data/SpanishBCBL \
  --cache-root /data/Brain2Qwerty_cache \
  --output-dir results/raw/m2_typed_seed33
```

## 7. Run the paper-aligned intended/reference baseline

```bash
python scripts/train_standardized_m2.py \
  --config configs/m2_whole_sentence_intended.yaml \
  --data-root /data/SpanishBCBL \
  --cache-root /data/Brain2Qwerty_cache \
  --output-dir results/raw/m2_intended_seed33
```

## 8. Run outputs

Each non-debug training run writes:

```text
best-cer.ckpt
best-loss.ckpt
last.ckpt
test_predictions.csv
run_summary.json
registry_row.csv
logs/
```

The run summary records:

- EEG2Qwerty commit
- upstream Brain2Qwerty commit
- target protocol
- typo threshold
- seed
- parameter count
- split keystroke counts
- hardware
- precision
- peak VRAM
- training/test runtime
- participant-level CER metrics
- pooled CER
- checkpoint paths

## 9. Independently evaluate exported predictions

```bash
python scripts/evaluate_predictions.py \
  results/raw/m2_typed_seed33/test_predictions.csv \
  --output results/raw/m2_typed_seed33/prediction_metrics.json
```

## 10. Promote a result into the curated registry

The training script writes `registry_row.csv` with status `candidate_reproduction`.

Promote it to `standardized` only after:

1. the final protocol audit passed,
2. no sentence is split across loader batches,
3. commit/config/seed provenance is complete,
4. participant-level metrics were inspected,
5. the row corresponds to a full non-debug run,
6. any external language-model stage is reported separately.

Historical numbers remain in `historical_results.csv`; do not overwrite them with standardized reproductions.
