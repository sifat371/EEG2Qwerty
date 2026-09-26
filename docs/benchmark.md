# EEG2Qwerty Benchmark Protocol

This document defines the public evaluation protocol for EEG2Qwerty. It describes completed/public evaluation infrastructure only and does not disclose unpublished model hypotheses.

## Task scope

The Brain2Qwerty-v1 EEG task is **keypress aligned**. Neural windows are extracted around known typing events and decoded at character level, after which sentence context may be applied.

This is different from unrestricted continuous thought-to-text decoding.

## Two target protocols must remain separate

### A. Released-code typed-key protocol

The current public Brain2Qwerty v1 configuration encodes the observed `button` field as the feature target.

EEG2Qwerty calls this:

```text
target: typed_key
```

This protocol asks:

> Given the EEG window around a physical keypress, which key was actually pressed?

The default public typed-key candidate config does **not** apply the paper typo-error filter unless explicitly requested.

### B. Paper-aligned intended/reference protocol

The Nature Neuroscience paper states that typed sentences were aligned with the original sentence using Python `difflib.SequenceMatcher`, and sentences with strictly more than ten typographical errors were removed.

EEG2Qwerty exposes this separately as:

```text
target: intended
max_typographical_errors: 10
```

For each observed keypress, the public alignment utility pairs a reference character where SequenceMatcher provides a correspondence.

Extra typed characters with no reference correspondence are assigned `IGNORE_INDEX` for the intended-target cross-entropy loss. They remain present at inference time, so sentence CER still penalizes the resulting length/content mismatch against the complete intended reference.

This unmatched-key handling is an **explicit EEG2Qwerty implementation choice**. The paper describes the SequenceMatcher procedure but the current public v1 training code does not expose a per-keystroke intended-target field, so this repository does not claim that the private paper implementation handled every ambiguous alignment identically.

## Typographical-error filtering

The optional public paper-aligned filter keeps sentences with:

```text
typographical_errors <= 10
```

and removes sentences with strictly more than ten errors.

For transparency, EEG2Qwerty reports:

- substitutions
- extra typed characters
- missing reference characters
- aligned-position coverage
- number of sentences kept/dropped at the threshold

## Complete-sentence batching

A flat sampler followed by fixed-size batching can still cut a sentence at a batch boundary even when sentence indices are contiguous.

The standardized loaders therefore use `WholeSentenceBatchSampler`:

- a sentence is an indivisible batch unit,
- several complete sentences may share one batch,
- a sentence longer than the keystroke budget is yielded intact by itself.

Sentence identity includes participant and recording context plus `sentence_UID`.

## Neural-only versus language-model-assisted results

Report separately:

1. **Neural-only CER**
2. **LM-assisted CER**, when an external language model is used

A final LM-assisted Brain2Qwerty result must not be compared directly with an EEG2Qwerty neural-only result.

## Sentence and participant metrics

Predictions are evaluated as complete sentence strings.

For each participant:

```text
participant CER
=
sum(character edit distances over that participant's sentences)
/
sum(reference characters over that participant's sentences)
```

The primary standardized EEG2Qwerty metric is the **mean participant CER**.

Also report:

- pooled CER across all reference characters
- participant median CER
- participant CER standard deviation or IQR
- sentence count
- participant count

## Known-participant versus held-out-participant protocols

These are different experiments and must be labeled explicitly. Results from one protocol should not be used as evidence for the other.

## Result status

### historical

Recorded from the earlier development pipeline.

### debug

Smoke test only.

### candidate_reproduction

A complete standardized run has executed, but its protocol audit/result review has not yet been promoted into the curated public benchmark.

### standardized

The dataset/protocol audit passed, provenance metadata is complete, and the curated result has been added to the public registry.

## Reproducibility metadata

Every standardized result should record:

- EEG2Qwerty commit SHA
- upstream Brain2Qwerty commit SHA
- config
- random seed
- split definition
- target protocol
- typo-error threshold, if used
- sentence batching strategy
- evaluator version
- parameter count
- compute hardware
- peak VRAM
- training time
- participant-level and pooled CER
- external language-model stage, if any

## Public protocol audit

Run the full metadata audit:

```bash
python scripts/audit_standardized_protocol.py \
  --data-root /path/to/SpanishBCBL \
  --cache-root /path/to/cache \
  --max-typographical-errors 10 \
  --output results/raw/protocol_audit.json
```

To additionally traverse the actual EEG loaders and verify that no sentence crosses a batch boundary:

```bash
python scripts/audit_standardized_protocol.py \
  --data-root /path/to/SpanishBCBL \
  --cache-root /path/to/cache \
  --max-typographical-errors 10 \
  --check-loaders \
  --output results/raw/protocol_audit_with_loaders.json
```
