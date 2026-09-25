# EEG2Qwerty Benchmark Protocol

This document defines the public evaluation vocabulary for EEG2Qwerty. It intentionally describes **evaluation**, not unpublished model hypotheses.

## Task scope

The current Brain2Qwerty-v1 EEG benchmark is **keypress aligned**. Neural windows are extracted around known typing events and mapped to character representations that can then be contextualized at sentence level.

This is different from unrestricted continuous thought-to-text decoding.

## Report targets separately

### Typed-key decoding

The target is the key physically observed in the typing event stream.

### Intended/stimulus decoding

The target is the intended character derived from the reference sentence/alignment protocol.

These metrics answer different questions and must not be mixed in one headline result.

## Separate neural and language-model stages

Report:

1. **Neural-only CER**
2. **LM-assisted CER**, when an external character/language model is used

A final LM-assisted Brain2Qwerty result should not be compared directly with an EEG2Qwerty neural-only result.

## Sentence-level evaluation

Predictions should be reconstructed and evaluated as complete sentences rather than treating arbitrary loader batches as sentences.

## Participant-level reporting

A standardized result should report more than one pooled CER. At minimum, retain per-participant sentence CER values so that the following can be calculated:

- participant mean
- participant median
- standard deviation or IQR
- failure/outlier analysis

## Known-participant versus held-out-participant protocols

These are distinct experiments and should be labeled explicitly.

## Reproducibility metadata

Every standardized result should record:

- commit SHA
- config
- random seed
- split definition
- target definition
- sentence batching strategy
- evaluator version
- parameter count
- compute hardware
- peak VRAM
- training time
- inference cost when measured
