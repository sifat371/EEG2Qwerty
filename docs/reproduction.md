# Reproduction Notes

## Historical experiments

The first public import preserves completed historical architecture code separately from the standardized framework.

Historical results should be interpreted as development records, not final benchmark numbers.

## Environment

The original experiments were developed inside a Brain2Qwerty checkout with an EEG-adapted v1 pipeline and NVIDIA GPU acceleration.

The historical scripts may therefore depend on upstream modules such as:

```text
brain2qwerty_v1
neuralset
neuraltrain
lightning
torch
```

The public repository will progressively move reusable pieces into the standalone `eeg2qwerty/` package rather than copying the complete upstream repository.

## Reproduction policy

A result should be promoted from "historical" to "standardized" only after:

1. the dataset/target protocol is documented,
2. complete-sentence evaluation is used,
3. the exact commit/config/seed is recorded,
4. participant-level metrics are retained,
5. the run is reproducible from the public code.
