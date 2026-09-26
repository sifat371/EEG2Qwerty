# Configurations

Configurations in this directory define **public benchmark candidates** and their protocol metadata. A config does not imply that a result has already been validated.

## Current public baseline configs

### `m2_whole_sentence_typed.yaml`

- standardized implementation of historical M2
- whole-sentence batching
- observed/pressed `button` target
- no external language model
- typo-error filter disabled by default to preserve the current released-code target/event set

### `m2_whole_sentence_intended.yaml`

- same standardized M2 architecture
- whole-sentence batching
- SequenceMatcher-aligned intended/reference target
- paper-stated maximum of ten typographical errors per sentence
- no external language model

The two configs answer different questions and their CERs should not be mixed into a single benchmark column.

Historical experiment scripts retain their original command-line arguments under `experiments/historical/`.
