# Dataset and Label Notes

EEG2Qwerty uses the public SpanishBCBL data released for Brain2Qwerty v1. The dataset is not redistributed in this repository.

## Public dataset location

https://huggingface.co/datasets/bcbl190626/SpanishBCBL

## EEG task used in this project

The EEG benchmark uses keypress-aligned windows with the Brain2Qwerty v1 character vocabulary.

The public upstream Brain2Qwerty repository already provides the `Pinet2024Eeg` study implementation used to expose the SpanishBCBL EEG recordings.

## Label fields

During pipeline auditing, the event metadata exposed fields including:

- `button` — observed keypress event
- `sentence_typed` — typed sequence metadata
- `true_sequence` — intended/reference sequence
- `sentence_UID` — sentence identifier

These fields should not be assumed interchangeable.

## Sentence integrity

An ordinary fixed-size DataLoader can cut a contiguous sentence at a batch boundary. Standardized EEG2Qwerty evaluation therefore treats complete sentences as indivisible batching/evaluation units.

The public `WholeSentenceBatchSampler` groups using participant and recording context plus `sentence_UID` so that a sentence is not accidentally merged with a same-named sentence from another recording.

## Local training-split audit

A complete global audit of the development training split reconstructed:

- 116,751 EEG-aligned keystroke events
- 3,175 complete sentence instances
- 224 sentences that crossed the earlier fixed-size input batch boundaries
- 100% coverage of the intended/reference sequence field in that audited split

These values describe the audited local split/pipeline and are recorded as implementation provenance, not as a replacement for the upstream dataset card.

## Data ownership

The data were collected by and belong to BCBL. Refer to the upstream Brain2Qwerty repository and dataset card for authoritative dataset terms and documentation.
