# Dataset and Label Notes

EEG2Qwerty uses the public SpanishBCBL data released for Brain2Qwerty v1. The dataset is not redistributed in this repository.

## Public dataset location

https://huggingface.co/datasets/bcbl190626/SpanishBCBL

## EEG task used in this project

The local EEG adaptation used keypress-aligned EEG windows with the Brain2Qwerty v1 character vocabulary.

During pipeline auditing, the event metadata exposed fields including:

- `button` — observed keypress event
- `sentence_typed` — typed sequence metadata
- `true_sequence` — intended/reference sequence
- `sentence_UID` — sentence identifier

These fields should not be assumed interchangeable.

## Sentence integrity

An important implementation detail is that an ordinary fixed-size DataLoader can cut a contiguous sentence at a batch boundary. Standardized EEG2Qwerty evaluation therefore treats complete sentences as evaluation units.

## Data ownership

The data were collected by and belong to BCBL. Refer to the upstream Brain2Qwerty repository and dataset card for authoritative dataset terms and documentation.
