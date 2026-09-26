# EEG2Qwerty

[![Public checks](https://github.com/sifat371/EEG2Qwerty/actions/workflows/syntax.yml/badge.svg)](https://github.com/sifat371/EEG2Qwerty/actions/workflows/syntax.yml)

**Scalable neural decoding of typed sentences from EEG**

EEG2Qwerty is an independent research framework for **keypress-aligned typed-sentence decoding from non-invasive EEG**. It builds on the public Brain2Qwerty v1 EEG benchmark while separating historical architecture experiments from a standardized, reproducible public baseline.

> **Scope:** the current task uses EEG windows aligned to known keypress events. EEG2Qwerty is not unrestricted continuous thought-to-text decoding.

## Public research boundary

This repository contains completed public-facing work, reproducibility infrastructure, and benchmark tooling. Ongoing unpublished hypotheses, model designs, and planned ablations are kept outside the public repository until they are ready for release.

## Quick start

```bash
git clone https://github.com/sifat371/EEG2Qwerty.git
cd EEG2Qwerty
python -m pip install -e .
python -m pip install pytest
pytest
```

Full EEG reproduction/training additionally requires the upstream Brain2Qwerty stack and SpanishBCBL data. See [docs/reproduction.md](docs/reproduction.md).

## Current public status

The safe historical migration is complete for **M0, M1, M2, M3, M4, and G1**.

The standardized public foundation now includes:

- whole-sentence batching
- SequenceMatcher target alignment
- optional paper-stated typo-error filtering
- typed-key and intended/reference target protocols
- standardized M2 implementation under `eeg2qwerty/models/`
- participant-level and pooled CER
- train/validation/test protocol audit
- loader-level sentence-integrity verification
- runnable standardized M2 training CLI
- prediction export
- run manifests and candidate registry rows
- package tests and GitHub CI

No standardized CER is claimed yet because the full SpanishBCBL GPU run must be executed on the actual dataset/hardware and then promoted after audit.

## Two public baseline protocols

| Config | Training target | Typo filter | Purpose |
|---|---|---:|---|
| [m2_whole_sentence_typed.yaml](configs/m2_whole_sentence_typed.yaml) | observed pressed key | off by default | released-code-aligned typed-key baseline |
| [m2_whole_sentence_intended.yaml](configs/m2_whole_sentence_intended.yaml) | SequenceMatcher-aligned reference character | >10 errors removed | paper-aligned intended/reference baseline |

The current public Brain2Qwerty v1 code encodes the observed `button` field as its feature target, while the 2026 paper describes SequenceMatcher-based typographical-error alignment. EEG2Qwerty therefore reports these as separate protocols rather than silently treating them as identical.

See [docs/benchmark.md](docs/benchmark.md).

## Historical experiment lineage

| Experiment | Approx. parameters | Validation CER | Test CER | Public description |
|---|---:|---:|---:|---|
| M0 | 2.679M | 75.30% | 74.74% | Compact EEGNet-style baseline |
| M1 | 2.684M | 75.10% | 74.56% | Temporal-attention variant |
| **M2** | **1.640M** | **72.39%** | **72.20%** | Balanced compact encoder |
| M3 | 13.69M | 76.78% | 76.58% | Geometry-aware spatial fusion |
| M4 | 6.642M | 73.15% | 72.81% | Residual geometry variant |
| G1 | — | 72.19% | — | Graph-temporal ablation |

**Important:** these are historical development results from the earlier internal pipeline. They are preserved for provenance and are **not presented as directly comparable** with the final published Brain2Qwerty EEG result.

## Run the standardized protocol audit

```bash
python scripts/audit_standardized_protocol.py \
  --data-root /path/to/SpanishBCBL \
  --cache-root /path/to/cache \
  --max-typographical-errors 10 \
  --check-loaders \
  --output results/raw/protocol_audit.json
```

## Run the standardized M2 baseline

Typed-key protocol:

```bash
python scripts/train_standardized_m2.py \
  --config configs/m2_whole_sentence_typed.yaml \
  --data-root /path/to/SpanishBCBL \
  --cache-root /path/to/cache \
  --output-dir results/raw/m2_typed_seed33 \
  --upstream-commit <BRAIN2QWERTY_SHA>
```

Intended/reference protocol:

```bash
python scripts/train_standardized_m2.py \
  --config configs/m2_whole_sentence_intended.yaml \
  --data-root /path/to/SpanishBCBL \
  --cache-root /path/to/cache \
  --output-dir results/raw/m2_intended_seed33 \
  --upstream-commit <BRAIN2QWERTY_SHA>
```

Each run writes checkpoints, sentence predictions, a JSON run manifest, and a candidate registry row.

## Repository layout

```text
EEG2Qwerty/
├── eeg2qwerty/
│   ├── data/
│   ├── metrics/
│   ├── models/
│   └── training/
├── experiments/
│   └── historical/
│       ├── m0/
│       ├── m1/
│       ├── m2/
│       ├── m3/
│       ├── m4/
│       └── g1/
├── configs/
├── scripts/
├── tests/
├── patches/
├── results/
├── docs/
├── pyproject.toml
├── NOTICE
├── CITATION.cff
└── LICENSE
```

## Data

EEG2Qwerty uses the public **SpanishBCBL** data released for Brain2Qwerty v1 and does not redistribute it.

- Dataset: https://huggingface.co/datasets/bcbl190626/SpanishBCBL
- Upstream Brain2Qwerty: https://github.com/facebookresearch/brain2qwerty
- Nature Neuroscience paper: *Non-invasive decoding of typed sentences from human brain activity* (2026)

The data were collected by and belong to BCBL. Refer to the upstream project and dataset card for authoritative terms.

## Licensing and attribution

The upstream Brain2Qwerty source is distributed under **CC BY-NC 4.0**. EEG2Qwerty preserves upstream attribution and uses a compatible non-commercial license for its public research code.

EEG2Qwerty is independent and is not affiliated with or endorsed by Meta Platforms, Inc., Meta FAIR, or BCBL.

See [NOTICE](NOTICE), [LICENSE](LICENSE), and [docs/reproduction.md](docs/reproduction.md).

---

### Research integrity note

Historical measurements are retained rather than silently rewritten. A new run is recorded as `debug` or `candidate_reproduction` first and is promoted to the curated standardized registry only after protocol and provenance review.
