# EEG2Qwerty

[![Public checks](https://github.com/sifat371/EEG2Qwerty/actions/workflows/syntax.yml/badge.svg)](https://github.com/sifat371/EEG2Qwerty/actions/workflows/syntax.yml)

**Scalable neural decoding of typed sentences from EEG**

EEG2Qwerty is an independent research framework for studying neural architectures for **keypress-aligned typed-sentence decoding from non-invasive EEG**. It builds on the public Brain2Qwerty v1 EEG benchmark and separates historical architecture experiments from a cleaner, reproducible evaluation framework.

> **Scope:** the current benchmark uses EEG windows aligned to known keypress events. EEG2Qwerty is not unrestricted continuous thought-to-text decoding.

## Public research boundary

This repository contains completed public-facing work, reproducibility infrastructure, and benchmark tooling.

Ongoing unpublished hypotheses, model designs, and planned ablations are intentionally kept outside the public repository until they are ready for release.

## Quick start

Clone and install the reusable EEG2Qwerty package:

```bash
git clone https://github.com/sifat371/EEG2Qwerty.git
cd EEG2Qwerty
python -m pip install -e .
```

Run the lightweight public tests:

```bash
python -m pip install pytest
pytest
```

Full Brain2Qwerty EEG reproduction additionally requires the upstream Brain2Qwerty stack and SpanishBCBL data. See [docs/reproduction.md](docs/reproduction.md).

## What is included

### Reusable public framework

`eeg2qwerty/` currently provides:

- whole-sentence batching utilities
- character error rate utilities
- participant-level CER aggregation
- reusable evaluation helpers

### Historical experiment lineage

`experiments/historical/` preserves source snapshots from completed development experiments:

| Experiment | Approx. parameters | Validation CER | Test CER | Public description |
|---|---:|---:|---:|---|
| M0 | 2.679M | 75.30% | 74.74% | Compact EEGNet-style baseline |
| M1 | 2.684M | 75.10% | 74.56% | Temporal-attention variant |
| **M2** | **1.640M** | **72.39%** | **72.20%** | Balanced compact encoder |
| M3 | 13.69M | 76.78% | 76.58% | Geometry-aware spatial fusion |
| M4 | 6.642M | 73.15% | 72.81% | Residual geometry variant |
| G1 | — | 72.19% | — | Graph-temporal ablation |

**Important:** these are historical development results produced with an earlier internal pipeline. They are preserved for provenance and are **not presented as directly comparable** with the final published Brain2Qwerty EEG result. Standardized results will be reported separately after reproduction under the current benchmark protocol.

See [docs/history.md](docs/history.md).

## Current status

The safe historical migration is complete for M0–M4 and G1.

The public framework is now focused on establishing a standardized Brain2Qwerty-v1-aligned EEG baseline with:

- explicit typed-key versus intended/stimulus target definitions
- complete-sentence batching/evaluation
- participant-level reporting
- reproducibility metadata
- model-size and compute/resource reporting
- separation of neural-only and LM-assisted results

A candidate public baseline configuration is available at [configs/m2_whole_sentence_typed.yaml](configs/m2_whole_sentence_typed.yaml). It is a reproducibility template, **not yet a validated standardized result**.

## Repository layout

```text
EEG2Qwerty/
├── eeg2qwerty/               # reusable public framework
│   ├── data/
│   └── metrics/
├── experiments/
│   └── historical/
│       ├── m0/
│       ├── m1/
│       ├── m2/
│       ├── m3/
│       ├── m4/
│       └── g1/
├── configs/                  # public benchmark candidate configs
├── scripts/                  # data audit and evaluation CLIs
├── tests/                    # public framework tests
├── patches/                  # minimal upstream adaptation record
├── results/                  # curated result registries
├── docs/
├── pyproject.toml
├── NOTICE
├── CITATION.cff
└── LICENSE
```

## Benchmark principles

EEG2Qwerty keeps different evaluation questions separate:

- **typed-key decoding** versus **intended/stimulus decoding**
- **neural-only** decoding versus **language-model-assisted** decoding
- sentence-level metrics versus participant-level aggregation
- known-participant experiments versus held-out-participant experiments
- historical results versus standardized reproduced results

See [docs/benchmark.md](docs/benchmark.md).

## Public utilities

Once the upstream Brain2Qwerty environment and dataset are available:

```bash
python scripts/build_all_eeg_events.py \
  --data-root /path/to/SpanishBCBL

python scripts/audit_eeg_training_events.py \
  --data-root /path/to/SpanishBCBL \
  --cache-root /path/to/cache
```

Evaluate a sentence-prediction CSV containing `subject`, `reference`, and `prediction`:

```bash
python scripts/evaluate_predictions.py predictions.csv
```

## Data

The project uses the public **SpanishBCBL** dataset released for Brain2Qwerty v1. EEG2Qwerty does not redistribute the dataset.

Dataset:

- https://huggingface.co/datasets/bcbl190626/SpanishBCBL

The data were collected by and belong to the Basque Center on Cognition, Brain and Language (BCBL). Refer to the upstream project and dataset card for authoritative data documentation and terms.

## Upstream project and attribution

EEG2Qwerty builds on the public Brain2Qwerty research ecosystem released by Meta FAIR:

- Brain2Qwerty: https://github.com/facebookresearch/brain2qwerty
- *Non-invasive decoding of typed sentences from human brain activity*, Nature Neuroscience (2026)

The upstream repository already provides the public `Pinet2024Eeg` SpanishBCBL study implementation; EEG2Qwerty does not duplicate that loader.

EEG2Qwerty is an **independent research project** and is not affiliated with or endorsed by Meta Platforms, Inc., Meta FAIR, or BCBL.

## Licensing

The upstream Brain2Qwerty source is distributed under **CC BY-NC 4.0**. EEG2Qwerty preserves upstream attribution and uses a compatible non-commercial license for the public research code.

Files substantially adapted from upstream remain subject to applicable upstream copyright and license terms.

See [NOTICE](NOTICE) and [LICENSE](LICENSE).

## Reproducibility policy

A standardized reportable experiment should record at least:

- EEG2Qwerty commit SHA
- upstream Brain2Qwerty commit SHA
- dataset split/protocol
- target definition
- sentence batching strategy
- evaluator version
- config and random seed
- parameter count
- peak VRAM
- training time
- sentence CER
- participant-level CER statistics
- language-model stage, if any

The registry schema lives in [results/experiment_registry.csv](results/experiment_registry.csv).

---

### Research integrity note

Historical measurements are retained rather than silently rewritten. Standardized reproductions should be added alongside them so the public record distinguishes what was originally observed from what is later reproduced under the corrected benchmark.
