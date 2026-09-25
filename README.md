# EEG2Qwerty

**Scalable neural decoding of typed sentences from EEG**

EEG2Qwerty is an independent research framework for studying neural architectures for **keypress-aligned typed-sentence decoding from non-invasive EEG**. The project builds on the public Brain2Qwerty v1 EEG benchmark and focuses on reproducible evaluation across model accuracy, computational efficiency, and participant-level consistency.

> **Scope.** The current benchmark uses EEG windows aligned to known keypress events. EEG2Qwerty should not be interpreted as unrestricted continuous thought-to-text decoding.

## Why this repository exists

This repository serves two purposes:

1. **Research record** — preserve completed EEG decoding experiments, including architectures that did not improve the benchmark.
2. **Reproducible foundation** — provide a clean base for standardized EEG2Qwerty training and evaluation going forward.

The repository intentionally separates historical experiments from future release models. Experimental names such as M0, M1, M2, M3, M4, and G1 record the research lineage; they are **not** product/model-size labels.

## Current status

The project is currently standardizing the Brain2Qwerty-v1 EEG evaluation pipeline before promoting any architecture into an official EEG2Qwerty model family.

Current work includes:

- EEG adaptation of the public Brain2Qwerty v1 pipeline
- compact EEG encoders and contextual decoders
- subject-conditioned adaptation
- temporal, spatial, and graph-based ablations
- sentence-level and participant-level evaluation
- whole-sentence batching and dataset auditing
- accuracy/efficiency benchmarking

Unpublished research directions and experimental hypotheses are intentionally maintained outside the public repository until they are ready for release.

**Migration status:** the repository foundation and historical M0–M2 source are now public. Additional completed, non-sensitive historical experiments are being migrated selectively rather than copying the original development checkout wholesale.

## Historical experiment lineage

| Experiment | Approx. parameters | Validation CER | Test CER | Public description |
|---|---:|---:|---:|---|
| M0 | 2.679M | 75.30% | 74.74% | Compact EEGNet-style baseline |
| M1 | 2.684M | 75.10% | 74.56% | Temporal-attention variant |
| **M2** | **1.640M** | **72.39%** | **72.20%** | Balanced compact encoder |
| M3 | 13.69M | 76.78% | 76.58% | Geometry-aware spatial fusion |
| M4 | 6.642M | 73.15% | 72.81% | Residual geometry variant |
| G1 | — | 72.19% | — | Graph-temporal ablation |

**Important:** these are historical development results produced with an earlier internal pipeline. They are preserved for research provenance and should **not** yet be treated as directly comparable with the final published Brain2Qwerty EEG result. Standardized benchmark results will be reported separately as models are reproduced under the current evaluation protocol. See [docs/history.md](docs/history.md) for the public development lineage.

## Repository layout

```text
EEG2Qwerty/
├── eeg2qwerty/               # reusable public framework
├── experiments/
│   └── historical/           # completed architecture lineage
├── scripts/                  # public data/evaluation entry points
├── configs/                  # reproducible experiment configuration
├── results/                  # curated benchmark tables, not raw runs
├── docs/
│   ├── benchmark.md
│   ├── dataset.md
│   └── reproduction.md
├── NOTICE
├── CITATION.cff
└── LICENSE
```

## Benchmark principles

EEG2Qwerty keeps different evaluation questions separate:

- **typed-key decoding** versus **intended/stimulus decoding**
- **neural-only** decoding versus **language-model-assisted** decoding
- sentence-level metrics versus participant-level aggregation
- known-participant experiments versus cross-participant experiments
- historical results versus standardized reproduced results

See [docs/benchmark.md](docs/benchmark.md).

## Data

The project uses the public **SpanishBCBL** dataset released for Brain2Qwerty v1. EEG2Qwerty does not redistribute the dataset.

Upstream dataset:

- https://huggingface.co/datasets/bcbl190626/SpanishBCBL

The data belong to the Basque Center on Cognition, Brain and Language (BCBL). See the upstream Brain2Qwerty project for dataset details and terms.

## Upstream project

EEG2Qwerty is inspired by and partially builds upon the public Brain2Qwerty research code released by Meta FAIR:

- Brain2Qwerty: https://github.com/facebookresearch/brain2qwerty
- Nature Neuroscience paper: *Non-invasive decoding of typed sentences from human brain activity*

EEG2Qwerty is an **independent research project** and is not affiliated with or endorsed by Meta Platforms, Inc. or BCBL.

## Licensing

The upstream Brain2Qwerty source is distributed under **CC BY-NC 4.0**. This repository preserves upstream attribution and uses a compatible non-commercial license for the public research code. Files substantially adapted from upstream should retain their original notices.

See [NOTICE](NOTICE) and [LICENSE](LICENSE).

## Reproducibility

Historical code is retained as historical code. Going forward, reportable experiments should record at least:

- commit SHA
- dataset split/protocol
- target definition
- evaluator version
- random seed
- parameter count
- peak VRAM
- training/inference cost
- sentence CER
- participant-level CER statistics

The curated registry lives under [results/](results/).

## Project direction

EEG2Qwerty is intended to support architectures spanning compact to higher-capacity models. Public model-family names will be assigned only after candidates are reproduced under the standardized benchmark and their accuracy/resource trade-offs are measured.

---

### Research integrity note

This repository documents completed public-facing work. Ongoing unpublished hypotheses, model designs, and planned ablations are intentionally not disclosed here.
