from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .sentence_batching import WholeSentenceBatchSampler


def upstream_eeg_data_config(
    *,
    data_root: Path,
    cache_root: Path,
    train_batch_keystrokes: int,
    eval_batch_keystrokes: int,
    num_workers: int,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Build a Brain2Qwerty-v1 data config adapted from the public MEG default
    to the public SpanishBCBL EEG study.
    """

    try:
        from brain2qwerty_v1.config.xp_config import experiment_config
    except ImportError as exc:
        raise RuntimeError(
            "The upstream Brain2Qwerty package is required for EEG reproduction. "
            "See docs/reproduction.md."
        ) from exc

    config = experiment_config()
    data = config["data"]

    data["study"]["name"] = "Pinet2024Eeg"
    data["study"]["path"] = str(data_root)
    data["study"]["infra"] = {"folder": str(cache_root)}
    data["study"]["infra_timelines"] = {
        "folder": str(cache_root),
        "cluster": None,
    }
    if debug:
        data["study"]["query"] = "timeline_index == 0"

    data["neuro"]["name"] = "EegExtractor"
    data["neuro"]["frequency"] = 50
    data["neuro"]["filter"] = (0.1, 20.0)
    data["neuro"]["baseline"] = (0.0, 0.2)
    data["neuro"]["clamp"] = 5
    data["neuro"]["scaler"] = "RobustScaler"
    data["neuro"]["infra"] = {
        "folder": str(cache_root),
        "cluster": None,
    }
    data["neuro"].pop("allow_maxshield", None)
    data["neuro"].pop("apply_proj", None)

    data["batch_size"] = train_batch_keystrokes
    data["val_batch_size"] = eval_batch_keystrokes
    data["test_batch_size"] = eval_batch_keystrokes
    data["num_workers"] = num_workers
    data["pin_memory"] = torch.cuda.is_available()
    data["persistent_workers"] = num_workers > 0

    return data


def build_upstream_eeg_events(
    *,
    data_root: Path,
    cache_root: Path,
    debug: bool = False,
):
    """Build the transformed Brain2Qwerty-v1 EEG event table."""

    try:
        import studies  # noqa: F401
        from brain2qwerty_v1.main import Data
    except ImportError as exc:
        raise RuntimeError(
            "The upstream Brain2Qwerty package is required for EEG reproduction. "
            "See docs/reproduction.md."
        ) from exc

    data_config = upstream_eeg_data_config(
        data_root=data_root,
        cache_root=cache_root,
        train_batch_keystrokes=256,
        eval_batch_keystrokes=512,
        num_workers=0,
        debug=debug,
    )
    data = Data(**data_config)
    return data, data.build_events()


def build_upstream_eeg_loaders(
    *,
    data_root: Path,
    cache_root: Path,
    train_batch_keystrokes: int = 256,
    eval_batch_keystrokes: int = 512,
    num_workers: int = 8,
    seed: int = 33,
    debug: bool = False,
) -> tuple[dict[str, DataLoader], Any]:
    """
    Build EEG loaders while fixing two public-v1 integration details:

    1. subject IDs follow the actual neuro extractor event type (EEG rather than
       the MEG-only literal in the current upstream Data.build implementation);
    2. batches are packed from complete sentence groups instead of relying on a
       flat sentence-grouped sampler followed by fixed-size batching.
    """

    try:
        import neuralset as ns
        import studies  # noqa: F401
        from brain2qwerty_v1.main import Data
        from brain2qwerty_v1.utils import ChannelPositions2D
    except ImportError as exc:
        raise RuntimeError(
            "The upstream Brain2Qwerty/NeuralSet stack is required. "
            "See docs/reproduction.md."
        ) from exc

    data_config = upstream_eeg_data_config(
        data_root=data_root,
        cache_root=cache_root,
        train_batch_keystrokes=train_batch_keystrokes,
        eval_batch_keystrokes=eval_batch_keystrokes,
        num_workers=num_workers,
        debug=debug,
    )
    data = Data(**data_config)

    events = data.build_events()
    data.neuro.prepare(events)
    data.feature.prepare(events)

    subject_id = ns.extractors.LabelEncoder(
        event_types=data.neuro.event_types,
        event_field="subject",
    )
    subject_id.prepare(events)

    channel_positions = ChannelPositions2D(neuro=data.neuro)
    channel_positions.prepare(events)

    extractors = {
        "neuro": data.neuro,
        "feature": data.feature,
        "subject_id": subject_id,
        "channel_positions": channel_positions,
    }

    budgets = {
        "train": train_batch_keystrokes,
        "val": eval_batch_keystrokes,
        "test": eval_batch_keystrokes,
    }

    loaders: dict[str, DataLoader] = {}

    for split, budget in budgets.items():
        mask = (events.split == split) & (events.type == "Keystroke")
        segments = ns.segments.list_segments(
            events,
            mask,
            start=data.start,
            duration=data.duration,
        )
        if not segments:
            continue

        dataset = ns.SegmentDataset(
            extractors=extractors,
            segments=segments,
            remove_incomplete_segments=True,
        )
        batch_sampler = WholeSentenceBatchSampler(
            segments=list(dataset.segments),
            max_keystrokes=budget,
            seed=seed,
            shuffle_sentences=False,
        )

        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": batch_sampler,
            "collate_fn": dataset.collate_fn,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        loaders[split] = DataLoader(**loader_kwargs)

    return loaders, events
