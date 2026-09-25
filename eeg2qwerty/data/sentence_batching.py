from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler


def _extra(segment: Any) -> dict[str, Any]:
    trigger = getattr(segment, "trigger", None)
    extra = getattr(trigger, "extra", None)

    if isinstance(extra, dict):
        return extra

    try:
        return dict(extra)
    except Exception:
        return {}


def sentence_key(
    segment: Any,
    fallback_index: int,
) -> tuple[str, str]:
    """
    Return a subject-aware sentence identity.

    sentence_UID should not be assumed globally unique across every
    participant/recording, so subject is included in the key.
    """

    extra = _extra(segment)

    subject = str(
        extra.get(
            "subject",
            "unknown_subject",
        )
    )
    sentence_uid = str(
        extra.get(
            "sentence_UID",
            f"unknown_sentence_{fallback_index}",
        )
    )

    return subject, sentence_uid


class WholeSentenceBatchSampler(Sampler[list[int]]):
    """
    Pack complete sentences into batches bounded by a keystroke budget.

    A normal fixed-size batch can cut a contiguous sentence at the batch
    boundary. This sampler treats each sentence as an indivisible group.
    A sentence longer than max_keystrokes is yielded intact by itself.
    """

    def __init__(
        self,
        segments: list[Any],
        max_keystrokes: int,
        seed: int = 33,
        shuffle_sentences: bool = False,
    ) -> None:
        super().__init__()

        if max_keystrokes < 1:
            raise ValueError("max_keystrokes must be positive")

        self.max_keystrokes = int(max_keystrokes)
        self.seed = int(seed)
        self.shuffle_sentences = bool(shuffle_sentences)
        self.epoch = 0

        groups: OrderedDict[
            tuple[str, str],
            list[int],
        ] = OrderedDict()

        for index, segment in enumerate(segments):
            key = sentence_key(
                segment,
                fallback_index=index,
            )
            groups.setdefault(key, []).append(index)

        self.groups = list(groups.values())
        self._batch_count = self._count_batches(self.groups)

    def _ordered_groups(self) -> list[list[int]]:
        if not self.shuffle_sentences:
            return self.groups

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        order = torch.randperm(
            len(self.groups),
            generator=generator,
        ).tolist()

        return [self.groups[index] for index in order]

    def _pack(
        self,
        groups: list[list[int]],
    ) -> Iterator[list[int]]:
        current_batch: list[int] = []

        for group in groups:
            if (
                current_batch
                and len(current_batch) + len(group) > self.max_keystrokes
            ):
                yield current_batch
                current_batch = []

            if not current_batch and len(group) > self.max_keystrokes:
                yield list(group)
                continue

            current_batch.extend(group)

        if current_batch:
            yield current_batch

    def _count_batches(
        self,
        groups: list[list[int]],
    ) -> int:
        return sum(1 for _ in self._pack(groups))

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._pack(self._ordered_groups())

    def __len__(self) -> int:
        return self._batch_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def rebuild_with_whole_sentences(
    loader: DataLoader,
    max_keystrokes: int,
    seed: int,
    shuffle_sentences: bool = False,
) -> DataLoader:
    """
    Reuse an existing dataset/collate function while replacing only its
    ordinary fixed-size batching with whole-sentence batching.
    """

    dataset = loader.dataset
    segments = getattr(dataset, "segments", None)

    if segments is None:
        raise TypeError("Expected the loader dataset to expose .segments")

    batch_sampler = WholeSentenceBatchSampler(
        segments=list(segments),
        max_keystrokes=max_keystrokes,
        seed=seed,
        shuffle_sentences=shuffle_sentences,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=getattr(dataset, "collate_fn", None),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=(
            loader.persistent_workers
            and loader.num_workers > 0
        ),
    )


def assert_complete_sentences(
    loader: DataLoader,
    maximum_batches: int = 0,
) -> dict[str, int]:
    """Check whether any sentence identity appears in multiple batches."""

    seen_in_batch: dict[tuple[str, str], int] = {}
    duplicate_across_batches = 0
    sentence_instances = 0

    for batch_index, batch in enumerate(loader):
        keys_in_batch: set[tuple[str, str]] = set()

        for segment_index, segment in enumerate(batch.segments):
            key = sentence_key(
                segment,
                fallback_index=segment_index,
            )
            keys_in_batch.add(key)

        sentence_instances += len(keys_in_batch)

        for key in keys_in_batch:
            previous = seen_in_batch.get(key)

            if previous is not None and previous != batch_index:
                duplicate_across_batches += 1
            else:
                seen_in_batch[key] = batch_index

        if maximum_batches > 0 and batch_index + 1 >= maximum_batches:
            break

    return {
        "unique_sentences": len(seen_in_batch),
        "sentence_instances": sentence_instances,
        "sentences_split_across_batches": duplicate_across_batches,
    }
