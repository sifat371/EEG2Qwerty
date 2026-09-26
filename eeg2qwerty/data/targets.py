from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import torch

from .target_alignment import align_typed_to_reference


IGNORE_INDEX = -100

CHAR_TO_INDEX = {
    "s": 0,
    "o": 1,
    "t": 2,
    "e": 3,
    "n": 4,
    "c": 5,
    "i": 6,
    "a": 7,
    " ": 8,
    "d": 9,
    "l": 10,
    "r": 11,
    "b": 12,
    "@": 13,
    "z": 14,
    "v": 15,
    "f": 16,
    "m": 17,
    "u": 18,
    "h": 19,
    "p": 20,
    "g": 21,
    "q": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "j": 26,
    "k": 27,
    "9": 28,
}


def canonical_key(value: object) -> str:
    text = str(value)
    return {
        "<space>": " ",
        "<special>": "@",
        "<number>": "9",
        "ý": "@",
        "ü": "@",
        "û": "@",
        "£": "@",
        "¤": "@",
        "-": "@",
        "¿": "@",
        "`": "@",
    }.get(text, text.lower())


def _extra(segment: Any) -> dict[str, Any]:
    trigger = getattr(segment, "trigger", None)
    extra = getattr(trigger, "extra", None)
    if isinstance(extra, dict):
        return extra
    try:
        return dict(extra)
    except Exception:
        return {}


def _sentence_key(segment: Any, index: int) -> tuple[str, str, str, str]:
    extra = _extra(segment)
    return (
        str(extra.get("subject", "unknown_subject")),
        str(extra.get("session", "unknown_session")),
        str(extra.get("task", extra.get("run", "unknown_task"))),
        str(extra.get("sentence_UID", f"unknown_sentence_{index}")),
    )


def intended_targets_for_segments(
    segments: Sequence[Any],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create one intended-character target per observed keystroke.

    Complete sentences must be present in segments. Extra typed characters
    that cannot be paired to a reference position receive IGNORE_INDEX.
    """

    grouped: OrderedDict[tuple[str, str, str, str], list[int]] = OrderedDict()
    for index, segment in enumerate(segments):
        grouped.setdefault(_sentence_key(segment, index), []).append(index)

    targets = torch.full(
        (len(segments),),
        IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )

    for indexes in grouped.values():
        extras = [_extra(segments[index]) for index in indexes]
        typed = "".join(canonical_key(extra.get("button", "")) for extra in extras)

        reference_candidates = [
            extra.get("true_sequence")
            for extra in extras
            if extra.get("true_sequence") is not None
        ]
        if not reference_candidates:
            continue

        reference = str(reference_candidates[0])
        alignment = align_typed_to_reference(typed, reference)

        if len(alignment.target_by_typed_position) != len(indexes):
            raise RuntimeError(
                "Alignment length does not match the sentence keystroke count."
            )

        for local_index, target_char in enumerate(
            alignment.target_by_typed_position
        ):
            if target_char is None:
                continue
            class_index = CHAR_TO_INDEX.get(canonical_key(target_char))
            if class_index is None:
                continue
            targets[indexes[local_index]] = class_index

    return targets
