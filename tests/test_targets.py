from dataclasses import dataclass

from eeg2qwerty.data.targets import (
    IGNORE_INDEX,
    intended_targets_for_segments,
)


@dataclass
class Trigger:
    extra: dict


@dataclass
class Segment:
    trigger: Trigger


def make_segment(button: str, uid: str, reference: str):
    return Segment(
        Trigger(
            {
                "subject": "s1",
                "session": "1",
                "task": "task1",
                "sentence_UID": uid,
                "button": button,
                "true_sequence": reference,
            }
        )
    )


def test_intended_targets_align_substitution_to_reference():
    segments = [
        make_segment("h", "A", "hola"),
        make_segment("i", "A", "hola"),
        make_segment("l", "A", "hola"),
        make_segment("a", "A", "hola"),
    ]
    targets = intended_targets_for_segments(segments)
    assert targets.tolist() == [19, 1, 10, 7]


def test_unmatched_extra_key_is_ignored():
    segments = [
        make_segment("h", "A", "hola"),
        make_segment("o", "A", "hola"),
        make_segment("l", "A", "hola"),
        make_segment("a", "A", "hola"),
        make_segment("a", "A", "hola"),
    ]
    targets = intended_targets_for_segments(segments)
    assert targets[-1].item() == IGNORE_INDEX
