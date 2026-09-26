from dataclasses import dataclass

from eeg2qwerty.data.sentence_batching import (
    WholeSentenceBatchSampler,
    sentence_key,
)


@dataclass
class Trigger:
    extra: dict


@dataclass
class Segment:
    trigger: Trigger


def make_segment(subject, sentence_uid, *, session="1", task="task1"):
    return Segment(
        Trigger(
            {
                "subject": subject,
                "session": session,
                "task": task,
                "sentence_UID": sentence_uid,
            }
        )
    )


def test_sampler_never_splits_sentence():
    segments = (
        [make_segment("s1", "A") for _ in range(3)]
        + [make_segment("s1", "B") for _ in range(2)]
        + [make_segment("s1", "C") for _ in range(7)]
    )

    sampler = WholeSentenceBatchSampler(
        segments=list(segments),
        max_keystrokes=5,
    )

    batches = list(sampler)

    assert [len(batch) for batch in batches] == [5, 7]
    assert batches[0] == [0, 1, 2, 3, 4]
    assert batches[1] == [5, 6, 7, 8, 9, 10, 11]
    assert sampler.sentence_count == 3
    assert sampler.longest_sentence == 7


def test_sentence_identity_is_subject_and_recording_aware():
    a = make_segment("s1", "A", session="1", task="task1")
    b = make_segment("s2", "A", session="1", task="task1")
    c = make_segment("s1", "A", session="2", task="task1")

    assert sentence_key(a, 0) != sentence_key(b, 1)
    assert sentence_key(a, 0) != sentence_key(c, 2)
