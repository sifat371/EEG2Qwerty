import pytest

from eeg2qwerty.metrics import (
    PredictionRecord,
    participant_cer,
    pooled_cer,
    summarize_participants,
)


def test_participant_cer_is_pooled_by_reference_characters():
    records = [
        PredictionRecord("s1", "abc", "abc"),
        PredictionRecord("s1", "de", "dx"),
        PredictionRecord("s2", "abcd", "abxd"),
    ]

    results = {item.subject: item for item in participant_cer(records)}

    assert results["s1"].cer == pytest.approx(1 / 5)
    assert results["s1"].sentences == 2
    assert results["s2"].cer == pytest.approx(1 / 4)
    assert pooled_cer(records) == pytest.approx(2 / 9)

    summary = summarize_participants(results.values())
    assert summary.count == 2
