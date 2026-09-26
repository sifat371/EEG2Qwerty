import pytest

from eeg2qwerty.metrics import (
    levenshtein_distance,
    normalized_cer,
    summarize_cer,
)


def test_levenshtein_exact_match():
    assert levenshtein_distance("abc", "abc") == 0


def test_levenshtein_substitution():
    assert levenshtein_distance("abc", "axc") == 1
    assert normalized_cer("abc", "axc") == pytest.approx(1 / 3)


def test_normalized_cer_rejects_empty_reference():
    with pytest.raises(ValueError):
        normalized_cer("", "abc")


def test_cer_summary():
    summary = summarize_cer([0.0, 0.5, 1.0])

    assert summary.count == 3
    assert summary.mean == pytest.approx(0.5)
    assert summary.median == pytest.approx(0.5)
    assert summary.minimum == pytest.approx(0.0)
    assert summary.maximum == pytest.approx(1.0)
