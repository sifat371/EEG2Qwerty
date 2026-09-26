import pandas as pd

from eeg2qwerty.data.protocol import apply_typo_error_filter


def test_typo_filter_preserves_neural_source_rows():
    rows = [
        {
            "type": "Eeg",
            "subject": "s1",
            "session": 1,
            "task": "task1",
            "sentence_UID": None,
            "button": None,
            "true_sequence": None,
            "start": 0.0,
        },
        {
            "type": "Keystroke",
            "subject": "s1",
            "session": 1,
            "task": "task1",
            "sentence_UID": "good",
            "button": "a",
            "true_sequence": "a",
            "start": 1.0,
        },
    ]

    for index in range(11):
        rows.append(
            {
                "type": "Keystroke",
                "subject": "s1",
                "session": 1,
                "task": "task1",
                "sentence_UID": "bad",
                "button": "a",
                "true_sequence": "b" * 11,
                "start": 2.0 + index,
            }
        )

    events = pd.DataFrame(rows)
    filtered = apply_typo_error_filter(events, max_errors=10)

    assert (filtered["type"] == "Eeg").sum() == 1
    assert "good" in set(filtered["sentence_UID"].dropna())
    assert "bad" not in set(filtered["sentence_UID"].dropna())
