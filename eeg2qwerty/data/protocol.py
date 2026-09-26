from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .target_alignment import align_typed_to_reference
from .targets import canonical_key


@dataclass(frozen=True)
class SentenceProtocolAudit:
    typed: str
    reference: str
    errors: int
    substitutions: int
    typed_insertions: int
    missing_reference: int
    aligned_coverage: float


def audit_sentence_rows(rows) -> SentenceProtocolAudit:
    """Audit one ordered sentence group from a pandas-like table."""

    typed = "".join(canonical_key(value) for value in rows["button"].tolist())

    reference_values = [
        value
        for value in rows["true_sequence"].tolist()
        if value is not None and str(value) != "nan"
    ]
    if not reference_values:
        raise ValueError("Sentence group has no true_sequence value.")

    alignment = align_typed_to_reference(typed, str(reference_values[0]))
    return SentenceProtocolAudit(
        typed=alignment.typed,
        reference=alignment.reference,
        errors=alignment.counts.errors,
        substitutions=alignment.counts.substitutions,
        typed_insertions=alignment.counts.typed_insertions,
        missing_reference=alignment.counts.missing_reference,
        aligned_coverage=alignment.coverage,
    )


def sentence_group_columns(events) -> list[str]:
    columns = [
        column
        for column in ("subject", "session", "task", "run", "sentence_UID")
        if column in events.columns
    ]
    if "sentence_UID" not in columns:
        raise ValueError("Events table does not contain sentence_UID.")
    return columns


def apply_typo_error_filter(
    events,
    *,
    max_errors: int = 10,
):
    """
    Return a copy of events containing only sentences with at most max_errors.

    This implements the paper's stated rule that sentences with strictly more
    than ten typographical errors are removed. It requires true_sequence and is
    therefore a paper-aligned protocol option, not a property of the current
    released-code typed-key target.
    """

    import pandas as pd

    if "true_sequence" not in events.columns:
        raise ValueError("Events table does not contain true_sequence.")

    group_columns = sentence_group_columns(events)
    keystrokes = events[events["type"] == "Keystroke"].copy()

    keep_keys: set[tuple[Any, ...]] = set()

    for key, group in keystrokes.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        order_columns = [
            column
            for column in ("start", "start_sample", "button_UID", "button_unique_id")
            if column in group.columns
        ]
        if order_columns:
            group = group.sort_values(order_columns[0], kind="stable")

        audit = audit_sentence_rows(group)
        if audit.errors <= max_errors:
            keep_keys.add(tuple(key))

    keys = events[group_columns].apply(lambda row: tuple(row.tolist()), axis=1)
    mask = keys.isin(keep_keys)

    filtered = events.loc[mask].copy()
    filtered.attrs.update(events.attrs)
    filtered.attrs["eeg2qwerty_max_typographical_errors"] = max_errors
    return filtered
