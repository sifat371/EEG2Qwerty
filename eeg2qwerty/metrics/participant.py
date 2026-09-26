from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cer import CERDistribution, levenshtein_distance, summarize_cer


@dataclass(frozen=True)
class PredictionRecord:
    subject: str
    reference: str
    prediction: str


@dataclass(frozen=True)
class ParticipantCER:
    subject: str
    sentences: int
    reference_characters: int
    edit_distance: int
    cer: float


def participant_cer(
    records: Iterable[PredictionRecord],
) -> list[ParticipantCER]:
    """Compute pooled character error rate independently for each participant."""

    grouped: dict[str, list[PredictionRecord]] = {}

    for record in records:
        grouped.setdefault(str(record.subject), []).append(record)

    results: list[ParticipantCER] = []

    for subject in sorted(grouped):
        subject_records = grouped[subject]
        reference_characters = sum(len(item.reference) for item in subject_records)

        if reference_characters == 0:
            raise ValueError(
                f"Participant {subject!r} has no reference characters."
            )

        edit_distance = sum(
            levenshtein_distance(item.reference, item.prediction)
            for item in subject_records
        )

        results.append(
            ParticipantCER(
                subject=subject,
                sentences=len(subject_records),
                reference_characters=reference_characters,
                edit_distance=edit_distance,
                cer=edit_distance / reference_characters,
            )
        )

    if not results:
        raise ValueError("At least one prediction record is required.")

    return results


def pooled_cer(
    records: Iterable[PredictionRecord],
) -> float:
    """Compute pooled CER across all supplied sentence records."""

    items = list(records)

    reference_characters = sum(len(item.reference) for item in items)

    if reference_characters == 0:
        raise ValueError("At least one reference character is required.")

    edit_distance = sum(
        levenshtein_distance(item.reference, item.prediction)
        for item in items
    )

    return edit_distance / reference_characters


def summarize_participants(
    values: Iterable[ParticipantCER],
) -> CERDistribution:
    """Summarize the distribution of participant-level pooled CER values."""

    items = list(values)

    if not items:
        raise ValueError("At least one participant result is required.")

    return summarize_cer(item.cer for item in items)
