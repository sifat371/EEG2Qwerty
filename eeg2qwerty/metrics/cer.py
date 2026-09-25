from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import median
from typing import Iterable


def levenshtein_distance(reference: str, prediction: str) -> int:
    """Compute character-level Levenshtein edit distance."""

    if reference == prediction:
        return 0

    if not reference:
        return len(prediction)

    if not prediction:
        return len(reference)

    # Keep the rolling row aligned to the shorter second dimension.
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference

    previous = list(range(len(prediction) + 1))

    for i, reference_char in enumerate(reference, start=1):
        current = [i]

        for j, prediction_char in enumerate(prediction, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (
                reference_char != prediction_char
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


def normalized_cer(
    reference: str,
    prediction: str,
) -> float:
    """
    Return normalized character error rate for one sequence.

    The denominator is the reference length. Empty references are rejected
    because their normalized CER is undefined.
    """

    if len(reference) == 0:
        raise ValueError(
            "Cannot compute normalized CER for an empty reference."
        )

    return (
        levenshtein_distance(
            reference,
            prediction,
        )
        / len(reference)
    )


@dataclass(frozen=True)
class CERDistribution:
    count: int
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float


def summarize_cer(
    values: Iterable[float],
) -> CERDistribution:
    """Summarize a collection of sentence- or participant-level CERs."""

    samples = [float(value) for value in values]

    if not samples:
        raise ValueError("At least one CER value is required.")

    mean_value = sum(samples) / len(samples)

    variance = (
        sum(
            (value - mean_value) ** 2
            for value in samples
        )
        / len(samples)
    )

    return CERDistribution(
        count=len(samples),
        mean=mean_value,
        median=float(median(samples)),
        standard_deviation=sqrt(variance),
        minimum=min(samples),
        maximum=max(samples),
    )
