from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable


@dataclass(frozen=True)
class AlignmentCounts:
    matches: int
    substitutions: int
    typed_insertions: int
    missing_reference: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.typed_insertions + self.missing_reference


@dataclass(frozen=True)
class TypedReferenceAlignment:
    typed: str
    reference: str
    target_by_typed_position: tuple[str | None, ...]
    counts: AlignmentCounts

    @property
    def aligned_positions(self) -> int:
        return sum(target is not None for target in self.target_by_typed_position)

    @property
    def coverage(self) -> float:
        if not self.target_by_typed_position:
            return 1.0
        return self.aligned_positions / len(self.target_by_typed_position)


def _canonical_characters(value: str | Iterable[str]) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = "".join(str(item) for item in value)

    return (
        text.replace("<space>", " ")
        .replace("<special>", "@")
        .replace("<number>", "9")
        .lower()
    )


def align_typed_to_reference(
    typed: str | Iterable[str],
    reference: str | Iterable[str],
    *,
    autojunk: bool = False,
) -> TypedReferenceAlignment:
    """Align a typed sequence to the reference with difflib.SequenceMatcher."""

    typed_text = _canonical_characters(typed)
    reference_text = _canonical_characters(reference)
    matcher = SequenceMatcher(a=typed_text, b=reference_text, autojunk=autojunk)

    targets: list[str | None] = [None] * len(typed_text)
    matches = 0
    substitutions = 0
    typed_insertions = 0
    missing_reference = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        typed_len = i2 - i1
        reference_len = j2 - j1

        if tag == "equal":
            for offset in range(typed_len):
                targets[i1 + offset] = reference_text[j1 + offset]
            matches += typed_len
        elif tag == "replace":
            paired = min(typed_len, reference_len)
            for offset in range(paired):
                targets[i1 + offset] = reference_text[j1 + offset]
            substitutions += paired
            typed_insertions += typed_len - paired
            missing_reference += reference_len - paired
        elif tag == "delete":
            typed_insertions += typed_len
        elif tag == "insert":
            missing_reference += reference_len
        else:
            raise RuntimeError(f"Unexpected SequenceMatcher opcode: {tag!r}")

    return TypedReferenceAlignment(
        typed=typed_text,
        reference=reference_text,
        target_by_typed_position=tuple(targets),
        counts=AlignmentCounts(
            matches=matches,
            substitutions=substitutions,
            typed_insertions=typed_insertions,
            missing_reference=missing_reference,
        ),
    )


def keep_under_typo_threshold(
    alignment: TypedReferenceAlignment,
    *,
    max_errors: int = 10,
) -> bool:
    """Keep sentences with at most max_errors typographical errors."""

    return alignment.counts.errors <= max_errors
