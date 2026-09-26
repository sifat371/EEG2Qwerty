from eeg2qwerty.data.target_alignment import (
    align_typed_to_reference,
    keep_under_typo_threshold,
)


def test_sequence_matcher_exact_alignment():
    aligned = align_typed_to_reference("hola", "hola")
    assert aligned.target_by_typed_position == tuple("hola")
    assert aligned.counts.matches == 4
    assert aligned.counts.errors == 0
    assert aligned.coverage == 1.0


def test_sequence_matcher_substitution():
    aligned = align_typed_to_reference("hila", "hola")
    assert aligned.target_by_typed_position == tuple("hola")
    assert aligned.counts.substitutions == 1
    assert aligned.counts.errors == 1


def test_extra_typed_character_is_explicitly_unmatched():
    aligned = align_typed_to_reference("holaa", "hola")
    assert aligned.target_by_typed_position[:-1] == tuple("hola")
    assert aligned.target_by_typed_position[-1] is None
    assert aligned.counts.typed_insertions == 1
    assert aligned.coverage == 4 / 5


def test_missing_reference_character_counts_as_error():
    aligned = align_typed_to_reference("hola", "holas")
    assert aligned.counts.missing_reference == 1
    assert aligned.counts.errors == 1


def test_paper_threshold_is_strictly_more_than_ten():
    ten = align_typed_to_reference("a" * 10, "b" * 10)
    eleven = align_typed_to_reference("a" * 11, "b" * 11)
    assert keep_under_typo_threshold(ten)
    assert not keep_under_typo_threshold(eleven)


def test_brain2qwerty_special_aliases_are_canonicalized():
    aligned = align_typed_to_reference(
        ["h", "<space>", "9", "<special>"],
        "h 9@",
    )
    assert aligned.typed == "h 9@"
