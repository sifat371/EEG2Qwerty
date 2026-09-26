from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg2qwerty.data.brain2qwerty_v1 import (
    build_upstream_eeg_events,
    build_upstream_eeg_loaders,
)
from eeg2qwerty.data.protocol import (
    audit_sentence_rows,
    sentence_group_columns,
)
from eeg2qwerty.data.sentence_batching import assert_complete_sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the standardized EEG2Qwerty sentence/target protocol."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-typographical-errors", type=int, default=10)
    parser.add_argument("--check-loaders", action="store_true")
    parser.add_argument("--train-batch-keystrokes", type=int, default=256)
    parser.add_argument("--eval-batch-keystrokes", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def _canonical_typed_metadata(value: object) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .replace("<space>", " ")
        .replace("<special>", "@")
        .replace("<number>", "9")
        .lower()
        .strip()
    )


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()

    _, events = build_upstream_eeg_events(
        data_root=data_root,
        cache_root=cache_root,
        debug=args.debug,
    )

    group_columns = sentence_group_columns(events)
    keystrokes = events[events["type"] == "Keystroke"].copy()

    payload: dict[str, object] = {
        "schema_version": 1,
        "target_alignment": "python difflib.SequenceMatcher",
        "max_typographical_errors": args.max_typographical_errors,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        split_rows = keystrokes[keystrokes["split"] == split].copy()
        sentence_count = 0
        kept_sentences = 0
        dropped_sentences = 0
        keystroke_count = 0
        aligned_positions = 0
        substitutions = 0
        typed_insertions = 0
        missing_reference = 0
        typed_metadata_present = 0
        typed_metadata_matches = 0
        examples: list[dict[str, object]] = []

        for key, group in split_rows.groupby(
            group_columns,
            dropna=False,
            sort=False,
        ):
            sentence_count += 1

            order_column = next(
                (
                    column
                    for column in ("start", "start_sample")
                    if column in group.columns
                ),
                None,
            )
            if order_column is not None:
                group = group.sort_values(order_column, kind="stable")

            audit = audit_sentence_rows(group)
            keystroke_count += len(group)
            aligned_positions += round(audit.aligned_coverage * len(group))
            substitutions += audit.substitutions
            typed_insertions += audit.typed_insertions
            missing_reference += audit.missing_reference

            if audit.errors <= args.max_typographical_errors:
                kept_sentences += 1
            else:
                dropped_sentences += 1

            if "sentence_typed" in group.columns:
                values = [
                    value
                    for value in group["sentence_typed"].tolist()
                    if value is not None and str(value) != "nan"
                ]
                if values:
                    typed_metadata_present += 1
                    metadata = _canonical_typed_metadata(values[0])
                    if audit.typed.strip() == metadata:
                        typed_metadata_matches += 1
                    elif len(examples) < 10:
                        examples.append(
                            {
                                "sentence_key": str(key),
                                "reconstructed_typed": audit.typed,
                                "sentence_typed": metadata,
                            }
                        )

        payload["splits"][split] = {
            "sentences": sentence_count,
            "keystrokes": keystroke_count,
            "sentences_kept_at_threshold": kept_sentences,
            "sentences_dropped_at_threshold": dropped_sentences,
            "aligned_position_coverage": (
                aligned_positions / keystroke_count
                if keystroke_count
                else None
            ),
            "substitutions": substitutions,
            "typed_insertions": typed_insertions,
            "missing_reference": missing_reference,
            "sentence_typed_metadata_coverage": (
                typed_metadata_present / sentence_count
                if sentence_count
                else None
            ),
            "reconstructed_typed_matches_sentence_typed": (
                typed_metadata_matches / typed_metadata_present
                if typed_metadata_present
                else None
            ),
            "typed_mismatch_examples": examples,
        }

    if args.check_loaders:
        loaders, _ = build_upstream_eeg_loaders(
            data_root=data_root,
            cache_root=cache_root,
            train_batch_keystrokes=args.train_batch_keystrokes,
            eval_batch_keystrokes=args.eval_batch_keystrokes,
            num_workers=args.num_workers,
            seed=33,
            debug=args.debug,
            max_typographical_errors=args.max_typographical_errors,
        )
        loader_audit = {}
        for split, loader in loaders.items():
            stats = assert_complete_sentences(loader)
            loader_audit[split] = stats
            if stats["sentences_split_across_batches"] != 0:
                raise RuntimeError(
                    f"{split} still splits sentences across loader batches."
                )
        payload["whole_sentence_loader_audit"] = loader_audit

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
