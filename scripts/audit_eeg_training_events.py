from __future__ import annotations

import argparse
from pathlib import Path

import neuralset as ns
import pandas as pd
import studies  # noqa: F401; registers the EEG study
from neuralset.events import Study

from brain2qwerty_v1.transforms import (
    Brain2QwertyV1Splitter,
    SpanishBCBLPreprocessing,
)
from brain2qwerty_v1.utils import BUTTON_MAPPING, NUM_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Brain2Qwerty v1 EEG training events and split integrity."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <data-root>/derived.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root / "derived"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    output_events = output_root / "eeg_training_events_audit.pkl.gz"
    output_splits = output_root / "eeg_split_summary.csv"
    output_classes = output_root / "eeg_character_summary.csv"

    print("EEG training-event audit")
    print("========================")
    print(f"Dataset root: {data_root}")
    print(f"Cache root:   {cache_root}")

    study = Study(
        name="Pinet2024Eeg",
        path=data_root,
        infra={"folder": cache_root},
        infra_timelines={
            "folder": cache_root,
            "cluster": None,
        },
    )

    print("\n1. Loading standardized events...")
    events = study.run()

    if not isinstance(events, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, received {type(events)}"
        )

    print(f"Raw standardized rows: {len(events):,}")
    print(events["type"].value_counts(dropna=False))

    print("\n2. Applying Brain2Qwerty v1 preprocessing...")
    events = SpanishBCBLPreprocessing().run(events)

    print("\n3. Applying sentence-level split...")
    events = Brain2QwertyV1Splitter(seed=1).run(events)
    events = ns.events.standardize_events(events)

    required_columns = {
        "type",
        "start",
        "duration",
        "subject",
        "sentence_UID",
        "split",
    }
    missing = required_columns - set(events.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns after preprocessing: {sorted(missing)}"
        )

    buttons = events.loc[events["type"].eq("Keystroke")].copy()
    if buttons.empty:
        raise RuntimeError("No Keystroke events remain after preprocessing.")

    print("\n4. Core dataset checks")
    print("----------------------")
    print(f"All rows:         {len(events):,}")
    print(f"Keystrokes:       {len(buttons):,}")
    print(f"Subjects:         {buttons['subject'].nunique()}")
    print(f"Sentence UIDs:    {buttons['sentence_UID'].nunique()}")
    print(f"Expected classes: {NUM_CLASSES}")

    unknown_buttons = sorted(
        set(buttons["button"].dropna().unique()) - set(BUTTON_MAPPING)
    )
    missing_button_count = int(buttons["button"].isna().sum())

    print(f"Missing button labels: {missing_button_count}")
    print(f"Unknown button labels: {unknown_buttons}")

    if missing_button_count:
        raise RuntimeError(
            f"{missing_button_count} keystrokes have missing button labels."
        )

    if unknown_buttons:
        raise RuntimeError(
            "Keystrokes outside BUTTON_MAPPING were found: "
            f"{unknown_buttons}"
        )

    buttons["class_id"] = buttons["button"].map(BUTTON_MAPPING)

    observed_classes = sorted(
        buttons["class_id"].dropna().astype(int).unique()
    )
    print(f"Observed class IDs: {observed_classes}")
    print(f"Observed classes:   {len(observed_classes)}")

    uid_split_counts = buttons.groupby("sentence_UID")["split"].nunique()
    leaking_uids = uid_split_counts[uid_split_counts > 1]

    print(f"Sentence UID split leakage: {len(leaking_uids)}")
    if len(leaking_uids):
        raise RuntimeError(
            "Some sentence UIDs appear in multiple splits: "
            f"{leaking_uids.index[:10].tolist()}"
        )

    print("\n5. Split summary")
    print("----------------")
    aggregations = {
        "keystrokes": ("button", "size"),
        "subjects": ("subject", "nunique"),
        "sentence_uids": ("sentence_UID", "nunique"),
    }
    if "sentence" in buttons.columns:
        aggregations["sentence_texts"] = ("sentence", "nunique")

    split_summary = (
        buttons.groupby("split", dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    split_summary["keystroke_fraction"] = (
        split_summary["keystrokes"] / len(buttons)
    )

    print(split_summary.to_string(index=False))

    expected_splits = {"train", "val", "test"}
    actual_splits = set(split_summary["split"].dropna())
    if actual_splits != expected_splits:
        raise RuntimeError(
            f"Expected splits {expected_splits}, found {actual_splits}."
        )

    print("\n6. Character-class summary")
    print("--------------------------")
    class_summary = (
        buttons.groupby(["class_id", "button"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["class_id", "count"], ascending=[True, False])
    )
    print(class_summary.to_string(index=False))

    events.to_pickle(output_events, compression="gzip")
    split_summary.to_csv(output_splits, index=False)
    class_summary.to_csv(output_classes, index=False)

    print("\nSaved outputs")
    print("-------------")
    print(f"Events audit:      {output_events}")
    print(f"Split summary:     {output_splits}")
    print(f"Character summary: {output_classes}")
    print("\nSUCCESS: EEG preprocessing and split audit passed.")


if __name__ == "__main__":
    main()
