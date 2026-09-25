from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import studies  # noqa: F401
from studies.spanishbcbl import Pinet2024Eeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate aligned event tables for all EEG timelines."
    )
    parser.add_argument("--data-root", type=Path, required=True)
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
    derived_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root / "derived"
    )
    timeline_root = derived_root / "all_timelines"

    derived_root.mkdir(parents=True, exist_ok=True)
    timeline_root.mkdir(parents=True, exist_ok=True)

    combined_output = derived_root / "pinet2024_eeg_all_events.pkl.gz"
    summary_output = derived_root / "pinet2024_eeg_summary.csv"
    failure_output = derived_root / "alignment_failures.json"

    loader = Pinet2024Eeg(path=data_root)
    timelines = list(loader.iter_timelines())

    print("Full EEG alignment")
    print("------------------")
    print(f"Timelines found: {len(timelines)}")

    all_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, timeline in enumerate(timelines, start=1):
        subject = str(timeline["subject"])
        session = int(timeline["session"])
        task = str(timeline["task"])

        timeline_id = f"{subject}_S{session}_{task}"
        output_path = timeline_root / f"{timeline_id}.pkl"

        print(f"\n[{index:02d}/{len(timelines):02d}] {timeline_id}")

        try:
            if output_path.exists():
                print(f"Loading cached alignment: {output_path}")
                events = pd.read_pickle(output_path)
            else:
                events = loader._load_timeline_events(timeline)

                if not isinstance(events, pd.DataFrame):
                    raise TypeError(
                        "Expected pandas DataFrame, received "
                        f"{type(events)}"
                    )

                required_columns = {
                    "type",
                    "start",
                    "duration",
                    "frequency",
                    "filepath",
                }
                missing_columns = required_columns - set(events.columns)
                if missing_columns:
                    raise RuntimeError(
                        "Missing required columns: "
                        f"{sorted(missing_columns)}"
                    )

                recording_mask = events["type"].eq("Eeg")
                timed_mask = events["type"].isin(
                    ["Keystroke", "Word", "Sentence"]
                )

                if recording_mask.sum() != 1:
                    raise RuntimeError(
                        "Expected exactly one Eeg row, found "
                        f"{int(recording_mask.sum())}."
                    )
                if timed_mask.sum() == 0:
                    raise RuntimeError(
                        "No timed linguistic events were found."
                    )
                if events.loc[timed_mask, "start"].isna().any():
                    raise RuntimeError(
                        "NaN start times found in timed events."
                    )
                if events.loc[timed_mask, "duration"].isna().any():
                    raise RuntimeError(
                        "NaN durations found in timed events."
                    )
                if events.loc[timed_mask, "duration"].le(0).any():
                    raise RuntimeError(
                        "Zero or negative timed-event durations found."
                    )

                events = events.copy()
                events["subject"] = subject
                events["session"] = session
                events["task"] = task
                events["timeline_id"] = timeline_id

                events.to_pickle(output_path)
                print(f"Saved: {output_path}")

            counts = events["type"].value_counts()

            summary_rows.append(
                {
                    "timeline_id": timeline_id,
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "rows": len(events),
                    "eeg": int(counts.get("Eeg", 0)),
                    "keystrokes": int(counts.get("Keystroke", 0)),
                    "words": int(counts.get("Word", 0)),
                    "sentences": int(counts.get("Sentence", 0)),
                }
            )
            all_frames.append(events)

            print(
                f"Rows={len(events):,}, "
                f"Keystrokes={int(counts.get('Keystroke', 0)):,}, "
                f"Words={int(counts.get('Word', 0)):,}, "
                f"Sentences={int(counts.get('Sentence', 0)):,}"
            )

        except Exception as exc:
            failures.append(
                {
                    "timeline": timeline,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            failure_output.write_text(
                json.dumps(failures, indent=2),
                encoding="utf-8",
            )
            print(f"FAILED: {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_output, index=False)

    print("\nAlignment summary")
    print("-----------------")
    print(f"Successful timelines: {len(all_frames)}")
    print(f"Failed timelines: {len(failures)}")
    print(f"Summary saved: {summary_output}")

    if failures:
        print(f"Failure report: {failure_output}")
        raise RuntimeError(
            f"{len(failures)} timeline alignment(s) failed. "
            "Inspect the failure report before continuing."
        )

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    combined.to_pickle(combined_output, compression="gzip")

    print("\nCombined event table")
    print("--------------------")
    print(f"Rows: {len(combined):,}")
    print(f"Subjects: {combined['subject'].nunique()}")
    print(f"Timelines: {combined['timeline_id'].nunique()}")
    print("\nEvent counts:")
    print(combined["type"].value_counts())
    print(f"\nCombined file: {combined_output}")
    print("\nSUCCESS: All EEG timelines aligned.")


if __name__ == "__main__":
    main()
