from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from eeg2qwerty.metrics import (
    PredictionRecord,
    participant_cer,
    pooled_cer,
    summarize_participants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize sentence predictions with pooled and participant-level CER."
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--subject-column", default="subject")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--prediction-column", default="prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records: list[PredictionRecord] = []

    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {
            args.subject_column,
            args.reference_column,
            args.prediction_column,
        }
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Prediction CSV is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            records.append(
                PredictionRecord(
                    subject=str(row[args.subject_column]),
                    reference=str(row[args.reference_column]),
                    prediction=str(row[args.prediction_column]),
                )
            )

    by_subject = participant_cer(records)
    distribution = summarize_participants(by_subject)

    payload = {
        "sentences": len(records),
        "pooled_cer": pooled_cer(records),
        "participant_distribution": asdict(distribution),
        "participants": [asdict(item) for item in by_subject],
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
