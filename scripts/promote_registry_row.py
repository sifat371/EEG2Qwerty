from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg2qwerty.registry import promote_registry_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote an audited candidate run into the curated registry."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("results/experiment_registry.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = promote_registry_row(
        candidate_path=args.candidate,
        audit_path=args.audit,
        registry_path=args.registry,
    )
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
