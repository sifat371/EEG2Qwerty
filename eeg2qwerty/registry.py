from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_SPLITS = ("train", "val", "test")


def validate_protocol_audit(
    payload: dict[str, Any],
    *,
    minimum_typed_match: float = 0.99,
) -> None:
    """Raise ValueError unless a protocol audit is safe to promote."""

    split_payload = payload.get("splits")
    if not isinstance(split_payload, dict):
        raise ValueError("Audit does not contain split summaries.")

    for split in REQUIRED_SPLITS:
        summary = split_payload.get(split)
        if not isinstance(summary, dict):
            raise ValueError(f"Audit is missing the {split!r} split.")

        sentences = int(summary.get("sentences", 0))
        keystrokes = int(summary.get("keystrokes", 0))
        if sentences <= 0 or keystrokes <= 0:
            raise ValueError(f"Audit split {split!r} is empty.")

        match = summary.get("reconstructed_typed_matches_sentence_typed")
        if match is not None and float(match) < minimum_typed_match:
            raise ValueError(
                f"Typed reconstruction match for {split!r} is {match}, "
                f"below required {minimum_typed_match}."
            )

    loader_audit = payload.get("whole_sentence_loader_audit")
    if not isinstance(loader_audit, dict):
        raise ValueError(
            "Audit does not contain whole_sentence_loader_audit. "
            "Re-run with --check-loaders."
        )

    for split in REQUIRED_SPLITS:
        summary = loader_audit.get(split)
        if not isinstance(summary, dict):
            raise ValueError(
                f"Loader audit is missing the {split!r} split."
            )
        split_count = int(summary.get("sentences_split_across_batches", -1))
        if split_count != 0:
            raise ValueError(
                f"{split!r} has {split_count} sentences split across batches."
            )


def promote_registry_row(
    *,
    candidate_path: Path,
    audit_path: Path,
    registry_path: Path,
) -> dict[str, str]:
    """
    Validate one candidate run and append it to the curated standardized registry.
    """

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_protocol_audit(audit)

    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 1:
        raise ValueError("Candidate registry file must contain exactly one row.")

    candidate = dict(rows[0])
    if candidate.get("status") != "candidate_reproduction":
        raise ValueError(
            "Only candidate_reproduction rows may be promoted."
        )

    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        existing = list(reader)

    if not fieldnames:
        raise ValueError("Curated registry has no header.")

    missing_fields = [field for field in fieldnames if field not in candidate]
    if missing_fields:
        raise ValueError(
            f"Candidate row is missing registry fields: {missing_fields}"
        )

    duplicate_key = (
        candidate.get("experiment"),
        candidate.get("commit_sha"),
        candidate.get("target"),
        candidate.get("seed"),
    )
    for row in existing:
        key = (
            row.get("experiment"),
            row.get("commit_sha"),
            row.get("target"),
            row.get("seed"),
        )
        if key == duplicate_key:
            raise ValueError(
                "A matching standardized result already exists in the registry."
            )

    candidate["status"] = "standardized"
    prior_notes = candidate.get("notes", "").strip()
    promotion_note = f"Promoted after protocol audit: {audit_path.name}"
    candidate["notes"] = (
        f"{prior_notes}; {promotion_note}"
        if prior_notes
        else promotion_note
    )

    with registry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow({field: candidate.get(field, "") for field in fieldnames})

    return candidate
