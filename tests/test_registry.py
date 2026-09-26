import csv
import json

import pytest

from eeg2qwerty.registry import (
    promote_registry_row,
    validate_protocol_audit,
)


def valid_audit():
    split = {
        "sentences": 2,
        "keystrokes": 10,
        "reconstructed_typed_matches_sentence_typed": 1.0,
    }
    loader = {
        "unique_sentences": 2,
        "sentence_instances": 2,
        "sentences_split_across_batches": 0,
    }
    return {
        "splits": {
            "train": dict(split),
            "val": dict(split),
            "test": dict(split),
        },
        "whole_sentence_loader_audit": {
            "train": dict(loader),
            "val": dict(loader),
            "test": dict(loader),
        },
    }


def test_audit_rejects_split_sentences():
    payload = valid_audit()
    payload["whole_sentence_loader_audit"]["test"][
        "sentences_split_across_batches"
    ] = 1

    with pytest.raises(ValueError):
        validate_protocol_audit(payload)


def test_candidate_promotion(tmp_path):
    registry = tmp_path / "registry.csv"
    candidate = tmp_path / "candidate.csv"
    audit = tmp_path / "audit.json"

    fields = [
        "experiment",
        "commit_sha",
        "upstream_commit_sha",
        "config",
        "protocol",
        "target",
        "max_typographical_errors",
        "evaluator",
        "seed",
        "parameters",
        "hardware",
        "peak_vram_gb",
        "train_time_seconds",
        "neural_cer",
        "participant_mean_cer",
        "participant_median_cer",
        "participant_sd_cer",
        "lm_assisted_cer",
        "status",
        "notes",
    ]

    with registry.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    row = {field: "" for field in fields}
    row.update(
        {
            "experiment": "m2",
            "commit_sha": "abc",
            "target": "typed_key",
            "seed": "33",
            "status": "candidate_reproduction",
        }
    )
    with candidate.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    audit.write_text(json.dumps(valid_audit()), encoding="utf-8")

    promoted = promote_registry_row(
        candidate_path=candidate,
        audit_path=audit,
        registry_path=registry,
    )

    assert promoted["status"] == "standardized"

    rows = list(csv.DictReader(registry.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "standardized"
