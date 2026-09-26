from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from eeg2qwerty.data.sentence_batching import sentence_key
from eeg2qwerty.data.target_alignment import align_typed_to_reference
from eeg2qwerty.data.targets import (
    IGNORE_INDEX,
    canonical_key,
    intended_targets_for_segments,
)
from eeg2qwerty.metrics import (
    PredictionRecord,
    participant_cer,
    pooled_cer,
    summarize_participants,
)
from eeg2qwerty.models import StandardizedM2


INDEX_TO_CHAR = {
    0: "s",
    1: "o",
    2: "t",
    3: "e",
    4: "n",
    5: "c",
    6: "i",
    7: "a",
    8: " ",
    9: "d",
    10: "l",
    11: "r",
    12: "b",
    13: "@",
    14: "z",
    15: "v",
    16: "f",
    17: "m",
    18: "u",
    19: "h",
    20: "p",
    21: "g",
    22: "q",
    23: "w",
    24: "x",
    25: "y",
    26: "j",
    27: "k",
    28: "9",
}


def _extra(segment: Any) -> dict[str, Any]:
    trigger = getattr(segment, "trigger", None)
    extra = getattr(trigger, "extra", None)
    if isinstance(extra, dict):
        return extra
    try:
        return dict(extra)
    except Exception:
        return {}


def sentence_groups(segments: list[Any]) -> list[list[int]]:
    groups: OrderedDict[tuple[str, str, str, str], list[int]] = OrderedDict()
    for index, segment in enumerate(segments):
        key = sentence_key(segment, fallback_index=index)
        groups.setdefault(key, []).append(index)
    return list(groups.values())


class StandardizedM2Module(pl.LightningModule):
    """
    Lightning wrapper for the public standardized M2 baseline.

    val_CER/test_CER are participant-mean pooled CER values. Pooled CER across
    all reference characters is logged separately.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        target_mode: str = "typed_key",
        n_subjects: int = 64,
        width: int = 192,
        encoder_blocks: int = 6,
        transformer_layers: int = 2,
        transformer_heads: int = 2,
        transformer_ff: int = 512,
        encoder_dropout: float = 0.25,
        transformer_dropout: float = 0.30,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()

        if target_mode not in {"typed_key", "intended"}:
            raise ValueError("target_mode must be 'typed_key' or 'intended'")

        self.save_hyperparameters()
        self.target_mode = target_mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = StandardizedM2(
            n_channels=n_channels,
            n_subjects=n_subjects,
            width=width,
            encoder_blocks=encoder_blocks,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ff=transformer_ff,
            encoder_dropout=encoder_dropout,
            transformer_dropout=transformer_dropout,
        )
        self.loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._val_records: list[PredictionRecord] = []
        self._test_records: list[PredictionRecord] = []

    def _targets(self, batch) -> torch.Tensor:
        if self.target_mode == "typed_key":
            return batch.data["feature"].long().view(-1).to(self.device)

        return intended_targets_for_segments(
            list(batch.segments),
            device=self.device,
        )

    def _forward_batch(self, batch):
        groups = sentence_groups(list(batch.segments))
        output = self.model(
            batch.data["neuro"],
            batch.data["subject_id"].long().view(-1),
            groups,
        )
        return output, groups

    def _loss(
        self,
        output,
        targets: torch.Tensor,
        groups: list[list[int]],
    ) -> torch.Tensor:
        target_sequences = [targets[indexes] for indexes in groups]
        padded_targets = pad_sequence(
            target_sequences,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        if padded_targets.shape[:2] != output.logits.shape[:2]:
            raise RuntimeError("Target and model sentence shapes disagree.")

        valid_targets = padded_targets.ne(IGNORE_INDEX) & ~output.padding_mask
        if not bool(valid_targets.any()):
            raise RuntimeError("No valid training targets are present in this batch.")

        return self.loss_function(
            output.logits.reshape(-1, output.logits.shape[-1]),
            padded_targets.reshape(-1),
        )

    def _prediction_records(
        self,
        batch,
        output,
        groups: list[list[int]],
    ) -> list[PredictionRecord]:
        predicted_ids = output.logits.argmax(dim=-1)
        records: list[PredictionRecord] = []

        for sentence_index, indexes in enumerate(groups):
            extras = [_extra(batch.segments[index]) for index in indexes]
            typed = "".join(
                canonical_key(extra.get("button", ""))
                for extra in extras
            )

            if self.target_mode == "typed_key":
                reference = typed
            else:
                candidates = [
                    extra.get("true_sequence")
                    for extra in extras
                    if extra.get("true_sequence") is not None
                ]
                if not candidates:
                    continue
                reference = align_typed_to_reference("", str(candidates[0])).reference

            prediction = "".join(
                INDEX_TO_CHAR[int(class_id)]
                for class_id in predicted_ids[
                    sentence_index,
                    : len(indexes),
                ].detach().cpu().tolist()
            )

            records.append(
                PredictionRecord(
                    subject=str(extras[0].get("subject", "unknown_subject")),
                    reference=reference,
                    prediction=prediction,
                )
            )

        return records

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        output, groups = self._forward_batch(batch)
        targets = self._targets(batch)
        loss = self._loss(output, targets, groups)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch.segments),
        )

        if stage == "val":
            self._val_records.extend(
                self._prediction_records(batch, output, groups)
            )
        elif stage == "test":
            self._test_records.extend(
                self._prediction_records(batch, output, groups)
            )

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self._val_records = []

    def on_test_epoch_start(self) -> None:
        self._test_records = []

    def _log_cer_summary(
        self,
        stage: str,
        records: list[PredictionRecord],
    ) -> None:
        by_subject = participant_cer(records)
        distribution = summarize_participants(by_subject)

        self.log(
            f"{stage}_CER",
            distribution.mean,
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            f"{stage}_pooled_CER",
            pooled_cer(records),
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            f"{stage}_median_CER",
            distribution.median,
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            f"{stage}_CER_std",
            distribution.standard_deviation,
            prog_bar=False,
            sync_dist=False,
        )

    def on_validation_epoch_end(self) -> None:
        if self._val_records:
            self._log_cer_summary("val", self._val_records)

    def _write_prediction_csv(
        self,
        records: list[PredictionRecord],
        filename: str,
    ) -> None:
        output = Path(self.trainer.default_root_dir) / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["subject", "reference", "prediction"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "subject": record.subject,
                        "reference": record.reference,
                        "prediction": record.prediction,
                    }
                )

    def on_test_epoch_end(self) -> None:
        if self._test_records:
            self._log_cer_summary("test", self._test_records)
            self._write_prediction_csv(
                self._test_records,
                "test_predictions.csv",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.10,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
