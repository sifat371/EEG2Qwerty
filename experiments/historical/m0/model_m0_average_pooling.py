from __future__ import annotations

from collections import OrderedDict

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from brain2qwerty_v1.metrics import CER


NUM_CLASSES = 29


class EEGNetEncoder(nn.Module):
    """Compact spatial-temporal encoder for one EEG keystroke window."""

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        spatial_filters = temporal_filters * depth_multiplier

        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, 5),
                padding=(0, 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(
                spatial_filters,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # Expected input: [keystrokes, EEG channels, time samples]
        if eeg.ndim != 3:
            raise ValueError(
                f"Expected EEG shape [N, C, T], received {tuple(eeg.shape)}"
            )

        eeg = eeg.float().unsqueeze(1)

        eeg = self.temporal(eeg)
        eeg = self.spatial(eeg)
        eeg = self.separable(eeg)
        eeg = self.pool(eeg)

        return self.projection(eeg)


def sinusoidal_position_encoding(
    length: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(
        length,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)

    frequencies = torch.exp(
        torch.arange(
            0,
            dimension,
            2,
            device=device,
            dtype=torch.float32,
        )
        * (-torch.log(torch.tensor(10000.0, device=device)) / dimension)
    )

    encoding = torch.zeros(
        length,
        dimension,
        device=device,
        dtype=torch.float32,
    )

    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)

    return encoding.to(dtype=dtype)


class LiteQwertyModule(pl.LightningModule):
    """EEGNet encoder with a lightweight sentence model."""

    def __init__(
        self,
        n_channels: int,
        sequence_model: str = "transformer",
        d_model: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ff: int = 768,
        gru_layers: int = 2,
        n_subjects: int = 64,
        dropout: float = 0.20,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()

        if sequence_model not in {"transformer", "bigru"}:
            raise ValueError(
                "sequence_model must be 'transformer' or 'bigru'"
            )

        self.save_hyperparameters()

        self.sequence_model_name = sequence_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.eeg_encoder = EEGNetEncoder(
            n_channels=n_channels,
            d_model=d_model,
            dropout=dropout,
        )

        self.subject_embedding = nn.Embedding(
            n_subjects,
            d_model,
        )

        if sequence_model == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.sequence_encoder = nn.TransformerEncoder(
                layer,
                num_layers=transformer_layers,
                norm=nn.LayerNorm(d_model),
            )
        else:
            self.sequence_encoder = nn.GRU(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=gru_layers,
                dropout=dropout if gru_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )

        self.classifier = nn.Linear(
            d_model,
            NUM_CLASSES,
        )

        self.loss_function = nn.CrossEntropyLoss()

        self.val_cer = CER()
        self.test_cer = CER()

    @staticmethod
    def _sentence_groups(batch) -> list[list[int]]:
        groups: OrderedDict[object, list[int]] = OrderedDict()

        for index, segment in enumerate(batch.segments):
            sentence_uid = segment.trigger.extra["sentence_UID"]
            groups.setdefault(sentence_uid, []).append(index)

        return list(groups.values())

    def _encode_sentences(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eeg = batch.data["neuro"]
        subject_ids = batch.data["subject_id"].long().view(-1)
        labels = batch.data["feature"].long().view(-1)

        embeddings = self.eeg_encoder(eeg)
        embeddings = embeddings + self.subject_embedding(subject_ids)

        groups = self._sentence_groups(batch)

        sequences = [
            embeddings[indexes]
            for indexes in groups
        ]

        targets = [
            labels[indexes]
            for indexes in groups
        ]

        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences],
            device=embeddings.device,
            dtype=torch.long,
        )

        padded = pad_sequence(
            sequences,
            batch_first=True,
        )

        max_length = padded.shape[1]

        padding_mask = (
            torch.arange(
                max_length,
                device=embeddings.device,
            ).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )

        positions = sinusoidal_position_encoding(
            length=max_length,
            dimension=padded.shape[-1],
            device=padded.device,
            dtype=padded.dtype,
        )

        padded = padded + positions.unsqueeze(0)

        if self.sequence_model_name == "transformer":
            contextual = self.sequence_encoder(
                padded,
                src_key_padding_mask=padding_mask,
            )
        else:
            packed = pack_padded_sequence(
                padded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            packed_output, _ = self.sequence_encoder(packed)

            contextual, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=max_length,
            )

        logits = self.classifier(
            contextual[~padding_mask]
        )

        grouped_targets = torch.cat(targets)

        return logits, grouped_targets

    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:
        logits, labels = self._encode_sentences(batch)

        loss = self.loss_function(
            logits,
            labels,
        )

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.numel(),
        )

        if stage == "val":
            self.val_cer.update(logits, labels)
            self.log(
                "val_CER",
                self.val_cer,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=labels.numel(),
            )

        if stage == "test":
            self.test_cer.update(logits, labels)
            self.log(
                "test_CER",
                self.test_cer,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=labels.numel(),
            )

        return loss

    def training_step(
        self,
        batch,
        batch_index: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch,
        batch_index: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(
        self,
        batch,
        batch_index: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
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
