from __future__ import annotations

import math
from collections import OrderedDict

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from brain2qwerty_v1.metrics import CER


NUM_CLASSES = 29


class DilatedResidualBlock(nn.Module):
    """Depthwise-separable dilated temporal residual block."""

    def __init__(
        self,
        width: int,
        dilation: int,
        dropout: float,
        expansion: int = 2,
    ) -> None:
        super().__init__()

        expanded_width = width * expansion

        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=width,
        )

        self.depthwise = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=width,
            bias=False,
        )

        self.pointwise_in = nn.Conv1d(
            width,
            expanded_width,
            kernel_size=1,
        )

        self.pointwise_out = nn.Conv1d(
            expanded_width,
            width,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        residual = features

        features = self.norm(features)
        features = self.depthwise(features)
        features = self.pointwise_in(features)
        features = F.gelu(features)
        features = self.dropout(features)
        features = self.pointwise_out(features)
        features = self.dropout(features)

        return residual + features


class BalancedEEGEncoder(nn.Module):
    """
    Compact EEG encoder incorporating Brain2Qwerty-inspired components:

    1. Temporal filtering per EEG channel
    2. Learned spatial channel mixing
    3. Subject-conditioned FiLM
    4. Residual dilated temporal blocks
    5. Learned temporal attention pooling
    """

    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 64,
        subject_dimension: int = 32,
        width: int = 192,
        blocks: int = 6,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        if blocks < 1:
            raise ValueError("blocks must be at least 1")

        self.width = width

        self.temporal_filter = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=5,
                padding=2,
                groups=n_channels,
                bias=False,
            ),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
        )

        self.spatial_projection = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=width,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=width,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.subject_embedding = nn.Embedding(
            n_subjects,
            subject_dimension,
        )

        self.film_projection = nn.Linear(
            subject_dimension,
            width * 2,
        )

        nn.init.zeros_(self.film_projection.weight)
        nn.init.zeros_(self.film_projection.bias)

        dilation_pattern = [1, 2, 4]

        self.residual_blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    width=width,
                    dilation=dilation_pattern[
                        index % len(dilation_pattern)
                    ],
                    dropout=dropout,
                )
                for index in range(blocks)
            ]
        )

        self.final_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=width,
        )

        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(
                width // 2,
                1,
                bias=False,
            ),
        )

        self.output_norm = nn.LayerNorm(width)

    def apply_film(
        self,
        features: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        subject_features = self.subject_embedding(
            subject_ids
        )

        film_parameters = self.film_projection(
            subject_features
        )

        gamma, beta = film_parameters.chunk(
            2,
            dim=-1,
        )

        gamma = torch.tanh(gamma)

        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        return features * (1.0 + gamma) + beta

    def forward(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(
                "Expected EEG shape [N, channels, time], "
                f"received {tuple(eeg.shape)}"
            )

        subject_ids = subject_ids.long().view(-1)

        features = eeg.float()
        features = self.temporal_filter(features)
        features = self.spatial_projection(features)

        features = self.apply_film(
            features,
            subject_ids,
        )

        for block in self.residual_blocks:
            features = block(features)

        features = self.final_norm(features)

        temporal_features = features.transpose(
            1,
            2,
        ).contiguous()

        attention_scores = self.temporal_attention(
            temporal_features
        ).squeeze(-1)

        attention_weights = torch.softmax(
            attention_scores.float(),
            dim=-1,
        ).to(dtype=temporal_features.dtype)

        pooled = torch.sum(
            temporal_features
            * attention_weights.unsqueeze(-1),
            dim=1,
        )

        return self.output_norm(pooled)


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
        * (-math.log(10000.0) / dimension)
    )

    encoding = torch.zeros(
        length,
        dimension,
        device=device,
        dtype=torch.float32,
    )

    encoding[:, 0::2] = torch.sin(
        positions * frequencies
    )

    encoding[:, 1::2] = torch.cos(
        positions * frequencies
    )

    return encoding.to(dtype=dtype)


class LiteQwertyBalancedModule(pl.LightningModule):
    """Balanced EEG encoder with compact sentence Transformer."""

    def __init__(
        self,
        n_channels: int,
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
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if width % transformer_heads != 0:
            raise ValueError(
                "width must be divisible by transformer_heads"
            )

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.eeg_encoder = BalancedEEGEncoder(
            n_channels=n_channels,
            n_subjects=n_subjects,
            width=width,
            blocks=encoder_blocks,
            dropout=encoder_dropout,
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.sequence_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )

        self.classifier = nn.Linear(
            width,
            NUM_CLASSES,
        )

        self.loss_function = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

        self.val_cer = CER()
        self.test_cer = CER()

    @staticmethod
    def _sentence_groups(
        batch,
    ) -> list[list[int]]:
        groups: OrderedDict[
            object,
            list[int],
        ] = OrderedDict()

        for index, segment in enumerate(batch.segments):
            sentence_uid = segment.trigger.extra[
                "sentence_UID"
            ]

            groups.setdefault(
                sentence_uid,
                [],
            ).append(index)

        return list(groups.values())

    def _forward_batch(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eeg = batch.data["neuro"]

        subject_ids = batch.data[
            "subject_id"
        ].long().view(-1)

        labels = batch.data[
            "feature"
        ].long().view(-1)

        embeddings = self.eeg_encoder(
            eeg,
            subject_ids,
        )

        groups = self._sentence_groups(batch)

        sentence_embeddings = [
            embeddings[indexes]
            for indexes in groups
        ]

        sentence_targets = [
            labels[indexes]
            for indexes in groups
        ]

        lengths = torch.tensor(
            [
                sentence.shape[0]
                for sentence in sentence_embeddings
            ],
            device=embeddings.device,
            dtype=torch.long,
        )

        padded = pad_sequence(
            sentence_embeddings,
            batch_first=True,
        )

        maximum_length = padded.shape[1]

        padding_mask = (
            torch.arange(
                maximum_length,
                device=embeddings.device,
            ).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )

        positions = sinusoidal_position_encoding(
            length=maximum_length,
            dimension=padded.shape[-1],
            device=padded.device,
            dtype=padded.dtype,
        )

        contextual = self.sequence_encoder(
            padded + positions.unsqueeze(0),
            src_key_padding_mask=padding_mask,
        )

        logits = self.classifier(
            contextual[~padding_mask]
        )

        targets = torch.cat(
            sentence_targets,
            dim=0,
        )

        return logits, targets

    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:
        logits, labels = self._forward_batch(batch)

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
            self.val_cer.update(
                logits,
                labels,
            )

            self.log(
                "val_CER",
                self.val_cer,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=labels.numel(),
            )

        elif stage == "test":
            self.test_cer.update(
                logits,
                labels,
            )

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
        return self._shared_step(
            batch,
            "train",
        )

    def validation_step(
        self,
        batch,
        batch_index: int,
    ) -> torch.Tensor:
        return self._shared_step(
            batch,
            "val",
        )

    def test_step(
        self,
        batch,
        batch_index: int,
    ) -> torch.Tensor:
        return self._shared_step(
            batch,
            "test",
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
