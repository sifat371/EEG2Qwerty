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


class ResidualGeometryGate(nn.Module):
    """
    Geometry-conditioned channel scaling.

    Unlike M3, this module never replaces or averages the raw EEG
    channels. It applies a bounded multiplicative correction while
    retaining every signed channel value.
    """

    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 64,
        n_frequencies: int = 6,
        position_hidden: int = 64,
        subject_dimension: int = 64,
        subject_rank: int = 16,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_frequencies = n_frequencies

        frequencies = (
            2.0
            ** torch.arange(
                n_frequencies,
                dtype=torch.float32,
            )
        ) * math.pi

        self.register_buffer(
            "frequencies",
            frequencies,
            persistent=False,
        )

        position_dimension = (
            2
            * 2
            * n_frequencies
        )

        self.position_network = nn.Sequential(
            nn.Linear(
                position_dimension,
                position_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                position_hidden,
                1,
            ),
        )

        self.subject_embedding = nn.Embedding(
            n_subjects,
            subject_dimension,
        )

        self.subject_to_rank = nn.Linear(
            subject_dimension,
            subject_rank,
        )

        self.channel_basis = nn.Parameter(
            torch.randn(
                subject_rank,
                n_channels,
            )
            * 0.02
        )

        # sigmoid(-2) ≈ 0.12, so geometry begins as a small correction.
        self.geometry_strength_logit = nn.Parameter(
            torch.tensor(-2.0)
        )

    def _prepare_positions(
        self,
        positions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)

        if positions.ndim != 3:
            raise ValueError(
                "Expected channel positions [N, C, D] or [C, D], "
                f"received {tuple(positions.shape)}"
            )

        if positions.shape[0] == 1 and batch_size > 1:
            positions = positions.expand(
                batch_size,
                -1,
                -1,
            )

        if (
            positions.shape[1] != self.n_channels
            and positions.shape[2] == self.n_channels
        ):
            positions = positions.transpose(1, 2)

        if positions.shape[1] != self.n_channels:
            raise ValueError(
                "Position-channel mismatch: "
                f"{positions.shape[1]} versus {self.n_channels}"
            )

        if positions.shape[-1] < 2:
            raise ValueError(
                "At least two electrode coordinates are required."
            )

        positions = positions[..., :2]
        positions = torch.nan_to_num(positions)

        positions = (
            positions
            - positions.mean(
                dim=1,
                keepdim=True,
            )
        )

        scale = (
            positions.square()
            .mean(
                dim=(1, 2),
                keepdim=True,
            )
            .sqrt()
            .clamp_min(1e-6)
        )

        return positions / scale

    def _fourier_encode(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        angles = (
            positions.unsqueeze(-1)
            * self.frequencies.view(
                1,
                1,
                1,
                -1,
            )
        )

        encoded = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )

        return encoded.flatten(
            start_dim=-2
        )

    def forward(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = eeg.shape[0]

        positions = self._prepare_positions(
            positions.to(
                device=eeg.device,
                dtype=eeg.dtype,
            ),
            batch_size=batch_size,
        )

        position_features = self._fourier_encode(
            positions
        )

        base_gate = self.position_network(
            position_features
        ).squeeze(-1)

        subject_factors = self.subject_to_rank(
            self.subject_embedding(
                subject_ids.long().view(-1)
            )
        )

        subject_gate = torch.einsum(
            "nr,rc->nc",
            subject_factors,
            self.channel_basis,
        )

        gate = torch.tanh(
            base_gate + subject_gate
        )

        strength = torch.sigmoid(
            self.geometry_strength_logit
        )

        return eeg * (
            1.0
            + strength
            * gate.unsqueeze(-1)
        )


class SubjectFiLM(nn.Module):
    """Subject-conditioned feature scaling and shifting."""

    def __init__(
        self,
        width: int,
        n_subjects: int = 64,
        subject_dimension: int = 32,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            n_subjects,
            subject_dimension,
        )

        self.projection = nn.Linear(
            subject_dimension,
            width * 2,
        )

        # Begin as the identity transformation.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        features: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        parameters = self.projection(
            self.embedding(
                subject_ids.long().view(-1)
            )
        )

        gamma, beta = parameters.chunk(
            2,
            dim=-1,
        )

        gamma = torch.tanh(
            gamma
        ).unsqueeze(-1)

        beta = beta.unsqueeze(-1)

        return (
            features
            * (1.0 + gamma)
            + beta
        )


class DilatedResidualBlock(nn.Module):
    """
    Depthwise-separable residual temporal block with LayerScale.

    LayerScale stabilizes the deeper width-320 encoder and begins each
    block as a small correction rather than a large transformation.
    """

    def __init__(
        self,
        width: int,
        dilation: int,
        expansion: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        expanded_width = (
            width
            * expansion
        )

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
            in_channels=width,
            out_channels=expanded_width,
            kernel_size=1,
        )

        self.pointwise_out = nn.Conv1d(
            in_channels=expanded_width,
            out_channels=width,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.layer_scale = nn.Parameter(
            torch.full(
                (1, width, 1),
                0.1,
            )
        )

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

        return (
            residual
            + self.layer_scale
            * features
        )


class TemporalAttentionPool(nn.Module):
    def __init__(
        self,
        width: int,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(
                width,
                width // 2,
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(
                width // 2,
                1,
                bias=False,
            ),
        )

        self.output_norm = nn.LayerNorm(
            width
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        temporal_features = features.transpose(
            1,
            2,
        ).contiguous()

        scores = self.attention(
            temporal_features
        ).squeeze(-1)

        weights = torch.softmax(
            scores.float(),
            dim=-1,
        ).to(
            dtype=temporal_features.dtype
        )

        pooled = torch.sum(
            temporal_features
            * weights.unsqueeze(-1),
            dim=1,
        )

        return self.output_norm(
            pooled
        )


class ResidualFusionEEGEncoder(nn.Module):
    """
    Medium-capacity EEG encoder.

    The raw M2-style path is always preserved. Electrode geometry
    contributes through a separately learned residual correction.
    """

    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 64,
        width: int = 320,
        blocks: int = 8,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.geometry_gate = ResidualGeometryGate(
            n_channels=n_channels,
            n_subjects=n_subjects,
        )

        # Main raw-EEG path, inherited from the successful M2 design.
        self.raw_temporal = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=5,
                padding=2,
                groups=n_channels,
                bias=False,
            ),
            nn.BatchNorm1d(
                n_channels
            ),
            nn.GELU(),
        )

        self.raw_spatial = nn.Sequential(
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

        # Geometry branch processes only the residual difference.
        self.geometry_temporal = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                padding=1,
                groups=n_channels,
                bias=False,
            ),
            nn.BatchNorm1d(
                n_channels
            ),
            nn.GELU(),
        )

        self.geometry_projection = nn.Conv1d(
            in_channels=n_channels,
            out_channels=width,
            kernel_size=1,
            bias=False,
        )

        nn.init.normal_(
            self.geometry_projection.weight,
            mean=0.0,
            std=0.01,
        )

        self.geometry_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=width,
        )

        # sigmoid(-2) ≈ 0.12.
        self.fusion_strength_logit = nn.Parameter(
            torch.tensor(-2.0)
        )

        self.subject_film = SubjectFiLM(
            width=width,
            n_subjects=n_subjects,
        )

        dilation_pattern = (
            1,
            2,
            4,
            8,
        )

        self.residual_blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    width=width,
                    dilation=dilation_pattern[
                        index
                        % len(dilation_pattern)
                    ],
                    expansion=2,
                    dropout=dropout,
                )
                for index in range(blocks)
            ]
        )

        self.final_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=width,
        )

        self.pool = TemporalAttentionPool(
            width=width,
            dropout=dropout,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor,
        channel_positions: torch.Tensor,
    ) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(
                "Expected EEG [N, C, T], "
                f"received {tuple(eeg.shape)}"
            )

        eeg = eeg.float()

        raw_features = self.raw_spatial(
            self.raw_temporal(eeg)
        )

        geometry_eeg = self.geometry_gate(
            eeg,
            subject_ids,
            channel_positions,
        )

        geometry_delta = (
            geometry_eeg
            - eeg
        )

        geometry_features = (
            self.geometry_projection(
                self.geometry_temporal(
                    geometry_delta
                )
            )
        )

        geometry_features = self.geometry_norm(
            geometry_features
        )

        fusion_strength = torch.sigmoid(
            self.fusion_strength_logit
        )

        features = (
            raw_features
            + fusion_strength
            * geometry_features
        )

        features = self.subject_film(
            features,
            subject_ids,
        )

        for block in self.residual_blocks:
            features = block(features)

        features = self.final_norm(
            features
        )

        return self.pool(features)


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
        * (
            -math.log(10000.0)
            / dimension
        )
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

    return encoding.to(
        dtype=dtype
    )


class LiteQwertyResidualFusionModule(
    pl.LightningModule
):
    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 64,
        width: int = 320,
        encoder_blocks: int = 8,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_ff: int = 1024,
        encoder_dropout: float = 0.30,
        transformer_dropout: float = 0.30,
        learning_rate: float = 2e-4,
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

        self.eeg_encoder = ResidualFusionEEGEncoder(
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

        for index, segment in enumerate(
            batch.segments
        ):
            sentence_uid = (
                segment.trigger.extra[
                    "sentence_UID"
                ]
            )

            groups.setdefault(
                sentence_uid,
                [],
            ).append(index)

        return list(
            groups.values()
        )

    def _forward_batch(
        self,
        batch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        eeg = batch.data["neuro"]

        subject_ids = batch.data[
            "subject_id"
        ].long().view(-1)

        channel_positions = batch.data[
            "channel_positions"
        ]

        labels = batch.data[
            "feature"
        ].long().view(-1)

        embeddings = self.eeg_encoder(
            eeg,
            subject_ids,
            channel_positions,
        )

        groups = self._sentence_groups(
            batch
        )

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
                sequence.shape[0]
                for sequence
                in sentence_embeddings
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
            padded
            + positions.unsqueeze(0),
            src_key_padding_mask=padding_mask,
        )

        logits = self.classifier(
            contextual[
                ~padding_mask
            ]
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
        logits, labels = self._forward_batch(
            batch
        )

        loss = self.loss_function(
            logits,
            labels,
        )

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(
                stage == "train"
            ),
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
            total_steps=(
                self.trainer
                .estimated_stepping_batches
            ),
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