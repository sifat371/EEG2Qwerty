from __future__ import annotations

from collections import OrderedDict

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from x_transformers import Encoder

from brain2qwerty_v1.metrics import CER


NUM_CLASSES = 29


class PositionAwareChannelMerger(nn.Module):
    """
    Lightweight version of Brain2Qwerty's position-aware channel merger.

    Electrode coordinates are Fourier encoded and converted into
    attention weights over the real EEG channels. A low-rank subject
    adapter permits participant-specific spatial mappings.
    """

    def __init__(
        self,
        n_channels: int,
        n_virtual_channels: int = 48,
        n_subjects: int = 64,
        n_frequencies: int = 6,
        position_hidden: int = 128,
        subject_dimension: int = 48,
        subject_rank: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_virtual_channels = n_virtual_channels
        self.n_frequencies = n_frequencies
        self.subject_rank = subject_rank

        frequencies = (
            2.0
            ** torch.arange(
                n_frequencies,
                dtype=torch.float32,
            )
        ) * torch.pi

        self.register_buffer(
            "frequencies",
            frequencies,
            persistent=False,
        )

        # Two coordinates × sine/cosine × n_frequencies.
        position_dimension = 2 * 2 * n_frequencies

        self.position_network = nn.Sequential(
            nn.Linear(
                position_dimension,
                position_hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                position_hidden,
                n_virtual_channels,
            ),
        )

        # Low-rank participant-specific correction to the spatial map.
        self.subject_embedding = nn.Embedding(
            n_subjects,
            subject_dimension,
        )

        self.subject_to_virtual = nn.Linear(
            subject_dimension,
            n_virtual_channels * subject_rank,
        )

        self.channel_factors = nn.Parameter(
            torch.randn(
                subject_rank,
                n_channels,
            )
            * 0.02
        )

        self.log_temperature = nn.Parameter(
            torch.zeros(n_virtual_channels)
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
                "Expected channel positions with shape "
                f"[N, C, D] or [C, D], received {positions.shape}"
            )

        if positions.shape[0] == 1 and batch_size > 1:
            positions = positions.expand(
                batch_size,
                -1,
                -1,
            )

        # Handle an unexpected [N, D, C] layout.
        if (
            positions.shape[1] != self.n_channels
            and positions.shape[2] == self.n_channels
        ):
            positions = positions.transpose(1, 2)

        if positions.shape[1] != self.n_channels:
            raise ValueError(
                "Channel-position count does not match EEG: "
                f"{positions.shape[1]} versus {self.n_channels}"
            )

        if positions.shape[-1] < 2:
            raise ValueError(
                "At least two electrode coordinates are required."
            )

        positions = positions[..., :2]
        positions = torch.nan_to_num(positions)

        # Translation- and approximately scale-invariant coordinates.
        positions = positions - positions.mean(
            dim=1,
            keepdim=True,
        )

        scale = positions.square().mean(
            dim=(1, 2),
            keepdim=True,
        ).sqrt().clamp_min(1e-6)

        return positions / scale

    def _fourier_encode(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        angles = (
            positions.unsqueeze(-1)
            * self.frequencies.view(1, 1, 1, -1)
        )

        encoded = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )

        return encoded.flatten(start_dim=-2)

    def forward(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor,
        channel_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = eeg.shape[0]

        positions = self._prepare_positions(
            channel_positions.to(
                device=eeg.device,
                dtype=eeg.dtype,
            ),
            batch_size=batch_size,
        )

        position_features = self._fourier_encode(
            positions
        )

        # [N, C, V] -> [N, V, C]
        spatial_logits = self.position_network(
            position_features
        ).transpose(1, 2)

        subject_features = self.subject_embedding(
            subject_ids.long().view(-1)
        )

        subject_virtual = self.subject_to_virtual(
            subject_features
        ).view(
            batch_size,
            self.n_virtual_channels,
            self.subject_rank,
        )

        subject_bias = torch.einsum(
            "nvr,rc->nvc",
            subject_virtual,
            self.channel_factors,
        )

        temperature = (
            F.softplus(self.log_temperature)
            + 0.5
        ).view(1, -1, 1)

        spatial_weights = torch.softmax(
            (
                spatial_logits
                + subject_bias
            ).float()
            / temperature.float(),
            dim=-1,
        ).to(dtype=eeg.dtype)

        return torch.einsum(
            "nvc,nct->nvt",
            spatial_weights,
            eeg,
        )


class MultiScaleTemporalStem(nn.Module):
    """Parallel short-, medium- and long-kernel temporal filters."""

    def __init__(
        self,
        in_channels: int,
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if width % 3 != 0:
            raise ValueError(
                "width must be divisible by three."
            )

        branch_width = width // 3

        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    branch_width,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in (3, 5, 9)
            ]
        )

        self.output = nn.Sequential(
            nn.GroupNorm(
                num_groups=1,
                num_channels=width,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                branch(features)
                for branch in self.branches
            ],
            dim=1,
        )

        return self.output(features)


class SubjectFiLM(nn.Module):
    """Feature-wise subject adaptation initialized as identity."""

    def __init__(
        self,
        width: int,
        n_subjects: int,
        subject_dimension: int = 64,
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

        gamma = torch.tanh(gamma).unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        return features * (1.0 + gamma) + beta


class GatedDilatedResidualBlock(nn.Module):
    """
    Larger Brain2Qwerty-inspired residual temporal block.

    Depthwise dilation captures temporal context efficiently. A GEGLU
    pointwise network increases capacity without using dense temporal
    convolutions at every layer.
    """

    def __init__(
        self,
        width: int,
        dilation: int,
        expansion: int = 2,
        dropout: float = 0.20,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()

        inner_width = width * expansion

        self.residual_scale = residual_scale

        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=width,
        )

        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=width,
            bias=False,
        )

        self.input_projection = nn.Conv1d(
            width,
            inner_width * 2,
            kernel_size=1,
        )

        self.output_projection = nn.Conv1d(
            inner_width,
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

        value, gate = self.input_projection(
            features
        ).chunk(2, dim=1)

        features = value * F.gelu(gate)
        features = self.dropout(features)
        features = self.output_projection(features)
        features = self.dropout(features)

        return (
            residual
            + self.residual_scale * features
        )


class TemporalAttentionPool(nn.Module):
    def __init__(
        self,
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
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

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        # [N, C, T] -> [N, T, C]
        temporal_features = features.transpose(
            1,
            2,
        ).contiguous()

        scores = self.network(
            temporal_features
        ).squeeze(-1)

        weights = torch.softmax(
            scores.float(),
            dim=-1,
        ).to(dtype=temporal_features.dtype)

        pooled = torch.sum(
            temporal_features
            * weights.unsqueeze(-1),
            dim=1,
        )

        return self.output_norm(pooled)


class GeoFusionEEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 64,
        n_virtual_channels: int = 48,
        width: int = 384,
        depth: int = 8,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.channel_merger = PositionAwareChannelMerger(
            n_channels=n_channels,
            n_virtual_channels=n_virtual_channels,
            n_subjects=n_subjects,
            dropout=dropout,
        )

        self.temporal_stem = MultiScaleTemporalStem(
            in_channels=n_virtual_channels,
            width=width,
            dropout=dropout,
        )

        self.subject_film = SubjectFiLM(
            width=width,
            n_subjects=n_subjects,
        )

        dilation_pattern = (1, 2, 4)

        self.blocks = nn.ModuleList(
            [
                GatedDilatedResidualBlock(
                    width=width,
                    dilation=dilation_pattern[
                        index % len(dilation_pattern)
                    ],
                    expansion=2,
                    dropout=dropout,
                    residual_scale=0.1,
                )
                for index in range(depth)
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
                "Expected EEG shape [N, C, T], "
                f"received {tuple(eeg.shape)}"
            )

        features = self.channel_merger(
            eeg.float(),
            subject_ids,
            channel_positions,
        )

        features = self.temporal_stem(features)

        features = self.subject_film(
            features,
            subject_ids,
        )

        for block in self.blocks:
            features = block(features)

        features = self.final_norm(features)

        return self.pool(features)


class LiteQwertyGeoFusionModule(pl.LightningModule):
    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 64,
        n_virtual_channels: int = 48,
        width: int = 384,
        encoder_depth: int = 8,
        transformer_depth: int = 4,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.25,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-2,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.eeg_encoder = GeoFusionEEGEncoder(
            n_channels=n_channels,
            n_subjects=n_subjects,
            n_virtual_channels=n_virtual_channels,
            width=width,
            depth=encoder_depth,
            dropout=0.20,
        )

        # x-transformers provides the same ALiBi mechanism used by the
        # released Brain2Qwerty sentence encoder.
        self.sequence_encoder = Encoder(
            dim=width,
            depth=transformer_depth,
            heads=transformer_heads,
            alibi_pos_bias=True,
            attn_dropout=transformer_dropout,
            ff_dropout=transformer_dropout,
            ff_mult=4,
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
            uid = segment.trigger.extra[
                "sentence_UID"
            ]

            groups.setdefault(
                uid,
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

        positions = batch.data[
            "channel_positions"
        ]

        labels = batch.data[
            "feature"
        ].long().view(-1)

        embeddings = self.eeg_encoder(
            eeg,
            subject_ids,
            positions,
        )

        groups = self._sentence_groups(batch)

        sentence_embeddings = [
            embeddings[indexes]
            for indexes in groups
        ]

        sentence_labels = [
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

        valid_mask = (
            torch.arange(
                maximum_length,
                device=embeddings.device,
            ).unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        contextual = self.sequence_encoder(
            padded,
            mask=valid_mask,
        )

        logits = self.classifier(
            contextual[valid_mask]
        )

        targets = torch.cat(
            sentence_labels,
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
            self.val_cer.update(logits, labels)

            self.log(
                "val_CER",
                self.val_cer,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=labels.numel(),
            )

        elif stage == "test":
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