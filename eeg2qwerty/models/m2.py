from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence


NUM_CLASSES = 29


class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        dilation: int,
        dropout: float,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        expanded_width = width * expansion

        self.norm = nn.GroupNorm(1, width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=width,
            bias=False,
        )
        self.pointwise_in = nn.Conv1d(width, expanded_width, kernel_size=1)
        self.pointwise_out = nn.Conv1d(expanded_width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
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
    """Standardized implementation of the historical M2 EEG encoder."""

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

        self.temporal_filter = nn.Sequential(
            nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=5,
                padding=2,
                groups=n_channels,
                bias=False,
            ),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
        )

        self.spatial_projection = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=1, bias=False),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.subject_embedding = nn.Embedding(n_subjects, subject_dimension)
        self.film_projection = nn.Linear(subject_dimension, width * 2)
        nn.init.zeros_(self.film_projection.weight)
        nn.init.zeros_(self.film_projection.bias)

        dilation_pattern = (1, 2, 4)
        self.residual_blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    width=width,
                    dilation=dilation_pattern[index % len(dilation_pattern)],
                    dropout=dropout,
                )
                for index in range(blocks)
            ]
        )

        self.final_norm = nn.GroupNorm(1, width)
        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1, bias=False),
        )
        self.output_norm = nn.LayerNorm(width)

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

        features = self.temporal_filter(eeg.float())
        features = self.spatial_projection(features)

        subject_features = self.subject_embedding(subject_ids.long().view(-1))
        gamma, beta = self.film_projection(subject_features).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        features = features * (1.0 + gamma) + beta

        for block in self.residual_blocks:
            features = block(features)

        temporal_features = self.final_norm(features).transpose(1, 2).contiguous()
        attention_scores = self.temporal_attention(temporal_features).squeeze(-1)
        attention_weights = torch.softmax(
            attention_scores.float(),
            dim=-1,
        ).to(dtype=temporal_features.dtype)

        pooled = torch.sum(
            temporal_features * attention_weights.unsqueeze(-1),
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
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding.to(dtype=dtype)


@dataclass(frozen=True)
class SentenceBatchOutput:
    logits: torch.Tensor
    padding_mask: torch.Tensor


class StandardizedM2(nn.Module):
    """
    Public standardized implementation of the completed historical M2 family.

    Sentence boundaries are supplied explicitly as groups of keystroke indices
    instead of inferred from arbitrary fixed-size loader batches.
    """

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
    ) -> None:
        super().__init__()

        if width % transformer_heads != 0:
            raise ValueError("width must be divisible by transformer_heads")

        self.encoder = BalancedEEGEncoder(
            n_channels=n_channels,
            n_subjects=n_subjects,
            width=width,
            blocks=encoder_blocks,
            dropout=encoder_dropout,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(width, NUM_CLASSES)

    def forward(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor,
        sentence_groups: list[list[int]],
    ) -> SentenceBatchOutput:
        embeddings = self.encoder(eeg, subject_ids)

        sentence_embeddings = [
            embeddings[indexes]
            for indexes in sentence_groups
        ]
        lengths = torch.tensor(
            [sentence.shape[0] for sentence in sentence_embeddings],
            device=embeddings.device,
            dtype=torch.long,
        )

        padded = pad_sequence(sentence_embeddings, batch_first=True)
        maximum_length = padded.shape[1]
        padding_mask = (
            torch.arange(maximum_length, device=embeddings.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )

        positions = sinusoidal_position_encoding(
            maximum_length,
            padded.shape[-1],
            padded.device,
            padded.dtype,
        )
        contextual = self.sequence_encoder(
            padded + positions.unsqueeze(0),
            src_key_padding_mask=padding_mask,
        )
        return SentenceBatchOutput(
            logits=self.classifier(contextual),
            padding_mask=padding_mask,
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
