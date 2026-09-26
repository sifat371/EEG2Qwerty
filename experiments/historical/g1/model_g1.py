from __future__ import annotations

from collections import OrderedDict

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from brain2qwerty_v1.metrics import CER


NUM_CLASSES = 29


def build_knn_adjacency(
    positions: torch.Tensor,
    k: int = 4,
) -> torch.Tensor:
    """
    Construct a symmetric, normalized k-nearest-neighbour graph.

    Args:
        positions:
            Electrode coordinates with shape [B, C, 2] or [C, 2].
        k:
            Number of spatial neighbours before symmetrization.

    Returns:
        Normalized adjacency with shape [B, C, C].
    """
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)

    if positions.ndim != 3:
        raise ValueError(
            "Expected channel positions [B, C, D] or [C, D], "
            f"received {tuple(positions.shape)}"
        )

    if positions.shape[-1] < 2:
        raise ValueError(
            "At least two channel-position coordinates are required."
        )

    positions = positions[..., :2].float()
    positions = torch.nan_to_num(positions)

    batch_size, n_channels, _ = positions.shape

    if k < 1 or k >= n_channels:
        raise ValueError(
            f"k must be between 1 and {n_channels - 1}, received {k}"
        )

    # Remove translation and normalize coordinate scale.
    positions = positions - positions.mean(
        dim=1,
        keepdim=True,
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

    positions = positions / scale

    distances = torch.cdist(
        positions,
        positions,
        p=2,
    )

    identity_mask = torch.eye(
        n_channels,
        device=positions.device,
        dtype=torch.bool,
    ).unsqueeze(0)

    masked_distances = distances.masked_fill(
        identity_mask,
        float("inf"),
    )

    neighbour_distances, neighbour_indices = torch.topk(
        masked_distances,
        k=k,
        dim=-1,
        largest=False,
    )

    # Use the local k-neighbour radius as the Gaussian scale.
    gaussian_scale = (
        neighbour_distances[..., -1]
        .mean(
            dim=1,
            keepdim=True,
        )
        .clamp_min(1e-4)
    )

    edge_weights = torch.exp(
        -neighbour_distances.square()
        / (
            2.0
            * gaussian_scale.unsqueeze(-1).square()
        )
    )

    adjacency = torch.zeros_like(
        distances
    )

    adjacency.scatter_(
        dim=2,
        index=neighbour_indices,
        src=edge_weights,
    )

    # Symmetrize the directed kNN graph.
    adjacency = torch.maximum(
        adjacency,
        adjacency.transpose(1, 2),
    )

    identity = torch.eye(
        n_channels,
        device=positions.device,
        dtype=adjacency.dtype,
    ).unsqueeze(0)

    adjacency = adjacency + identity

    # Symmetric graph normalization: D^(-1/2) A D^(-1/2).
    degree = adjacency.sum(
        dim=-1
    ).clamp_min(1e-6)

    inverse_sqrt_degree = degree.rsqrt()

    adjacency = (
        inverse_sqrt_degree.unsqueeze(-1)
        * adjacency
        * inverse_sqrt_degree.unsqueeze(-2)
    )

    return adjacency


class SubjectFiLM(nn.Module):
    """Participant-specific feature scaling and shifting."""

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

        # [B, F] -> [B, 1, 1, F]
        gamma = torch.tanh(
            gamma
        ).unsqueeze(1).unsqueeze(1)

        beta = (
            beta
            .unsqueeze(1)
            .unsqueeze(1)
        )

        return (
            features
            * (1.0 + gamma)
            + beta
        )


class GraphTemporalBlock(nn.Module):
    """
    Residual spatial-graph, temporal-convolution and feed-forward block.

    The tensor retains all electrode nodes throughout:
        [batch, electrode, time, feature]
    """

    def __init__(
        self,
        width: int,
        dilation: int,
        expansion: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        expanded_width = width * expansion

        self.graph_norm = nn.LayerNorm(
            width
        )

        self.graph_projection = nn.Linear(
            width,
            width,
            bias=False,
        )

        self.temporal_norm = nn.LayerNorm(
            width
        )

        self.temporal_depthwise = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=width,
            bias=False,
        )

        self.temporal_projection = nn.Linear(
            width,
            width,
            bias=False,
        )

        self.ffn_norm = nn.LayerNorm(
            width
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                width,
                expanded_width,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                expanded_width,
                width,
            ),
            nn.Dropout(dropout),
        )

        self.graph_dropout = nn.Dropout(
            dropout
        )

        self.temporal_dropout = nn.Dropout(
            dropout
        )

        self.graph_scale = nn.Parameter(
            torch.full(
                (width,),
                0.1,
            )
        )

        self.temporal_scale = nn.Parameter(
            torch.full(
                (width,),
                0.1,
            )
        )

        self.ffn_scale = nn.Parameter(
            torch.full(
                (width,),
                0.1,
            )
        )

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        # Spatial graph message passing.
        graph_features = self.graph_norm(
            features
        )

        graph_features = torch.einsum(
            "bij,bjtf->bitf",
            adjacency,
            graph_features,
        )

        graph_features = self.graph_projection(
            graph_features
        )

        graph_features = self.graph_dropout(
            graph_features
        )

        features = (
            features
            + graph_features
            * self.graph_scale.view(
                1,
                1,
                1,
                -1,
            )
        )

        # Temporal depthwise convolution for each electrode.
        temporal_features = self.temporal_norm(
            features
        )

        batch_size, n_nodes, n_times, width = (
            temporal_features.shape
        )

        temporal_features = (
            temporal_features
            .permute(0, 1, 3, 2)
            .reshape(
                batch_size * n_nodes,
                width,
                n_times,
            )
        )

        temporal_features = self.temporal_depthwise(
            temporal_features
        )

        temporal_features = (
            temporal_features
            .reshape(
                batch_size,
                n_nodes,
                width,
                n_times,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        temporal_features = self.temporal_projection(
            temporal_features
        )

        temporal_features = self.temporal_dropout(
            temporal_features
        )

        features = (
            features
            + temporal_features
            * self.temporal_scale.view(
                1,
                1,
                1,
                -1,
            )
        )

        feedforward_features = self.ffn(
            self.ffn_norm(features)
        )

        features = (
            features
            + feedforward_features
            * self.ffn_scale.view(
                1,
                1,
                1,
                -1,
            )
        )

        return features


class ElectrodeTemporalAttention(nn.Module):
    """
    Hierarchical pooling:

    1. Attention over the 61 electrode nodes.
    2. Attention over the 25 temporal positions.
    """

    def __init__(
        self,
        width: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        hidden_width = max(
            width // 2,
            32,
        )

        self.electrode_attention = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(
                width,
                hidden_width,
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_width,
                1,
                bias=False,
            ),
        )

        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(
                width,
                hidden_width,
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_width,
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
        return_attention: bool = False,
    ):
        # Electrode scores use each electrode's time-averaged state.
        electrode_context = features.mean(
            dim=2
        )

        electrode_scores = self.electrode_attention(
            electrode_context
        ).squeeze(-1)

        electrode_weights = torch.softmax(
            electrode_scores.float(),
            dim=1,
        ).to(dtype=features.dtype)

        # Preserve the temporal sequence while pooling electrodes.
        temporal_features = torch.einsum(
            "bc,bctf->btf",
            electrode_weights,
            features,
        )

        temporal_scores = self.temporal_attention(
            temporal_features
        ).squeeze(-1)

        temporal_weights = torch.softmax(
            temporal_scores.float(),
            dim=1,
        ).to(dtype=features.dtype)

        pooled = torch.einsum(
            "bt,btf->bf",
            temporal_weights,
            temporal_features,
        )

        pooled = self.output_norm(
            pooled
        )

        if return_attention:
            return (
                pooled,
                electrode_weights,
                temporal_weights,
            )

        return pooled


class GraphANNEncoder(nn.Module):
    """
    Electrode-preserving Graph-ANN keystroke encoder.

    Input:
        [B, 61 electrodes, 25 temporal samples]

    Output:
        [B, embedding width]
    """

    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 64,
        stem_features: int = 32,
        width: int = 192,
        graph_blocks: int = 6,
        graph_k: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.graph_k = graph_k

        # Shared per-electrode temporal filters. No electrode mixing
        # occurs in this frontend.
        self.temporal_stem = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=stem_features,
                kernel_size=(1, 5),
                padding=(0, 2),
                bias=False,
            ),
            nn.BatchNorm2d(
                stem_features
            ),
            nn.GELU(),
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                stem_features,
                width,
            ),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.subject_film = SubjectFiLM(
            width=width,
            n_subjects=n_subjects,
        )

        dilation_pattern = (
            1,
            2,
            4,
        )

        self.blocks = nn.ModuleList(
            [
                GraphTemporalBlock(
                    width=width,
                    dilation=dilation_pattern[
                        index
                        % len(dilation_pattern)
                    ],
                    expansion=2,
                    dropout=dropout,
                )
                for index in range(graph_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(
            width
        )

        self.pool = ElectrodeTemporalAttention(
            width=width,
            dropout=dropout,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor,
        channel_positions: torch.Tensor,
        return_attention: bool = False,
    ):
        if eeg.ndim != 3:
            raise ValueError(
                "Expected EEG [B, C, T], "
                f"received {tuple(eeg.shape)}"
            )

        if eeg.shape[1] != self.n_channels:
            raise ValueError(
                "EEG channel mismatch: "
                f"{eeg.shape[1]} versus {self.n_channels}"
            )

        eeg = eeg.float()

        adjacency = build_knn_adjacency(
            channel_positions.to(
                device=eeg.device,
                dtype=torch.float32,
            ),
            k=self.graph_k,
        )

        # [B, C, T] -> [B, 1, C, T]
        features = eeg.unsqueeze(1)

        # [B, stem, C, T]
        features = self.temporal_stem(
            features
        )

        # [B, C, T, stem]
        features = (
            features
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        features = self.input_projection(
            features
        )

        features = self.subject_film(
            features,
            subject_ids,
        )

        for block in self.blocks:
            features = block(
                features,
                adjacency,
            )

        features = self.final_norm(
            features
        )

        return self.pool(
            features,
            return_attention=return_attention,
        )


class NeuroQwertyGraphANNModule(
    pl.LightningModule
):
    """G1: fixed electrode graph + ANN dynamics + two-layer BiGRU."""

    def __init__(
        self,
        n_channels: int,
        n_subjects: int = 64,
        width: int = 192,
        graph_blocks: int = 6,
        graph_k: int = 4,
        gru_hidden: int = 192,
        gru_layers: int = 2,
        encoder_dropout: float = 0.25,
        gru_dropout: float = 0.25,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.eeg_encoder = GraphANNEncoder(
            n_channels=n_channels,
            n_subjects=n_subjects,
            width=width,
            graph_blocks=graph_blocks,
            graph_k=graph_k,
            dropout=encoder_dropout,
        )

        self.sequence_encoder = nn.GRU(
            input_size=width,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=(
                gru_dropout
                if gru_layers > 1
                else 0.0
            ),
            batch_first=True,
            bidirectional=True,
        )

        self.sequence_norm = nn.LayerNorm(
            gru_hidden * 2
        )

        self.classifier = nn.Linear(
            gru_hidden * 2,
            NUM_CLASSES,
        )

        # Label smoothing is used only for optimization.
        self.train_loss_function = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

        # Validation/test NLL remains ordinary cross-entropy so it is
        # directly comparable across experiments.
        self.eval_loss_function = nn.CrossEntropyLoss(
            label_smoothing=0.0,
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
                for sequence in sentence_embeddings
            ],
            device=embeddings.device,
            dtype=torch.long,
        )

        padded = pad_sequence(
            sentence_embeddings,
            batch_first=True,
        )

        packed = pack_padded_sequence(
            padded,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_contextual, _ = self.sequence_encoder(
            packed
        )

        contextual, _ = pad_packed_sequence(
            packed_contextual,
            batch_first=True,
            total_length=padded.shape[1],
        )

        contextual = self.sequence_norm(
            contextual
        )

        maximum_length = contextual.shape[1]

        valid_mask = (
            torch.arange(
                maximum_length,
                device=contextual.device,
            ).unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        logits = self.classifier(
            contextual[valid_mask]
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

        if stage == "train":
            loss = self.train_loss_function(
                logits,
                labels,
            )
        else:
            loss = self.eval_loss_function(
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