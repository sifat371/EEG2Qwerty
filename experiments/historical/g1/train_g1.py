from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import studies  # noqa: F401
import torch

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from brain2qwerty_v1.config.xp_config import (
    experiment_config,
)
from brain2qwerty_v1.main import Data

from liteqwerty_eeg.model_g1 import (
    NeuroQwertyGraphANNModule,
    build_knn_adjacency,
)


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=33,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--graph-blocks",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--graph-k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gru-hidden",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--gru-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of physical batches accumulated per optimizer step.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pl.seed_everything(
        args.seed,
        workers=True,
    )

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    torch.set_float32_matmul_precision(
        "high"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = experiment_config()
    data_config = config["data"]

    if args.debug:
        data_config["study"]["query"] = (
            "timeline_index == 0"
        )

    data_config["batch_size"] = (
        args.batch_size
    )

    data_config["val_batch_size"] = (
        args.eval_batch_size
    )

    data_config["test_batch_size"] = (
        args.eval_batch_size
    )

    data_config["num_workers"] = (
        args.num_workers
    )

    data_config["pin_memory"] = True

    data_config["persistent_workers"] = (
        args.num_workers > 0
    )

    print(
        "Building official Brain2Qwerty EEG loaders..."
    )

    loaders = Data(
        **data_config
    ).build()

    first_batch = next(
        iter(loaders["train"])
    )

    eeg_shape = tuple(
        first_batch.data[
            "neuro"
        ].shape
    )

    position_shape = tuple(
        first_batch.data[
            "channel_positions"
        ].shape
    )

    n_channels = eeg_shape[1]

    model = NeuroQwertyGraphANNModule(
        n_channels=n_channels,
        width=args.width,
        graph_blocks=args.graph_blocks,
        graph_k=args.graph_k,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    parameters = count_parameters(
        model
    )

    with torch.no_grad():
        adjacency = build_knn_adjacency(
            first_batch.data[
                "channel_positions"
            ][:1].float(),
            k=args.graph_k,
        )

        symmetry_error = (
            adjacency
            - adjacency.transpose(1, 2)
        ).abs().max().item()

        active_edges = (
            adjacency[0] > 0
        ).sum().item()

        average_connections = (
            active_edges
            / adjacency.shape[1]
        )

    print(
        "Model: G1 NeuroQwerty Graph-ANN + BiGRU"
    )
    print("EEG shape:", eeg_shape)
    print(
        "Channel-position shape:",
        position_shape,
    )
    print(
        "Graph neighbours:",
        args.graph_k,
    )
    print(
        "Average graph connections including self:",
        f"{average_connections:.2f}",
    )
    print(
        "Adjacency symmetry error:",
        f"{symmetry_error:.8f}",
    )
    print(
        "Graph width:",
        args.width,
    )
    print(
        "Graph-temporal blocks:",
        args.graph_blocks,
    )
    print(
        "BiGRU hidden size:",
        args.gru_hidden,
    )
    print(
        "BiGRU layers:",
        args.gru_layers,
    )
    print(
        "Physical batch size:",
        args.batch_size,
    )
    print(
        "Gradient accumulation:",
        args.accumulate_grad_batches,
    )
    print(
        "Effective batch size:",
        args.batch_size * args.accumulate_grad_batches,
    )
    print(
        f"Trainable parameters: {parameters:,}"
    )
    print(
        "Reduction versus 623M reference: "
        f"{623_000_000 / parameters:.1f}x"
    )

    checkpoint_cer = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-cer",
        monitor="val_CER",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    checkpoint_loss = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-loss",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_CER",
        mode="min",
        patience=args.patience,
    )

    logger = CSVLogger(
        save_dir=args.output_dir,
        name="logs",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            checkpoint_cer,
            checkpoint_loss,
            early_stopping,
            LearningRateMonitor(
                logging_interval="step"
            ),
        ],
        logger=logger,
        default_root_dir=args.output_dir,
        log_every_n_steps=20,
        benchmark=True,
        deterministic=False,
    )

    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )

    trainer.test(
        model,
        dataloaders=loaders["test"],
        ckpt_path=checkpoint_cer.best_model_path,
    )

    print(
        "Best CER checkpoint:",
        checkpoint_cer.best_model_path,
    )
    print(
        "Best validation CER:",
        checkpoint_cer.best_model_score,
    )
    print(
        "Best loss checkpoint:",
        checkpoint_loss.best_model_path,
    )
    print(
        "Best validation loss:",
        checkpoint_loss.best_model_score,
    )


if __name__ == "__main__":
    main()