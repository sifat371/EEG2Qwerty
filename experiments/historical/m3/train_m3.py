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

from brain2qwerty_v1.config.xp_config import experiment_config
from brain2qwerty_v1.main import Data
from liteqwerty_eeg.model_m3 import (
    LiteQwertyGeoFusionModule,
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

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=33)

    parser.add_argument("--virtual-channels", type=int, default=48)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

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

    data_config["batch_size"] = args.batch_size
    data_config["val_batch_size"] = args.eval_batch_size
    data_config["test_batch_size"] = args.eval_batch_size
    data_config["num_workers"] = args.num_workers
    data_config["pin_memory"] = True
    data_config["persistent_workers"] = args.num_workers > 0

    print("Building official Brain2Qwerty EEG loaders...")

    loaders = Data(**data_config).build()

    first_batch = next(iter(loaders["train"]))

    eeg_shape = tuple(
        first_batch.data["neuro"].shape
    )

    position_shape = tuple(
        first_batch.data[
            "channel_positions"
        ].shape
    )

    n_channels = eeg_shape[1]

    model = LiteQwertyGeoFusionModule(
        n_channels=n_channels,
        n_virtual_channels=args.virtual_channels,
        width=args.width,
        encoder_depth=args.encoder_depth,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
    )

    parameters = count_parameters(model)

    print("Model: M3 LiteQwerty-GeoFusion")
    print("EEG shape:", eeg_shape)
    print("Channel-position shape:", position_shape)
    print("Virtual channels:", args.virtual_channels)
    print("Encoder width:", args.width)
    print("Encoder depth:", args.encoder_depth)
    print("Transformer depth:", args.transformer_depth)
    print("Transformer heads:", args.transformer_heads)
    print(f"Trainable parameters: {parameters:,}")
    print(
        "Reduction versus 623M reference: "
        f"{623_000_000 / parameters:.1f}x"
    )

    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best",
        monitor="val_CER",
        mode="min",
        save_top_k=1,
        save_last=True,
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
        callbacks=[
            checkpoint,
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
        ckpt_path="best",
    )

    print("Best checkpoint:", checkpoint.best_model_path)
    print("Best validation CER:", checkpoint.best_model_score)


if __name__ == "__main__":
    main()