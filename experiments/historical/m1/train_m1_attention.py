from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import studies  # noqa: F401

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from brain2qwerty_v1.config.xp_config import experiment_config
from brain2qwerty_v1.main import Data

from liteqwerty_eeg.model import LiteQwertyModule


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-model",
        choices=["transformer", "bigru"],
        default="transformer",
    )

    parser.add_argument(
        "--pooling",
        choices=["avg", "attention"],
        default="avg",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=33,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/historical/m1"),
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
    torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = experiment_config()
    data_config = config["data"]

    if args.debug:
        data_config["study"]["query"] = "timeline_index == 0"

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

    n_channels = eeg_shape[1]

    print("First EEG batch:", eeg_shape)
    print("EEG channels:", n_channels)
    print("Sequence model:", args.sequence_model)
    print("Temporal pooling:", args.pooling)
    print("Training batch size:", args.batch_size)
    print("Precision: bf16-mixed")

    model = LiteQwertyModule(
        n_channels=n_channels,
        sequence_model=args.sequence_model,
        pooling=args.pooling,
        learning_rate=args.learning_rate,
    )

    parameters = count_parameters(model)

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

    if "test" in loaders:
        trainer.test(
            model,
            dataloaders=loaders["test"],
            ckpt_path="best",
        )

    print("Best checkpoint:", checkpoint.best_model_path)
    print("Best validation CER:", checkpoint.best_model_score)


if __name__ == "__main__":
    main()
