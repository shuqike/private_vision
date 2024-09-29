import argparse
from pathlib import Path
import torch
from torch.nn import Identity
from torchvision.datasets import CIFAR10

import timm

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.utils.benchmarking import KNNClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero


def knn_eval(
    model: LightningModule,
    data_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    num_classes: int,
) -> None:
    """Runs KNN evaluation on the given model.

    Parameters follow InstDisc [0] settings.

    The most important settings are:
        - Num nearest neighbors: 200
        - Temperature: 0.1

    References:
       - [0]: InstDict, 2018, https://arxiv.org/abs/1805.01978
    """
    print_rank_zero("Running KNN evaluation...")

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    val_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_dataset = LightlyDataset.from_torch_dataset(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True,
    )

    # Setup validation data.
    val_dataset = LightlyDataset.from_torch_dataset(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    classifier = KNNClassifier(
        model=model,
        num_classes=num_classes,
        feature_dtype=torch.float16,
    )

    # Run KNN evaluation.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=devices,
        logger=TensorBoardLogger(save_dir=str(log_dir), name="knn_eval"),
        callbacks=[
            DeviceStatsMonitor(),
            metric_callback,
        ],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    for metric in ["val_top1", "val_top5"]:
        print_rank_zero(f"knn {metric}: {max(metric_callback.val_metrics[metric])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    seed = 9
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
    model.head = Identity()
    checkpoint = torch.load("C:/Science/private_vision/checkpoints/vit_base_patch16_224_epoch=0_cnt=1.ckpt")
    msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print_rank_zero("Backbone missing keys", msg.missing_keys)
    pl.seed_everything(seed)
    knn_eval(
        model=model,
        data_dir=Path("C:/Science/datasets/cifar10"),
        log_dir=Path("C:/Science/iclr2024_distort/logs"),
        batch_size_per_device=100,
        num_workers=2,
        accelerator="gpu",
        devices=[1],
        num_classes=10
    )
