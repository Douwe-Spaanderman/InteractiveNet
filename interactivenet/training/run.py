from pathlib import Path
import numpy as np
import os
import json

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    RandFlipd,
    RandScaleIntensityd,
    ConcatItemsd,
    ToTensord,
    SpatialPadd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    CastToTyped,
)
from monai.networks.nets import DynUNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch

from interactivenet.transforms.transforms import LoadPreprocessed
from interactivenet.networks.unet import UNet

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

import mlflow.pytorch

class Net(pl.LightningModule):
    def __init__(self, data, metadata, split=0):
        super().__init__()
        self._model = DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            kernel_size=metadata["Plans"]["kernels"],
            strides=metadata["Plans"]["strides"],
            upsample_kernel_size=metadata["Plans"]["strides"][1:],
            filters=[4, 8, 16, 32, 64, 128],
            act_name = 'leakyrelu',
            deep_supervision = False,
        )
        self.data = data
        self.metadata = metadata
        self.split = split
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 3000
        self.batch_size = 1

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        set_determinism(seed=0)

        train_transforms = Compose(
            [
                LoadPreprocessed(keys=["npz", "metadata"], new_keys=["image", "annotation", "mask"]),
                RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
                RandFlipd(keys=["image", "annotation", "mask"], spatial_axis=[0], prob=0.5),
                RandFlipd(keys=["image", "annotation", "mask"], spatial_axis=[1], prob=0.5),
                RandFlipd(keys=["image", "annotation", "mask"], spatial_axis=[2], prob=0.5),
                CastToTyped(keys=["image", "annotation", "mask"], dtype=(np.float32, np.float32, np.uint8)),
                ConcatItemsd(keys=["image", "annotation"], name="image"),
                ToTensord(keys=["image", "mask"]),
                ]
        )
        val_transforms = Compose(
            [
                LoadPreprocessed(keys=["npz", "metadata"], new_keys=["image", "annotation", "mask"]),
                CastToTyped(keys=["image", "annotation", "mask"], dtype=(np.float32, np.float32, np.uint8)),
                ConcatItemsd(keys=["image", "annotation"], name="image"),
                ToTensord(keys=["image", "mask"]),
            ]
        )

        split = self.metadata["Plans"]["splits"][self.split]
        train_data = [x for x in self.data if x['npz'].stem in split['train']]
        val_data = [x for x in self.data if x['npz'].stem in split['val']]

        self.train_ds = Dataset(
            data=train_data, transform=train_transforms,
        )
        self.val_ds = Dataset(
            data=val_data, transform=val_transforms,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01, momentum=0.99, weight_decay=3e-5, nesterov=True)
        self.lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: (1 - epoch / self.max_epochs) ** 0.9),
            'name': 'lr_sched'
        }
        return [self.optimizer], [self.lr_scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["mask"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        self.log("loss", loss, on_epoch=True, batch_size=self.batch_size)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # Only required for logging it to mlflow for some reason.
        self.log("lr", self.lr_scheduler["scheduler"].get_last_lr()[0])

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["mask"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("curent epoch", self.current_epoch, on_epoch=True)
        self.log("current mean dice", mean_val_dice, on_epoch=True)
        self.log("best mean dice", self.best_val_dice, on_epoch=True)
        self.log("at epoch", self.best_val_epoch, on_epoch=True)
        return mean_val_dice, mean_val_loss

if __name__=="__main__":
    lr_logger = LearningRateMonitor(logging_interval="epoch")

    import argparse
    import os

    parser = argparse.ArgumentParser(
             description="Preprocessing of "
         )
    parser.add_argument(
         "-t",
         "--task",
         nargs="?",
         default="Task710_STTMRI",
         help="Task name"
    )
    parser.add_argument(
         "-f",
         "--fold",
         nargs="?",
         default=0,
         type=int,
         help="which fold do you want to train?"
    )
    args = parser.parse_args()
    exp = os.environ["interactiveseg_processed"]

    arrays = [x for x in Path(exp, args.task, "network_input").glob('**/*.npz') if x.is_file()]
    metafile = [x for x in Path(exp, args.task, "network_input").glob('**/*.pkl') if x.is_file()]
    metadata = Path(exp, args.task, "plans.json")
    if metadata.is_file():
        with open(metadata) as f:
            metadata = json.load(f)
    else:
        raise KeyError("metadata not found")

    data = [
            {"npz": npz_path, "metadata": metafile_path}
            for npz_path, metafile_path in zip(sorted(arrays), sorted(metafile))
        ]

    network = Net(data, metadata, split=args.fold)
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=500,
        num_sanity_val_steps=1,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        callbacks=[lr_logger],
        accumulate_grad_batches=4,
    )

    experiment_id = mlflow.get_experiment_by_name(args.task)
    if experiment_id == None:
        print(f"experiment_id not found will create {args.task}")
        experiment_id = mlflow.create_experiment(args.task)
    else: experiment_id = experiment_id.experiment_id

    mlflow.pytorch.autolog()

    with mlflow.start_run(experiment_id=experiment_id, run_name=args.fold) as run:
        mlflow.set_tag('Mode', 'training')
        mlflow.log_param("fold", args.fold)
        trainer.fit(network)
