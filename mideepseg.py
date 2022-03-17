from pathlib import Path
import numpy as np

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    RandRotated,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    ToTensord,
    SpatialPadd,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    CropForegroundd,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    CastToTyped,
)
from monai.networks.nets import DynUNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, decollate_batch

from transforms import PreprocessAnisotropic, EGDMapd, BoudingBoxd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import mlflow.pytorch

def get_kernels_strides(sizes, spacings):
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

class Net(pl.LightningModule):
    def __init__(self, data_dir):
        super().__init__()

        self.kernels, self.strides = get_kernels_strides((192, 192, 32), (0.69097221, 0.69097221, 3.59999912))
        self._model = DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            kernel_size=self.kernels,
            strides=self.strides,
            upsample_kernel_size=self.strides[1:],
            filters=[4, 8, 16, 32, 64, 128],
            act_name = 'PRELU',
            deep_supervision = False,
        )
                
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.data_dir = data_dir
        
    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        images = sorted([f for f in Path(self.data_dir, "imagesTr").glob('**/*') if f.is_file()])
        points = sorted([f for f in Path(self.data_dir, "labelsTr").glob('**/*') if f.is_file()])
        labels = sorted([f for f in Path(self.data_dir, "interactionTr").glob('**/*') if f.is_file()])
        train_files =[ 
            {"image": image_name, "point": point_name, "label": label_name}
            for image_name, point_name, label_name in zip(images, points, labels)
            ]

        images = sorted([f for f in Path(self.data_dir, "imagesTs").glob('**/*') if f.is_file()])
        points = sorted([f for f in Path(self.data_dir, "labelsTs").glob('**/*') if f.is_file()])
        labels = sorted([f for f in Path(self.data_dir, "interactionTs").glob('**/*') if f.is_file()])       
        val_files =[ 
            {"image": image_name, "point": point_name, "label": label_name}
            for image_name, point_name, label_name in zip(images, points, labels)
            ]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        train_transforms = Compose(                                              
            [
                LoadImaged(keys=["image", "point", "label"]),  
                EnsureChannelFirstd(keys=["image", "point", "label"]),
                EGDMapd(
                    keys=["point"],
                    image="image",
                    lamb=1,
                    iter=4,
                ),
                PreprocessAnisotropic(
                    keys=["image", "point", "label"],
                    clip_values=(0, 0),
                    pixdim=(0.69097221, 0.69097221, 3.59999912),
                    normalize_values=(0, 0),
                    model_mode="train",
                ),
                BoudingBoxd(
                    keys=["image", "point", "label"],
                    on="point",
                    relaxation=(15, 15, 4),
                ),
                SpatialPadd(keys=["image", "point", "label"], spatial_size=(192, 192, 32)),
                RandCropByPosNegLabeld(
                    keys=["image", "point", "label"],
                    label_key="label",
                    spatial_size=(192, 192, 32),
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                RandZoomd(
                    keys=["image", "point", "label"],
                    min_zoom=0.9,
                    max_zoom=1.2,
                    mode=("trilinear", "trilinear", "nearest"),
                    align_corners=(True, True, None),
                    prob=0.15,
                ),
                RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
                RandFlipd(["image", "point", "label"], spatial_axis=[0], prob=0.5),
                RandFlipd(["image", "point", "label"], spatial_axis=[1], prob=0.5),
                RandFlipd(["image", "point", "label"], spatial_axis=[2], prob=0.5),
                CastToTyped(keys=["image", "point", "label"], dtype=(np.float32, np.float32, np.uint8)),
                ConcatItemsd(keys=["image", "point"], name="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "point", "label"]),
                EnsureChannelFirstd(keys=["image", "point", "label"]),
                EGDMapd(
                    keys=["point"],
                    image="image",
                    lamb=1,
                    iter=4,
                ),
                PreprocessAnisotropic(
                    keys=["image", "point", "label"],
                    clip_values=(0, 0),
                    pixdim=(0.69097221, 0.69097221, 3.59999912),
                    normalize_values=(0, 0),
                    model_mode="test",
                ),
                BoudingBoxd(
                    keys=["image", "point", "label"],
                    on="label",
                    relaxation=(10, 10, 2),
                ),
                SpatialPadd(keys=["image", "point", "label"], spatial_size=(192, 192, 32)),
                CastToTyped(keys=["image", "point", "label"], dtype=(np.float32, np.float32, np.uint8)),
                ConcatItemsd(keys=["image", "point"], name="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        self.train_ds = CacheDataset(
            data=train_files, transform=train_transforms,
        )
        self.val_ds = CacheDataset(
            data=val_files, transform=val_transforms,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=2, shuffle=True,                       
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
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)         
        self.log("loss", loss, on_epoch=True, batch_size=self.batch_size)
        return {"loss": loss, "logs": logs}
        
    def training_epoch_end(self, outputs):
        # Only required for logging it to mlflow for some reason.
        self.log("lr", self.lr_scheduler["scheduler"].get_last_lr()[0])
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (192, 192, 32)
        sw_batch_size = 2
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,
            overlap=0.5
        )
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

    network = Net("data/exp/LipoMoved/")
    
    trainer = pl.Trainer(
        gpus=-1, 
        max_epochs=3000, 
        num_sanity_val_steps=1,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        precision = 16,
        amp_backend="native",
        callbacks=[lr_logger],
    )

    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        trainer.fit(network)
        mlflow.pytorch.log_state_dict(network._model.state_dict(), "final_model_state_dict")