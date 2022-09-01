from pathlib import Path
import numpy as np
import os
import json
import argparse

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    ConcatItemsd,
    ToTensord,
    CastToTyped,
    KeepLargestConnectedComponent,
    FillHoles
)

from monai.networks.nets import DynUNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import Dataset, DataLoader, decollate_batch

from interactivenet.transforms.transforms import LoadPreprocessed
from interactivenet.networks.unet import UNet

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import mlflow.pytorch

class Net(pl.LightningModule):
    def __init__(self, data, metadata, model, split=0, checkpoint=None):
        super().__init__()
        self._model = mlflow.pytorch.load_model(model, map_location=torch.device('cpu'))
        self.data = data
        self.metadata = metadata
        self.split = split
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)
        self.supervision_weights = metadata["Plans"]["deep supervision weights"]

        self.fillholes = FillHoles(applied_labels=None, connectivity=2)
        self.largestcomponent = KeepLargestConnectedComponent(applied_labels=None, connectivity=2)

        self.postprocessing = Compose(
            [
                self.fillholes,
                self.largestcomponent
            ]
        )
        self.configurations = ["standard", "fillholes", "largestcomponent", "both"]

        self.checkpoint = checkpoint
        if self.checkpoint:
            self.checkpoint = DynUNet(
                spatial_dims=3,
                in_channels=2,
                out_channels=2,
                kernel_size=metadata["Plans"]["kernels"],
                strides=metadata["Plans"]["strides"],
                upsample_kernel_size=metadata["Plans"]["strides"][1:],
                filters=[4, 8, 16, 32, 64, 128],
                norm_name= 'instance',
                act_name = 'leakyrelu',
                deep_supervision = True,
                deep_supr_num = metadata["Plans"]["deep supervision"]
            )
            self.checkpoint = self.checkpoint.load_from_checkpoint(checkpoint)
            self.configurations = ["standard", "checkpoint", "fillholes", "fillholes_ckpt", "largestcomponent", "largestcomponent_ckpt", "both", "both_ckpt"]

    def forward(self, x, model):
        return model(x)

    def prepare_data(self):
        set_determinism(seed=0)
        
        val_transforms = Compose(
            [
                LoadPreprocessed(keys=["npz", "metadata"], new_keys=["image", "annotation", "mask"]),
                CastToTyped(keys=["image", "annotation", "mask"], dtype=(np.float32, np.float32, np.uint8)),
                ConcatItemsd(keys=["image", "annotation"], name="image"),
                ToTensord(keys=["image", "mask"]),
            ]
        )

        split = self.metadata["Plans"]["splits"][self.split]
        val_data = [x for x in self.data if x['npz'].stem in split['val']]

        self.val_ds = Dataset(
            data=val_data, transform=val_transforms,
        )

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["mask"]
        outputs = []
        outputs.append(self.forward(images, self._model))

        if self.checkpoint:
            outputs.append(self.forward(images, self.checkpoint))

        tmp = []
        for output in outputs:
            tmp.extend([self.post_pred(i)[1] for i in decollate_batch(output)])

        outputs = tmp.copy()
        for postprocessing in [self.fillholes, self.largestcomponent, self.postprocessing]:
            outputs.extend([postprocessing(x) for x in tmp])

        print(outputs[0].shape)
        print(torch.equal(outputs[0], outputs[-1]))

        outputs = torch.stack(outputs, dim=0).unsqueeze(0)        
        labels = torch.cat([self.post_label(i)[1:] for i in decollate_batch(labels)]*outputs.shape[1], dim=0).unsqueeze(0)
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        mean_val_dice = self.dice_metric.aggregate()
        print(mean_val_dice)
        print(mean_val_dice.item())
        [self.log(self.configurations[i], x) for i, x in enumerate(mean_val_dice)]
        return mean_val_dice


if __name__=="__main__":
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
    args = parser.parse_args()
    exp = Path(os.environ["interactiveseg_processed"], args.task)

    from interactivenet.utils.utils import read_processed, read_metadata
    data = read_processed(exp)
    metadata = read_metadata(exp / "plans.json")


    from interactivenet.utils.mlflow import mlflow_get_runs
    runs, experiment_id = mlflow_get_runs(args.task)

    runs = [runs.loc[1]]

    mlflow.pytorch.autolog()
    for idx, run in enumerate(runs):#runs.iterrows():
        if run["tags.Mode"] != "training": #or run["tags.Postprocessing"] == "Done":
            continue
        
        run_id = run["run_id"]
        fold = int(run["params.fold"])
        model = "runs:/" + run_id + "/model"

        with mlflow.start_run(experiment_id=experiment_id, run_name='test') as run: #run_id=run_id
            #mlflow.set_tag('Postprocessing', 'Done')
            artifact_path = Path(mlflow.get_artifact_uri().split('file://')[-1])

            network = Net(data, metadata, model, split=fold)
            trainer = pl.Trainer(
                gpus=0,
                default_root_dir=artifact_path
            )
            
            trainer.validate(network)
