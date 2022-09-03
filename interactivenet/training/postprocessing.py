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
from interactivenet.training.run import Net as TrainNet

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import mlflow.pytorch

class Net(pl.LightningModule):
    def __init__(self, data, metadata, model, split=0, checkpoint=None):
        super().__init__()
        self._model = mlflow.pytorch.load_model(model, map_location=torch.device('cuda'))
        self.data = data
        self.metadata = metadata
        self.split = split
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)

        self.fillholes = FillHoles(applied_labels=None, connectivity=2)
        self.largestcomponent = KeepLargestConnectedComponent(applied_labels=None, connectivity=2)

        self.postprocessing = Compose(
            [
                self.fillholes,
                self.largestcomponent
            ]
        )
        self.configurations = ["standard", "fillholes", "largestcomponent", "fillholes_and_largestcomponent"]

        self.checkpoint = checkpoint
        if self.checkpoint:
            self.checkpoint = TrainNet.load_from_checkpoint(checkpoint_path=checkpoint, data=data, metadata=metadata)._model
            self.configurations = ["standard", "checkpoint", "fillholes", "fillholes_ckpt", "largestcomponent", "largestcomponent_ckpt", "fillholes_and_largestcomponent", "fillholes_and_largestcomponent_ckpt"]

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

        outputs = torch.stack(outputs, dim=0).unsqueeze(0)        
        labels = torch.cat([self.post_label(i)[1:] for i in decollate_batch(labels)]*outputs.shape[1], dim=0).unsqueeze(0)
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        mean_val_dice = self.dice_metric.aggregate()
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

    mlflow.pytorch.autolog()
    for idx, run in runs.iterrows():
        if run["tags.Mode"] != "training": #or run["tags.Postprocessing"] == "Done":
            continue
        
        run_id = run["run_id"]
        fold = int(run["params.fold"])
        model = "runs:/" + run_id + "/model"
        ckpt = [x for x in Path(run["artifact_uri"].split('file://')[-1], "lightning_logs").glob("**/*.ckpt")][-1]
        with mlflow.start_run(run_id=run_id) as run:
            artifact_path = Path(mlflow.get_artifact_uri().split('file://')[-1])

            network = Net(data, metadata, model, split=fold, checkpoint=ckpt)
            trainer = pl.Trainer(
                gpus=-1,
                default_root_dir=artifact_path
            )
            
            output = trainer.validate(network)[0]
            values = list(output.values())
            best_score = list(output.keys())[values.index(max(values))]
            postprocessing = {}
            if best_score == "checkpoint" or "ckpt" in best_score:
                postprocessing["using_checkpoint"] = True
                mlflow.pytorch.log_model(network.checkpoint, "model_checkpoint")
                if best_score != "checkpoint":
                    postprocessing["postprocessing"] = "_".join(best_score.split("_")[:-1])
                else:
                    postprocessing["postprocessing"] = False
            else:
                postprocessing["using_checkpoint"] = False
                if best_score != "standard":
                    postprocessing["postprocessing"] = best_score
                else:
                    postprocessing["postprocessing"] = False

            mlflow.log_dict(postprocessing, "postprocessing.json")
            mlflow.set_tag('Postprocessing', 'Done')