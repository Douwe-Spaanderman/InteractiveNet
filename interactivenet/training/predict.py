from pathlib import Path
import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    ToTensord,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    DivisiblePadd,
    CastToTyped,
)


from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import compute_meandice, compute_average_surface_distance

from interactivenet.transforms.transforms import Resamplingd, EGDMapd, BoudingBoxd
from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.statistics import ResultPlot

import torch
import pytorch_lightning as pl

import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

class Net(pl.LightningModule):
    def __init__(self, data, metadata, model):
        super().__init__()
        self._model = mlflow.pytorch.load_model(model, map_location=torch.device('cpu'))
        self.data = data
        self.metadata = metadata
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)
        self.batch_size = 1

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        set_determinism(seed=0)

        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "annotation", "mask"]),
                EnsureChannelFirstd(keys=["image", "annotation", "mask"]),
                Resamplingd(
                    keys=["image", "annotation", "mask"],
                    pixdim=metadata["Fingerprint"]["Target spacing"],
                ),
                BoudingBoxd(
                    keys=["image", "annotation", "mask"],
                    on="mask",
                    relaxation=0.1,
                    divisiblepadd=[16, 16, 4],
                ),
                NormalizeIntensityd(
                    keys=["image"],
                    nonzero=False,
                    channel_wise=False,
                ),
                EGDMapd(
                    keys=["annotation"],
                    image="image",
                    lamb=1,
                    iter=4,
                    logscale=True,
                ),
                DivisiblePadd(
                    keys=["image", "annotation", "mask"],
                    k=[16, 16, 4]
                ),
                CastToTyped(keys=["image", "annotation", "mask"], dtype=(np.float32, np.float32, np.uint8)),
                #ConcatItemsd(keys=["image", "annotation"], name="image"),
                ToTensord(keys=["image", "mask"]),
            ]
        )

        self.predict_ds = Dataset(
            data=self.data, transform=test_transforms,
        )

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4,
        )
        return predict_loader

    def predict_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["mask"]
        outputs = self.forward(images)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)

        return images, labels, outputs, batch["mask_meta_dict"]

if __name__=="__main__":
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
        "-c",
        "--classes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to splits classes"
    )
    args = parser.parse_args()
    exp = os.environ["interactiveseg_processed"]
    raw = Path(os.environ["interactiveseg_raw"], args.task)

    results = Path(os.environ["interactiveseg_results"], args.task)
    results.mkdir(parents=True, exist_ok=True)
    
    images = sorted([x for x in (raw / "imagesTs").glob('**/*') if x.is_file()])
    masks = sorted([x for x in (raw / "labelsTs").glob('**/*') if x.is_file()])
    annotations = sorted([x for x in (raw / "interactionTs").glob('**/*') if x.is_file()])

    data = [
        {"image": img_path, "mask": mask_path, "annotation": annot_path}
        for img_path, mask_path, annot_path in zip(images, masks, annotations)
    ]

    metadata = Path(exp, args.task, "plans.json")
    if metadata.is_file():
        with open(metadata) as f:
            metadata = json.load(f)
    else:
        raise KeyError("metadata not found")

    experiment_id = mlflow.get_experiment_by_name(args.task)
    if experiment_id == None:
        raise ValueError("Experiments not found, please first train models")
    else: experiment_id = experiment_id.experiment_id

    runs = mlflow.search_runs(experiment_id)

    if args.classes:
        types = raw / "types.json"
        if types.is_file():
            with open(types) as f:
                types = json.load(f)
                types = {v: key for key, value in types.items() for v in value}
                unseen = [{d["left out"]: False} for d in metadata["Plans"]["splits"]]
        else:
            raise KeyError("types file not found")
    else:
        types = False
        unseen = [False] * len(runs)

    for idx, run in runs.iterrows():
        if "tags.mlflow.parentRunId" in run:
            if run["tags.mlflow.parentRunId"] != None:
                continue

        run_id = run["run_id"]
        fold = run["params.fold"]
        model = "runs:/" + run_id + "/model"
        network = Net(data, metadata, model)

        trainer = pl.Trainer(
            gpus=0,
        )

        with mlflow.start_run(experiment_id=experiment_id, tags={MLFLOW_PARENT_RUN_ID: run_id}) as run:
            mlflow.set_tag('Mode', 'testing')
            outputs = trainer.predict(model=network)
            dices = {}
            surface = {}
            for image, label, output, meta in outputs:
                name = Path(meta["filename_or_obj"][0]).name.split('.')[0]
                save = Path(results, fold, name)

                dice = compute_meandice(output[0][None,:], label[0][None,:], include_background=False)
                dices[name] = dice.item()

                surface_distance = compute_average_surface_distance(output[0][None,:], label[0][None,:], include_background=False)
                surface[name] = surface_distance.item()
                
                f = ImagePlot(image[0][:1].numpy(), output[0][1:].numpy())
                mlflow.log_figure(f, f"images/{name}.png")

            mlflow.log_metric("Mean dice", np.mean(list(dices.values())))
            mlflow.log_metric("Std dice", np.std(list(dices.values())))

            f = ResultPlot(dices, "Dice", types, unseen[int(fold)])
            plt.close("all")
            mlflow.log_figure(f, f"dice.png")
            mlflow.log_dict(dices, "dice.json")

            f = ResultPlot(surface, "Surface Distance", types, unseen[int(fold)])
            plt.close("all")
            mlflow.log_figure(f, f"surface_distance.png")
            mlflow.log_dict(surface, "surface_distance.json")