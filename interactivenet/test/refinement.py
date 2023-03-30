##################################
##################################
## ALL CODE HERE IS OUTDATED!!! ##
##################################
##################################
##################################

from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import uuid

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    ToTensord,
    Compose,
    LoadImaged,
    ConcatItemsd,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    DivisiblePadd,
    CastToTyped,
    EnsureType,
    MeanEnsemble,
    Activationsd,
)

from monai.data import Dataset, DataLoader, decollate_batch

from interactivenet.transforms.transforms import (
    Resamplingd,
    EGDMapd,
    BoudingBoxd,
    NormalizeValuesd,
    OriginalSize,
    TestTimeFlipping,
    LoadWeightsd,
)
from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.statistics import ResultPlot, CalculateScores
from interactivenet.utils.postprocessing import ApplyPostprocessing

import torch
import pytorch_lightning as pl
import numpymaxflow

import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID


class Refinement(pl.LightningModule):
    def __init__(self, data, metadata):
        super().__init__()
        self.data = data
        self.metadata = metadata
        self.post_numpy = EnsureType("numpy", device="cpu")
        self.original_size = OriginalSize(
            metadata["Fingerprint"]["Anisotropic"], resample=False
        )

    def forward(self, x):
        return None

    def prepare_data(self):
        set_determinism(seed=0)

        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "annotation"]),
                LoadWeightsd(
                    keys=["weights"],
                    ref_image="image",
                ),
                Activationsd(keys=["weights"], softmax=True),
                EnsureChannelFirstd(keys=["image", "annotation"]),
                BoudingBoxd(
                    keys=["image", "annotation", "weights"],
                    on="annotation",
                    relaxation=0.1,
                    divisiblepadd=self.metadata["Plans"]["divisible by"],
                ),
                NormalizeValuesd(
                    keys=["image"],
                    clipping=self.metadata["Fingerprint"]["Clipping"],
                    mean=self.metadata["Fingerprint"]["Intensity_mean"],
                    std=self.metadata["Fingerprint"]["Intensity_std"],
                ),
                EGDMapd(
                    keys=["annotation"],
                    image="image",
                    lamb=1,
                    iter=4,
                    logscale=True,
                    powerof=2,
                    ct=self.metadata["Fingerprint"]["CT"],
                    backup=True,
                ),
                CastToTyped(
                    keys=["image", "annotation", "weights"],
                    dtype=(np.float32, np.float32, np.float32),
                ),
            ]
        )

        self.predict_ds = Dataset(
            data=self.data,
            transform=test_transforms,
        )

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )
        return predict_loader

    def predict_step(self, batch, batch_idx):
        img = np.asarray(batch["image"][0])
        cue_maps = np.asarray(batch["annotation"][0])
        interactions = np.asarray(batch["annotation_backup"][0])
        predictions = np.asarray(batch["weights"][0])

        if cue_maps.shape[0] == 1:
            background = np.zeros(cue_maps.shape[1:])
            cue_maps = np.stack([cue_maps[0], background])
            interactions = np.stack([interactions[0], background])

        seed = np.zeros(cue_maps.shape[1:], np.uint8)
        seed[interactions[0] > 0] = 255
        seed[interactions[1] > 0] = 170

        seed = np.asarray([seed == 255, seed == 170], np.uint8)

        foreground = np.maximum(cue_maps[0], predictions[1])
        background = np.maximum(cue_maps[1], predictions[0])

        prob = np.asarray(np.stack([background, foreground]), dtype=np.float32)
        softmax = np.exp(prob) / np.sum(np.exp(prob), axis=0)
        softmax = np.exp(softmax) / np.sum(np.exp(softmax), axis=0)

        lamda = 5.0
        sigma = 0.1
        connectivity = 6

        output = numpymaxflow.maxflow_interactive(
            img, prob, seed, lamda, sigma, connectivity
        )

        output = output.astype(int)
        output = np.stack([~output + 2, output])[None, :]

        meta = [
            self.post_numpy(i) for i in decollate_batch(batch["annotation_meta_dict"])
        ]
        output = [
            self.original_size(output, meta) for output, meta in zip(output, meta)
        ]

        return output, meta
        # return output, img, meta


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Preprocessing of ")
    parser.add_argument(
        "-t", "--task", nargs="?", default="Task710_STTMRI", help="Task name"
    )
    parser.add_argument(
        "-c",
        "--classes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to splits classes",
    )
    parser.add_argument(
        "-n",
        "--save_nifti",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save the output as nifti?",
    )

    args = parser.parse_args()
    exp = os.environ["interactivenet_processed"]
    raw = Path(os.environ["interactivenet_raw"], args.task)

    from interactivenet.utils.utils import (
        read_metadata,
        read_data,
        read_types,
        read_nifti,
    )

    data = read_data(raw, test=True)
    raw_data = read_data(raw)
    raw_data = read_nifti(raw_data)

    metadata = Path(exp, args.task, "plans.json")
    metadata = read_metadata(metadata)

    from interactivenet.utils.mlflow import mlflow_get_runs

    runs, experiment_id = mlflow_get_runs(args.task)

    if args.classes:
        types = read_types(raw / "types.json")
    else:
        types = False

    to_discrete = AsDiscrete(to_onehot=2)
    for idx, run in runs.iterrows():
        if run["tags.Mode"] != "ensemble":
            continue

        experiment = Path(run["artifact_uri"].split("//")[-1])
        weights = experiment / "weights"
        if weights.is_dir():
            weights = sorted([x for x in weights.glob("*.npz")])
        else:
            raise ValueError(
                "No weights are available to refine, please use ensemble with -w or --weights to save outputs as weights"
            )

        data = [dict(d, **{"weights": weight}) for d, weight in zip(data, weights)]

        method = Refinement(data, metadata)

        trainer = pl.Trainer(
            gpus=0,
        )

        with mlflow.start_run(
            experiment_id=experiment_id, run_name="refinement"
        ) as run:
            mlflow.set_tag("Mode", "refinement")

            outputs = trainer.predict(model=method)

            """
            for output, img, meta in outputs:
                output, img, meta = output[0], img[0], meta[0]
                name = Path(meta["filename_or_obj"]).name.split('.')[0]

                output = ApplyPostprocessing(output, "fillholes_and_largestcomponent")

                f = ImagePlot(img, output)
                mlflow.log_figure(f, f"images/{name}.png")
            """

            dices = {}
            hausdorff = {}
            surface = {}
            tmp_dir = Path(exp, str(uuid.uuid4()))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for output, meta in outputs:
                output, meta = output[0], meta[0]
                name = Path(meta["filename_or_obj"]).name.split(".")[0]

                image = raw_data[name]["image"]
                label = raw_data[name]["masks"]

                output = ApplyPostprocessing(output, "fillholes_and_largestcomponent")

                f = ImagePlot(
                    image,
                    label,
                    additional_scans=[output[1]],
                    CT=metadata["Fingerprint"]["CT"],
                )
                mlflow.log_figure(f, f"images/{name}.png")

                label = to_discrete(label[None, :])

                dice, hausdorff_distance, surface_distance = CalculateScores(
                    output, label
                )
                dices[name] = dice.item()
                hausdorff[name] = hausdorff_distance.item()
                surface[name] = surface_distance.item()

            mlflow.log_metric("Mean dice", np.mean(list(dices.values())))
            mlflow.log_metric("Std dice", np.std(list(dices.values())))

            f = ResultPlot(dices, "Dice", types)
            plt.close("all")
            mlflow.log_figure(f, f"dice.png")
            mlflow.log_dict(dices, "dice.json")

            f = ResultPlot(hausdorff, "Hausdorff Distance", types)
            plt.close("all")
            mlflow.log_figure(f, f"hausdorff_distance.png")
            mlflow.log_dict(hausdorff, "hausdorff_distance.json")

            f = ResultPlot(surface, "Surface Distance", types)
            plt.close("all")
            mlflow.log_figure(f, f"surface_distance.png")
            mlflow.log_dict(surface, "surface_distance.json")
            tmp_dir.rmdir()
