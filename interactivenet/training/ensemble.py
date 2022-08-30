from pathlib import Path
import argparse
import os
import json
import numpy as np
import torch

import matplotlib.pyplot as plt

import mlflow.pytorch

from monai.transforms import (
    AsDiscrete,
    VoteEnsemble,
    MeanEnsemble
)
from monai.metrics import compute_meandice, compute_average_surface_distance, compute_hausdorff_distance

from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.statistics import ResultPlot

import nibabel as nib
from interactivenet.utils.resample import resample_label

if __name__=="__main__":
    parser = argparse.ArgumentParser(
            description="Ensembling of predicted weights"
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
    parser.add_argument(
        "-m",
        "--method",
        nargs="?",
        default="mean",
        help="Do you want to use mean or vote ensembling (default = mean)"
    )
    parser.add_argument(
        "-o",
        "--original_size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to change labels to the original size?"
    )

    args = parser.parse_args()
    raw = Path(os.environ["interactiveseg_raw"], args.task)
    exp = os.environ["interactiveseg_processed"]

    discrete = AsDiscrete(argmax=True, to_onehot=2)
    if args.method == "mean":
        method = MeanEnsemble()
    elif args.method == "vote":
        method = VoteEnsemble()
    else:
        raise KeyError(f"please provide either mean or vote as method for ensembling not {args.method}")

    if args.classes:
        types = raw / "types.json"
        if types.is_file():
            with open(types) as f:
                types = json.load(f)
                types = {v: key for key, value in types.items() for v in value}
        else:
            raise KeyError("types file not found")
    else:
        types = False

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

    n = 0
    names = []
    metas = []
    outputs = []
    for idx, run in runs.iterrows():
        if run["tags.Mode"] == "testing":
            experiment = Path(run["artifact_uri"].split("//")[-1])
            weights = experiment / "weights"
            if weights.is_dir():
                outputs.append(sorted([x for x in weights.glob("*.npz")]))
                metas(sorted([x for x in weights.glob("*.pkl")]))
                names.append(sorted([x.stem for x in output]))
            else:
                raise ValueError("No weights are available to ensemble, please use predict with -w or --weights to save outputs as weights")

            n += 1

    print(f"founds {n} folds to use in ensembling")
    if n <= 1:
        raise ValueError("Ensemble not possible because zero or 1 runs")
    elif any([set(x) != set(names[0]) for x in names]):
        raise ValueError("Not all runs have the same images")
    else:
        with mlflow.start_run(experiment_id=experiment_id, run_name="ensemble") as run:
            mlflow.set_tag('Mode', 'ensemble')
            mlflow.log_param("method", args.method)

            dices = {}
            hausdorff = {}
            surface = {}
            for output in zip(*outputs):
                name = output[0].stem
                pred = torch.from_numpy(np.stack([np.load(x)["weights"] for x in output], axis=0))
                
                if args.method == "vote":
                    pred = torch.stack([discrete(x) for x in pred])
                
                pred = method(pred)
                if args.method == "mean":
                    pred = discrete(pred)

                main = np.load(output[0])
                image = torch.from_numpy(main["image"])
                label = torch.from_numpy(main["label"])
                dice = compute_meandice(pred[None,:], label[None,:], include_background=False)
                dices[name] = dice.item()

                if args.original_size:
                    print('normal')
                    image = nib.load(str(raw / f"imagesTs/{name}_0000.nii.gz"))
                    label = nib.load(str(raw / f"labelsTs/{name}.nii.gz"))

                    image = torch.from_numpy(image.get_fdata())
                    label = torch.from_numpy(label.get_fdata())
                    new_pred = []

                    spacing_ratio = np.array(image_spacings) / np.array(self.target_spacing)
                    resample_shape = self.calculate_new_shape(spacing_ratio, original_shape)
                    for idx in range(pred.shape[0]):
                        new_pred.append(torch.from_numpy(resample_label(pred[idx].numpy()[None,:], label.shape, metadata["Fingerprint"]["Anisotropic"])))

                    pred = torch.cat(new_pred, 0)
                    test = AsDiscrete(to_onehot=2)
                    label = test(label[None,:])
                    image = image[None,:]

                    dice = compute_meandice(pred[None,:], label[None,:], include_background=False)
                    dices[name] = dice.item()
                    print(dices[name])
                else:
                    main = np.load(output[0])
                    image = torch.from_numpy(main["image"])
                    label = torch.from_numpy(main["label"])
                    image = image[:1]

                #dice = compute_meandice(pred[None,:], label[None,:], include_background=False)
                #dices[name] = dice.item()

                hausdorff_distance = compute_hausdorff_distance(pred[None,:], label[None,:], include_background=False)
                hausdorff[name] = hausdorff_distance.item()

                surface_distance = compute_average_surface_distance(pred[None,:], label[None,:], include_background=False)
                surface[name] = surface_distance.item()

                f = ImagePlot(image.numpy(), label[:1].numpy(), additional_scans=[pred[1:].numpy()])
                mlflow.log_figure(f, f"images/{name}.png")

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
            