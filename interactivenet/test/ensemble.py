from pathlib import Path
import argparse
import os
import json
import numpy as np
import torch
import pickle
import uuid

import matplotlib.pyplot as plt

import mlflow.pytorch

from monai.transforms import (
    AsDiscrete,
    VoteEnsemble,
    MeanEnsemble
)
from monai.metrics import compute_meandice, compute_average_surface_distance, compute_hausdorff_distance

from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.statistics import ResultPlot, CalculateScores

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
        "-w",
        "--weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save weights as .npy in order to use in refinement?"
    )
    parser.add_argument(
        "-m",
        "--method",
        nargs="?",
        default="mean",
        help="Do you want to use mean or vote ensembling (default = mean)"
    )

    args = parser.parse_args()
    exp = os.environ["interactiveseg_processed"]
    raw = Path(os.environ["interactiveseg_raw"], args.task)

    from interactivenet.utils.utils import read_metadata, read_data, read_types, read_nifti
    raw_data = read_data(raw)
    raw_data = read_nifti(raw_data)

    to_discrete = AsDiscrete(to_onehot=2)
    to_discrete_argmax = AsDiscrete(argmax=True)
    if args.method == "mean":
        method = MeanEnsemble()
    elif args.method == "vote":
        method = VoteEnsemble()
    else:
        raise KeyError(f"please provide either mean or vote as method for ensembling not {args.method}")

    metadata = Path(exp, args.task, "plans.json")
    metadata = read_metadata(metadata)

    from interactivenet.utils.mlflow import mlflow_get_runs
    runs, experiment_id = mlflow_get_runs(args.task)

    if args.classes:
        types = read_types(raw / "types.json")
    else:
        types = False

    n = 0
    names = []
    metas = []
    outputs = []
    for idx, run in runs.iterrows():
        if run["tags.Mode"] == "testing":
            experiment = Path(run["artifact_uri"].split("//")[-1])
            weights = experiment / "weights"
            if weights.is_dir():
                output = sorted([x for x in weights.glob("*.npz")])
                metas.append(sorted([x for x in weights.glob("*.pkl")]))
                names.append(sorted([x.stem for x in output]))
                outputs.append(output)
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
            tmp_dir = Path(exp, str(uuid.uuid4()))
            tmp_dir.mkdir(parents=True, exist_ok=True)

            for output, name in zip(zip(*outputs), names[0]):
                image = raw_data[name]["image"]
                label = raw_data[name]["masks"]
                output = np.stack([np.load(x)["weights"] for x in output], axis=0)

                if args.method == "vote":
                    if args.weights:
                        raise KeyError("Cannot use vote ensembling when you want to save weights")
                    output = np.stack([to_discrete_argmax(x) for x in output])
                
                output = method(output)
                weight = output.copy()
                if args.method == "mean":
                    output = to_discrete_argmax(output)

                f = ImagePlot(image, label, additional_scans=[output[0]], CT=metadata["Fingerprint"]["CT"])
                mlflow.log_figure(f, f"images/{name}.png")
                
                label = to_discrete(label[None,:])
                output = to_discrete(output)
                
                dice, hausdorff_distance, surface_distance = CalculateScores(output, label)
                dices[name] = dice.item()
                hausdorff[name] = hausdorff_distance.item()
                surface[name] = surface_distance.item()

                if args.weights:
                    data_file = tmp_dir / f"{name}.npz"

                    np.savez(str(data_file), weights=weight, pred=output, label=label)
                    mlflow.log_artifact(str(data_file), artifact_path="weights")
                    data_file.unlink()

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
            