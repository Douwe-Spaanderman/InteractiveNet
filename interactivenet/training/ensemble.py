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

    args = parser.parse_args()
    raw = Path(os.environ["interactiveseg_raw"], args.task)

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

    experiment_id = mlflow.get_experiment_by_name(args.task)
    if experiment_id == None:
        raise ValueError("Experiments not found, please first train models")
    else: experiment_id = experiment_id.experiment_id

    runs = mlflow.search_runs(experiment_id)

    n = 0
    names = []
    outputs = []
    for idx, run in runs.iterrows():
        if run["tags.Mode"] == "testing":
            exp = Path(run["artifact_uri"].split("//")[-1])
            weights = exp / "weights"
            if weights.is_dir():
                output = [x for x in weights.glob("*.npz")]
                names.append([x.stem for x in output])
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

            outputs = [sorted(x) for x in outputs]

            dices = {}
            hausdorff = {}
            surface = {}
            for output in zip(*outputs):
                name = output[0].stem
                main = np.load(output[0])
                image = torch.from_numpy(main["image"])
                label = torch.from_numpy(main["label"])

                pred = torch.from_numpy(np.stack([np.load(x)["weights"] for x in output], axis=0))

                if args.method == "vote":
                    pred = torch.stack([discrete(x) for x in pred])
                
                pred = method(pred)
                if args.method == "mean":
                    pred = discrete(pred)

                dice = compute_meandice(pred[None,:], label[None,:], include_background=False)
                dices[name] = dice.item()

                hausdorff_distance = compute_hausdorff_distance(pred[None,:], label[None,:], include_background=False)
                hausdorff[name] = hausdorff_distance.item()

                surface_distance = compute_average_surface_distance(pred[None,:], label[None,:], include_background=False)
                surface[name] = surface_distance.item()

                f = ImagePlot(image[:1].numpy(), label[:1].numpy(), additional_scans=[pred[1:].numpy()])
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
            