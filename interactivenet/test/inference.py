from pathlib import Path
import argparse
import os
import pickle
import json
import numpy as np
import torch
import uuid

import matplotlib.pyplot as plt

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
    VoteEnsemble,
    MeanEnsemble
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import compute_meandice, compute_average_surface_distance, compute_hausdorff_distance

import nibabel as nib
from interactivenet.transforms.transforms import Resamplingd, EGDMapd, BoudingBoxd, NormalizeValuesd
from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.statistics import ResultPlot, CalculateScores, CalculateClinicalFeatures
from interactivenet.test.predict import Net

import torch
import pytorch_lightning as pl

import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

if __name__=="__main__":
    parser = argparse.ArgumentParser(
            description="Ensembling of predicted weights"
         )
    parser.add_argument(
        "-i",
        "--input",
        nargs="?",
        default="Task710_STTMRI",
        help="Task input name"
    )
    parser.add_argument(
        "-t",
        "--task",
        nargs="?",
        default="Task710_STTMRI",
        help="Task models name"
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
        "--save_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save weights as .npy in order to use in refinement?"
    )
    parser.add_argument(
        "-n",
        "--save_nifti",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save the output as nifti?"
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
    raw = Path(os.environ["interactiveseg_raw"], args.input)
    results = Path(os.environ["interactiveseg_results"], args.input)
    results.mkdir(parents=True, exist_ok=True)

    to_discrete = AsDiscrete(to_onehot=2)
    to_discrete_argmax = AsDiscrete(argmax=True)
    if args.method == "mean":
        method = MeanEnsemble()
    elif args.method == "vote":
        method = VoteEnsemble()
    else:
        raise KeyError(f"please provide either mean or vote as method for ensembling not {args.method}")

    from interactivenet.utils.utils import read_metadata, read_data, read_types, read_nifti
    data = read_data(raw, test=True)
    raw_data = read_data(raw)
    raw_data = read_nifti(raw_data)

    metadata = Path(exp, args.task, "plans.json")
    metadata = read_metadata(metadata)

    from interactivenet.utils.mlflow import mlflow_get_runs, mlflow_get_id
    runs, experiment_id = mlflow_get_runs(args.task)

    if args.classes:
        types = read_types(raw / "types.json")
    else:
        types = False

    outputs = []
    metas = []
    for idx, run in runs.iterrows():
        if run["tags.Mode"] != "training":
            continue

        run_id = run["run_id"]
        fold = run["params.fold"]
        postprocessing = Path(run["artifact_uri"].split('file://')[-1], "postprocessing.json")
        postprocessing = read_metadata(postprocessing, error_message="postprocessing hasn't been run yet, please do this before predictions")
        if postprocessing["using_checkpoint"]:
            model = "runs:/" + run_id + "/model_checkpoint"
        else:
            model = "runs:/" + run_id + "/model"

        network = Net(data, metadata, model)

        trainer = pl.Trainer(
            gpus=-1,
        )
        
        output = trainer.predict(model=network)
        tmp_output = []
        tmp_meta = []
        for weight, meta in output:
            weight, meta = weight[0], meta[0]
            tmp_output.append(weight)
            tmp_meta.append(meta)

        outputs.append(tmp_output)
        metas.append(tmp_meta)

    experiment_id = mlflow_get_id("ensemble")

    tmp_dir = Path(exp, str(uuid.uuid4()))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(experiment_id=experiment_id, run_name=args.input) as run:
        mlflow.set_tag('Mode', 'ensemble')
        mlflow.log_param("method", args.method)

        dices = {}
        hausdorff = {}
        surface = {}
        for weight, meta in zip(zip(*outputs), zip(*metas)):
            name = Path(meta[0]["filename_or_obj"]).name.split('.')[0]
            image = raw_data[name]["image"]
            label = raw_data[name]["masks"]
                
            output = np.stack(weight, axis=0)

            if args.method == "vote":
                if args.save_weights:
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

            if args.save_weights:
                data_file = tmp_dir / f"{name}.npz"

                np.savez(str(data_file), weights=weight, pred=output, label=label)
                mlflow.log_artifact(str(data_file), artifact_path="weights")
                data_file.unlink()
            
            if args.save_nifti:
                data_file = tmp_dir / f"{name}.nii.gz"
                meta_dict = raw_data[name]["image_meta_dict"]

                output = nib.Nifti1Image(output[1], meta_dict.get_sform())
                nib.save(output, str(data_file))
                mlflow.log_artifact(str(data_file), artifact_path="niftis")
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