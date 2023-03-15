import argparse

import os
from typing import List, Tuple, Dict, Sequence, Optional, Callable, Union

from pathlib import Path
import argparse
import os
import json
import numpy as np
import torch
import pickle
import uuid
import warnings
from collections import Counter

import matplotlib.pyplot as plt

import mlflow.pytorch

from monai.transforms import (
    AsDiscrete,
    VoteEnsemble,
    MeanEnsemble
)
from monai.metrics import compute_meandice, compute_average_surface_distance, compute_hausdorff_distance

from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.statistics import ResultPlot, ComparePlot, CalculateScores, CalculateClinicalFeatures
from interactivenet.utils.results import AnalyzeResults
from interactivenet.utils.resample import resample_label
from interactivenet.utils.utils import read_metadata, read_types, read_nifti, read_dataset, check_gpu, save_niftis, save_weights, read_pickle
from interactivenet.utils.mlflow import mlflow_get_runs

import torch
import pytorch_lightning as pl

import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

def read_prediction(
    data:List[Dict[str, str]],
    task:str,
    raw_path:Optional[Union[str, os.PathLike]],
    results:Optional[Union[str, os.PathLike]],
):
    mlflow.set_tracking_uri(results)
    runs, experiment_id = mlflow_get_runs(task)

    raw_data, labels = read_nifti(data=data, raw_path=raw_path, rename_image="image_raw")

    n = 0
    outputs = []
    metas = []
    names = []
    postprocessing = []
    for idx, run in runs.iterrows():
        # looping to test experiments to extract weights and meta info
        if run["tags.Mode"] == "testing":
            experiment = Path(run["artifact_uri"].split("//")[-1])
            weights = experiment / "weights"
            if weights.is_dir():
                output = sorted([x for x in weights.glob("*.npz")])
                names.append(sorted([x.stem for x in output]))
                outputs.append(output)

                meta = sorted([x for x in weights.glob("*.pkl")])
                metas.append(meta)
            else:
                raise ValueError("No weights are available to ensemble, please use predict with -w or --weights to save outputs as weights")

            postprocessing.append(read_metadata(experiment / "postprocessing.json", error_message="postprocessing hasn't been run yet, please do this before predictions")["postprocessing"])
            
            n += 1

    print(f"founds {n} folds to use in ensembling")
    if n <= 1:
        raise ValueError("Ensemble not possible because zero or 1 runs")
    elif any([set(x) != set(names[0]) for x in names]):
        raise ValueError("Not all runs have the same images")
    elif n != 5:
        warnings.warn("NOT ALL 5 FOLDS WERE TRAINED!")

    results = []
    method = MeanEnsemble()
    for output, meta, names in zip(zip(*outputs), zip(*metas), zip(*names)):
        name = names[0]
        
        if any([x != name for x in names[1:]]):
            raise ValueError("Not images in runs match")

        meta = [read_pickle(x) for x in meta]

        output = np.stack([np.load(x)["weights"] for x in output], axis=0)
        output = method(output)

        raw = raw_data[name]
        results.append([
            [output],
            meta[0],
            raw,
            ]
        )
        
    return results, postprocessing, labels

def ensemble(
    outputs:list, 
    metadata:dict, 
    task:str,
    results:Optional[Union[str, os.PathLike]],
    postprocessing:List[str],
    weights:bool=False,
    niftis:bool=True,
    labels:bool=False,
    ):
    mlflow.set_tracking_uri(results)
    runs, experiment_id = mlflow_get_runs(task)

    postprocessing = Counter(postprocessing).most_common()[0][0]

    with mlflow.start_run(experiment_id=experiment_id, run_name="ensemble") as run:
        mlflow.set_tag('Mode', 'ensemble')
        mlflow.log_param("method", "mean ensembling")

        if weights:
            save_weights(mlflow, outputs)

        if niftis:
            save_niftis(mlflow, outputs, postprocessing=postprocessing)

        AnalyzeResults(mlflow=mlflow, outputs=outputs, postprocessing=postprocessing, metadata=metadata, labels=labels)

def main():
    parser = argparse.ArgumentParser(
            description="Ensembling of predicted weights"
         )
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    parser.add_argument(
        "-n",
        "--niftis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save the output as nifti?"
    )
    parser.add_argument(
        "-w",
        "--weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save weights as .npy in order to use in refinement?"
    )

    args = parser.parse_args()
    raw = Path(os.environ["interactiveseg_raw"], args.task)
    exp = Path(os.environ["interactiveseg_processed"], args.task)
    results = Path(os.environ["interactiveseg_results"], "mlruns")

    data, modalities = read_dataset(raw, mode="test")
    metadata = read_metadata(exp / "plans.json")

    outputs, postprocessing, labels = read_prediction(data=data, raw_path=raw, task=args.task, results=results)
    ensemble(outputs=outputs, metadata=metadata, task=args.task, results=results, postprocessing=postprocessing, weights=args.weights, niftis=args.niftis, labels=labels)

if __name__=="__main__":
    main()

