from pathlib import Path
import argparse
import os
import json
import numpy as np
import torch
import shutil

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
    EnsureType,
    MeanEnsemble,
)

from interactivenet.transforms.transforms import (
    Resamplingd,
    EGDMapd,
    BoudingBoxd,
    NormalizeValuesd,
    OriginalSize,
    TestTimeFlipping,
)

from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import (
    compute_meandice,
    compute_average_surface_distance,
    compute_hausdorff_distance,
)

import nibabel as nib
from interactivenet.transforms.transforms import (
    Resamplingd,
    EGDMapd,
    BoudingBoxd,
    NormalizeValuesd,
)
from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.results import AnalyzeResults
from interactivenet.utils.statistics import (
    ResultPlot,
    ComparePlot,
    CalculateScores,
    CalculateClinicalFeatures,
)
from interactivenet.transforms.set_transforms import inference_transforms
from interactivenet.utils.utils import (
    save_weights,
    save_niftis,
    read_metadata,
    read_types,
    read_nifti,
    read_dataset,
    check_gpu,
    read_data_inference,
    to_array,
)
from interactivenet.utils.mlflow import mlflow_get_runs, mlflow_get_id
from interactivenet.utils.postprocessing import ApplyPostprocessing

import torch
import pytorch_lightning as pl

import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID


def main():
    parser = argparse.ArgumentParser(
        description="Predict on the interactivenet network"
    )
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    args = parser.parse_args()

    exp = Path(os.environ["interactivenet_processed"], args.task)
    results = Path(os.environ["interactivenet_results"], "mlruns")
    metadata = read_metadata(exp / "plans.json")
    if "Cases" in metadata:
        del metadata["Cases"]
    if "splits" in metadata["Plans"]:
        del metadata["Plans"]["splits"]

    mlflow.set_tracking_uri(results)
    runs, experiment_id = mlflow_get_runs(args.task)

    models = {}
    postprocessings = {}
    for idx, run in runs.iterrows():
        if run["tags.Mode"] != "training":
            continue

        run["run_id"]
        fold = run["params.fold"]
        artifact_uri = Path(run["artifact_uri"].split("file://")[-1])
        postprocessing = read_metadata(
            artifact_uri / "postprocessing.json",
            error_message="postprocessing hasn't been run yet, please do this before predictions",
        )
        if postprocessing["using_checkpoint"]:
            models[fold] = artifact_uri / "model_checkpoint"
        else:
            models[fold] = artifact_uri / "model"

        postprocessings[fold] = postprocessing

    results = Path(os.environ["interactivenet_results"], "models", args.task)
    results.mkdir(parents=True, exist_ok=True)

    print(f"Saving models in {results}")
    with open(results / "plans.json", "w") as outfile:
        json.dump(metadata, outfile)

    with open(results / "postprocessings.json", "w") as outfile:
        json.dump(postprocessings, outfile)

    for fold, model in models.items():
        shutil.copytree(model, results / "model" / fold)


if __name__ == "__main__":
    main()
