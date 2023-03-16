import argparse

import os
from typing import List, Tuple, Dict, Sequence, Optional, Callable, Union

from pathlib import Path
import numpy as np
import pickle
import json
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
    MeanEnsemble
)

from monai.data import Dataset, DataLoader, decollate_batch

from interactivenet.transforms.transforms import (
    Resamplingd, 
    EGDMapd, 
    BoudingBoxd, 
    NormalizeValuesd, 
    OriginalSize,
    TestTimeFlipping
)
from interactivenet.transforms.set_transforms import inference_transforms
from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.statistics import ResultPlot, CalculateScores
from interactivenet.utils.postprocessing import ApplyPostprocessing
from interactivenet.utils.results import AnalyzeResults
from interactivenet.utils.utils import save_weights, save_niftis, read_metadata, read_types, read_nifti, read_dataset, check_gpu
from interactivenet.utils.mlflow import mlflow_get_runs

import torch
import pytorch_lightning as pl

import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

class PredictModule(pl.LightningModule):
    def __init__(
        self, 
        data:List[Dict[str, str]], 
        metadata:dict, 
        task:str,
        model:str,
        accelerator:Optional[str]="cuda",
        tta:bool=True
        ):
        super().__init__()
        if accelerator == "gpu":
            accelerator = "cuda"

        self._model = mlflow.pytorch.load_model(model, map_location=torch.device(accelerator))
        self.data = data
        self.metadata = metadata
        self.tta = tta
        self.post_numpy = EnsureType("numpy", device="cpu")
        self.original_size = OriginalSize(metadata["Fingerprint"]["Anisotropic"])
        self.labels = all([idx["label"] != "" for idx in self.data])
        self.raw = Path(os.environ["interactiveseg_raw"], task)

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        transforms = inference_transforms(metadata=self.metadata, labels=self.labels, raw_path=self.raw)

        self.predict_ds = Dataset(
            data=self.data, transform=transforms,
        )

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_ds, batch_size=1, shuffle=False,
            num_workers=4,
        )
        return predict_loader

    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        if self.tta:
            flip = TestTimeFlipping()
            ensembling = MeanEnsemble()

            image = flip(image)
            output = self.forward(image)
            
            flip.back = True
            output = flip(output)
            output = ensembling(output)
            output = [self.post_numpy(output)]
        else:
            output = self.forward(image)
            output = [self.post_numpy(i) for i in decollate_batch(output)]

        meta = [self.post_numpy(i) for i in decollate_batch(batch["interaction_meta_dict"])]
        output = [self.original_size(output, meta) for output, meta in zip(output, meta)]

        return output, meta, batch

def predict(
    data:List[Dict[str, str]], 
    metadata:dict, 
    task:str,
    accelerator:Optional[str],
    devices:Optional[str],
    results:Optional[Union[str, os.PathLike]],
    tta:bool=True,
    weights:bool=False,
    niftis:bool=False
    ):
    mlflow.set_tracking_uri(results)
    runs, experiment_id = mlflow_get_runs(task)

    all_outputs = []
    postprocessings = []
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

        network = PredictModule(data=data, metadata=metadata, task=task, model=model, accelerator=accelerator, tta=tta)

        # Required to log artifacts
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
        )

        with mlflow.start_run(experiment_id=experiment_id, tags={MLFLOW_PARENT_RUN_ID: run_id}, run_name="predict") as run:
            mlflow.set_tag('Mode', 'testing')
            mlflow.log_dict(postprocessing, "postprocessing.json")
            outputs = trainer.predict(model=network)

            if weights:
                save_weights(mlflow, outputs)

            if niftis:
                save_niftis(mlflow, outputs, postprocessing=postprocessing["postprocessing"])

            AnalyzeResults(mlflow=mlflow, outputs=outputs, postprocessing=postprocessing["postprocessing"], metadata=metadata, labels=network.labels)

            postprocessings.append(postprocessing["postprocessing"])
            all_outputs.append(outputs)

    return all_outputs, postprocessings, network.labels
            
def main():
    parser = argparse.ArgumentParser(
            description="Predict on the interactivenet network"
        )
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    parser.add_argument(
        "-a",
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do you want to use test time augmentations?"
    )
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
        help="Do you want to save weights as .npy in order to ensembling?"
    )

    args = parser.parse_args()
    raw = Path(os.environ["interactiveseg_raw"], args.task)
    exp = Path(os.environ["interactiveseg_processed"], args.task)
    results = Path(os.environ["interactiveseg_results"], "mlruns")

    data, modalities = read_dataset(raw, mode="test")
    metadata = read_metadata(exp / "plans.json")

    accelerator, devices, _ = check_gpu()

    predict(data=data, metadata=metadata, task=args.task, accelerator=accelerator, devices=devices, results=results, tta=args.tta, weights=args.weights, niftis=args.niftis)

if __name__=="__main__":
    main()