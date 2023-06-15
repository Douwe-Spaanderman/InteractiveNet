#   Copyright 2023 Biomedical Imaging Group Rotterdam, Departments of
#   Radiology and Nuclear Medicine, Erasmus MC, Rotterdam, The Netherlands
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#   
#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from pathlib import Path
import argparse
import os
from typing import List, Dict, Optional, Union
import numpy as np
import torch
from collections import Counter


from monai.transforms import (
    AsDiscrete,
    EnsureType,
    MeanEnsemble,
)

from interactivenet.transforms.transforms import (
    OriginalSize,
    TestTimeFlipping,
)

from monai.data import Dataset, DataLoader, decollate_batch

import nibabel as nib
from interactivenet.utils.results import AnalyzeResults
from interactivenet.transforms.set_transforms import inference_transforms
from interactivenet.utils.utils import (
    save_weights,
    save_niftis,
    read_metadata,
    check_gpu,
    read_data_inference,
    to_array,
)
from interactivenet.utils.mlflow import mlflow_get_runs, mlflow_get_id
from interactivenet.utils.postprocessing import ApplyPostprocessing

import torch
import pytorch_lightning as pl

import mlflow.pytorch


class InferenceModule(pl.LightningModule):
    def __init__(
        self,
        data: List[Dict[str, str]],
        metadata: dict,
        models: List[str],
        accelerator: Optional[str] = "cuda",
        tta: bool = True,
    ):
        super().__init__()
        if accelerator == "gpu":
            accelerator = "cuda"

        self._model = torch.nn.ModuleList(
            [
                mlflow.pytorch.load_model(model, map_location=torch.device(accelerator))
                for model in models
            ]
        )
        self.data = data
        self.metadata = metadata
        self.tta = tta
        self.post_numpy = EnsureType("numpy", device="cpu")
        self.original_size = OriginalSize(metadata["Fingerprint"]["Anisotropic"])
        self.labels = all([idx["label"] != "" for idx in self.data])

    def forward(self, x, model):
        return model(x)

    def prepare_data(self):
        transforms = inference_transforms(metadata=self.metadata, labels=self.labels)

        self.predict_ds = Dataset(
            data=self.data,
            transform=transforms,
        )

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_ds,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )
        return predict_loader

    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        outputs = []
        ensembling = MeanEnsemble()
        for model in self._model:
            if self.tta:
                flip = TestTimeFlipping()

                image = flip(image)
                output = self.forward(image, model)

                flip.back = True
                output = flip(output)
                outputs.append(ensembling(output))
            else:
                outputs.append(self.forward(image, model))

        outputs = torch.stack(outputs)
        outputs = ensembling(outputs)

        if self.tta:
            outputs = [self.post_numpy(outputs)]
        else:
            outputs = [self.post_numpy(i) for i in decollate_batch(outputs)]

        meta = [
            self.post_numpy(i) for i in decollate_batch(batch["interaction_meta_dict"])
        ]
        outputs = [
            self.original_size(output, meta) for output, meta in zip(outputs, meta)
        ]

        return outputs, meta, batch


def infer(
    data: List[Dict[str, str]],
    save: str,
    task: str,
    results: Optional[Union[str, os.PathLike]],
    accelerator: Optional[str],
    devices: Optional[str],
    tta: bool = True,
    weights: bool = False,
    log_mlflow: bool = False,
):
    # Get model
    if accelerator == "gpu":
        accelerator = "cuda"

    mlflow.set_tracking_uri(results / "mlruns")
    runs, experiment_id = mlflow_get_runs(task)

    # Check if models in results / models
    deployed_model = results / "models" / task
    if deployed_model.is_dir():
        print("Using deployed/pretrained model")

        metadata = read_metadata(deployed_model / "plans.json")

        postprocessing = read_metadata(
            deployed_model / "postprocessings.json",
            error_message="no postprocessing file found, this is a weird error message for a deployed model...",
        )
        postprocessings = [x["postprocessing"] for x in postprocessing.values()]

        models = [x for x in deployed_model.glob("model/*") if x.is_dir()]
    else:
        print("Using self-trained model")
        exp = Path(os.environ["interactivenet_processed"], task)
        metadata = read_metadata(exp / "plans.json")

        models = []
        postprocessings = []
        for idx, run in runs.iterrows():
            if run["tags.Mode"] != "training":
                continue

            run_id = run["run_id"]
            run["params.fold"]
            postprocessing = Path(
                run["artifact_uri"].split("file://")[-1], "postprocessing.json"
            )
            postprocessing = read_metadata(
                postprocessing,
                error_message="postprocessing hasn't been run yet, please do this before predictions",
            )
            if postprocessing["using_checkpoint"]:
                models.append("runs:/" + run_id + "/model_checkpoint")
            else:
                models.append("runs:/" + run_id + "/model")

            postprocessings.append(postprocessing["postprocessing"])

    postprocessing = Counter(postprocessings).most_common()[0][0]

    network = InferenceModule(
        data=data, metadata=metadata, models=models, accelerator=accelerator, tta=tta
    )

    # Required to log artifacts
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
    )

    outputs = trainer.predict(model=network)

    # Here writing of the outputs is done to either MLflow or simply in output folder
    print("writing outputs")
    argmax = AsDiscrete(argmax=True)
    save = Path(save)
    save.mkdir(parents=True, exist_ok=True)
    labels = isinstance(outputs[0][2]["label"][0], np.ndarray) or isinstance(
        outputs[0][2]["label"][0], torch.Tensor
    )
    if not log_mlflow:
        if labels:
            print(
                "We found labels, but logging in mlflow was false, so we are not measuring overlap, please adjust mlflow argument if you wish to calculate overlap"
            )

        for output in outputs:
            name = Path(output[1][0]["filename_or_obj"]).name.split(".")[0]
            pred = output[0][0]
            meta = output[1][0]

            pred = argmax(pred)
            pred = ApplyPostprocessing(pred, postprocessing)
            pred = pred[0]  # Get out of channel

            data_file = save / f"{name}.nii.gz"

            segmentation = nib.Nifti1Image(to_array(pred), to_array(meta["affine"]))
            nib.save(segmentation, str(data_file))
    else:
        experiment_id = mlflow_get_id("Inference")
        with mlflow.start_run(experiment_id=experiment_id, run_name=save.name) as run:
            save_niftis(mlflow, outputs, postprocessing=postprocessing)

            if weights:
                save_weights(mlflow, outputs)

            if labels:
                AnalyzeResults(
                    mlflow=mlflow,
                    outputs=outputs,
                    postprocessing=postprocessing,
                    metadata=metadata,
                    labels=network.labels,
                )


def main():
    parser = argparse.ArgumentParser(
        description="Inference on the interactivenet network"
    )
    parser.add_argument(
        "-t",
        "--task",
        required=True,
        type=str,
        help="Task name, defines which model and preprocessing is used (REQUIRED)",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to input images for inference (REQUIRED)",
    )
    parser.add_argument(
        "-in",
        "--interactions",
        required=True,
        type=str,
        help="Path to interactions for inference (REQUIRED)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to save predictions (REQUIRED)",
    )
    parser.add_argument(
        "-a",
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do you want to use test time augmentations?",
    )
    parser.add_argument(
        "-w",
        "--weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save raw weights as .npy?",
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="?",
        default=None,
        type=str,
        help="If you have ground truth segmentation, provide Path to folder here",
    )
    parser.add_argument(
        "-m",
        "--log_mlflow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want MLflow instead to log the results (This has extra features, when labels are also provided)?",
    )

    args = parser.parse_args()
    Path(os.environ["interactiveseg_raw"], args.task)
    results = Path(os.environ["interactiveseg_results"])

    accelerator, devices, _ = check_gpu()

    data = read_data_inference(
        images=args.input, interactions=args.interactions, labels=args.labels
    )

    infer(
        data=data,
        task=args.task,
        results=results,
        accelerator=accelerator,
        devices=devices,
        tta=args.tta,
        save=args.output,
        weights=args.weights,
        log_mlflow=args.log_mlflow,
    )


if __name__ == "__main__":
    main()
