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

import argparse

import os

from pathlib import Path

from interactivenet.utils.utils import (
    read_metadata,
    read_dataset,
    check_gpu,
    read_processed,
)
from interactivenet.training.postprocessing import run_postprocessing
from interactivenet.test.predict import predict
from interactivenet.test.ensemble import ensemble


def main():
    parser = argparse.ArgumentParser(description="Run the prediction pipeline")
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    parser.add_argument(
        "-a",
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do you want to use test time augmentations?",
    )
    parser.add_argument(
        "-p",
        "--postprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do you want to check for postprocessing?",
    )
    parser.add_argument(
        "-n",
        "--niftis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save the output as nifti?",
    )
    parser.add_argument(
        "-w",
        "--weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save weights as .npy?",
    )
    parser.add_argument(
        "-i",
        "--intermediates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to save intermediate niftis and weights (before ensembling)?",
    )

    args = parser.parse_args()
    raw = Path(os.environ["interactivenet_raw"], args.task)
    exp = Path(os.environ["interactivenet_processed"], args.task)
    results = Path(os.environ["interactivenet_results"], "mlruns")

    accelerator, devices, _ = check_gpu()

    data, modalities = read_dataset(raw, mode="test")
    metadata = read_metadata(exp / "plans.json")

    if args.postprocessing:
        print("Running Postprocessing on validation set")
        run_postprocessing(
            data=read_processed(exp),
            metadata=metadata,
            task=args.task,
            accelerator=accelerator,
            devices=devices,
            results=results,
        )
        print("Postprocessing done!")

    outputs, postprocessing, labels = predict(
        data=data,
        metadata=metadata,
        task=args.task,
        accelerator=accelerator,
        devices=devices,
        results=results,
        tta=args.tta,
        weights=args.intermediates,
        niftis=args.intermediates,
    )
    ensemble(
        outputs=outputs,
        metadata=metadata,
        task=args.task,
        results=results,
        postprocessing=postprocessing,
        weights=args.weights,
        niftis=args.niftis,
        labels=labels,
    )


if __name__ == "__main__":
    main()
