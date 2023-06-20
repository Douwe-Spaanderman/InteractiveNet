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

import os
import argparse
from pathlib import Path

from interactivenet.utils.utils import read_dataset
from interactivenet.experiment_planning.fingerprinting import FingerPrint
from interactivenet.experiment_planning.preprocessing import Preprocessing


def main():
    parser = argparse.ArgumentParser(
        description="InteractiveNet fingerprinting, experiment planning and procesing"
    )
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    parser.add_argument(
        "-f",
        "--cross_validation_folds",
        nargs="?",
        default=5,
        type=int,
        help="How many folds do you want to use for cross-validation?",
    )
    parser.add_argument(
        "-s",
        "--seed",
        nargs="?",
        default=None,
        help="Do you want to specify the seed for the cross-validation split?",
    )
    parser.add_argument(
        "-sf",
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do you want to stratify the cross-validation split on class?",
    )
    parser.add_argument(
        "-o",
        "--leave_one_out",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to conduct a leave-one-type-out experiment?",
    )
    parser.add_argument(
        "-r",
        "--relax_bbox",
        nargs="?",
        default=0.1,
        type=float,
        help="By how much do you want to relax the bounding box?",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to run verbose and generate images?",
    )
    args = parser.parse_args()

    seed = args.seed
    if args.seed:
        seed = int(seed)

    raw_path = Path(os.environ["interactivenet_raw"], args.task)
    data, modalities = read_dataset(raw_path)

    fingerprint = FingerPrint(
        task=args.task,
        data=data,
        modalities=modalities,
        relax_bbox=args.relax_bbox,
        seed=seed,
        folds=args.cross_validation_folds,
        stratified=args.stratified,
        leave_one_out=args.leave_one_out,
    )
    fingerprint()

    preprocess = Preprocessing(
        task=args.task,
        data=data,
        target_spacing=fingerprint.target_spacing,
        relax_bbox=fingerprint.relax_bbox,
        divisble_using=fingerprint.divisible_by,
        clipping=fingerprint.clipping,
        intensity_mean=fingerprint.intensity_mean,
        intensity_std=fingerprint.intensity_std,
        ct=fingerprint.ct,
        verbose=args.verbose,
    )
    preprocess()


if __name__ == "__main__":
    main()
