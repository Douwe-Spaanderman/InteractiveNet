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

from typing import List, Dict, Tuple, Union
from pathlib import Path
import math
import random
import json

import argparse

from statistics import median, stdev
import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, train_test_split
from interactivenet.utils.jsonencoders import NumpyEncoder
from interactivenet.utils.utils import read_dataset


class FingerPrint(object):
    def __init__(
        self,
        task: str,
        data: List[Dict[str, str]],
        modalities: Dict[str, str],
        relax_bbox: float = 0.1,
        seed: Union[int, None] = None,
        folds: int = 5,
        stratified: bool = True,
        leave_one_out: bool = False,
    ) -> None:
        print("Initializing Fingerprinting")
        self.task = task
        self.raw_path = Path(os.environ["interactivenet_raw"], task)
        self.processed_path = Path(os.environ["interactivenet_processed"], task)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        self.data = data
        self.relax_bbox = relax_bbox
        self.seed = seed
        self.folds = folds
        self.leave_one_out = leave_one_out
        self.modalities = modalities

        self.dim = []
        self.pixdim = []
        self.orientation = []
        self.anisotrophy = []
        self.names = []
        self.classes = []
        self.bbox = []
        self.clipping = []

        if self.modalities["0"].upper() == "CT":
            self.ct = True
            self.intensity_mean = []
            self.intensity_std = []
        else:
            self.ct = False
            self.intensity_mean = 0
            self.intensity_std = 0

    def __call__(self):
        print("Starting Fingerprinting: \n")
        print(f"Path: {self.raw_path}")

        for entry in self.data:
            image, label, interaction, subtype = (
                entry["image"],
                entry["label"],
                entry["interaction"],
                entry["class"],
            )

            name = Path(label).name.split(".")[0]
            print(f"File: {name}")

            image = nib.load(self.raw_path / image)
            label = nib.load(self.raw_path / label)
            inter = nib.load(self.raw_path / interaction)
            self.sanity_same_metadata(image, label, inter)

            self.dim.append(image.shape)
            spacing = image.header.get_zooms()
            self.pixdim.append(spacing)
            self.anisotrophy.append(self.check_anisotrophy(spacing))
            self.orientation.append(nib.orientations.aff2axcodes(image.affine))
            self.sanity_annotation_in_mask(label, inter)
            if self.ct:
                self.get_normalization_strategy(image, label)

            bbox = self.calculate_bbox(label)
            self.bbox.append(bbox[1] - bbox[0])
            self.classes.append(subtype)
            self.names.append(name)

        print("\nFingeprint:")
        print("- Database Structure: Correct")
        print(f"- All annotions in mask: {self.in_mask}")
        print(f"- All images anisotropic: {all(self.anisotrophy)}")

        if self.ct:
            self.clipping = [median(x) for x in zip(*self.clipping)]
            self.intensity_mean = median(self.intensity_mean)
            self.intensity_std = median(self.intensity_std)
            print("- CT: True")
            print(f"- Clipping to values: {self.clipping}")
            print(f"- Mean and stdev: {self.intensity_mean}, {self.intensity_std}")

        # Spacing
        self.target_spacing, self.resample_strategy = self.get_resampling_strategy(
            self.pixdim
        )
        self.spacing_ratios = [
            np.array(x) / np.array(self.target_spacing) for x in self.pixdim
        ]
        print(f"- Resampling strategy: {self.resample_strategy}")
        print(f"- Target spacing: {self.target_spacing}")

        # Size
        self.median_dim = self.calculate_median(self.dim)
        self.median_bbox = self.calculate_median(self.bbox)
        print(f"- Median shape: {self.median_dim}")
        print(f"- Bounding box extracted based on: Mask")
        print(f"- Median shape of bbox: {self.median_bbox}")

        # Resampled shape -
        self.resampled_shape = [
            self.calculate_new_shape(x, y)
            for x, y in zip(self.bbox, self.spacing_ratios)
        ]
        self.median_resampled_shape = self.calculate_median(self.resampled_shape)
        print(f"- Median shape of bbox after resampling: {self.median_resampled_shape}")

        # Experiment planning
        self.kernels, self.strides = self.get_kernels_strides(
            self.median_resampled_shape, self.target_spacing
        )
        self.deep_supervision, self.supervision_weights = self.get_supervision(
            self.strides
        )
        self.divisible_by = self.get_divisible(self.strides)
        print(f"- Network selection: {self.kernels} (kernels)")
        print(f"- Network selection: {self.strides} (strides)")

        # Get Final shape with right padding
        self.final_shape = [
            self.calculate_padded_shape(x, self.relax_bbox, self.divisible_by)
            for x in self.resampled_shape
        ]
        self.median_final_shape = self.calculate_median(self.final_shape)
        print(
            f"- Median shape of bbox after padding: {self.median_final_shape} (final shape)"
        )

        # Check orientations of images
        self.orientation_message = self.check_orientation(self.orientation)
        print(f"- {self.orientation_message}")

        print("\nCreating train-val splits:")
        self.splits = self.crossval()
        print(f"- using seed {self.seed}")
        print(f"- using {self.folds} folds")
        print(f"- using the following splits: {self.splits}")
        print("\n")
        self.save()

    def sanity_same_metadata(self, image, label, inter):
        def check(a, b, c, all_check=True):
            if all_check == True:
                return np.logical_and((a == b).all(), (b == c).all())
            else:
                return np.logical_and((a == b), (b == c))

        if not check(image.affine, label.affine, inter.affine) or not check(
            image.shape, label.shape, inter.shape, False
        ):
            raise AssertionError(
                "Metadata of image, mask and or annotation do not match"
            )

    def sanity_annotation_in_mask(self, label, inter):
        _check = True
        for inds_x, inds_y, inds_z in np.column_stack(
            (np.where(inter.get_fdata() > 0.5))
        ):
            if not label.dataobj[inds_x, inds_y, inds_z] == 1:
                _check = False
                warn.warning("Some annotations are not in the mask")

        self.in_mask = _check

    def check_anisotrophy(self, spacing: Tuple[int]):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing)

    def check_orientation(self, orientations: List[Tuple]):
        unique_orientations = list(set(orientations))
        if len(unique_orientations) == 1:
            orientation_message = (
                f"All images have the same orientation: {unique_orientations[0]}"
            )
        else:
            from collections import Counter

            unique_orientations = list(Counter(self.orientation).keys())
            orientation_message = f"Warning: Not all images have the same orientation, most are {unique_orientations[0]} but some also have {unique_orientations[1:]}\n  consider adjusting the orientation"

        return orientation_message

    def get_resampling_strategy(self, spacing: List[Tuple]):
        target_spacing = list(self.calculate_median(spacing))
        strategy = "Median"

        if self.anisotrophy.count(True) >= len(self.anisotrophy) / 2:
            index_max = np.argmax(target_spacing)
            target_spacing[index_max] = np.percentile(
                list(zip(*spacing))[index_max], 10
            )
            strategy = "Anisotropic"

        return tuple(target_spacing), strategy

    def get_normalization_strategy(self, image, label):
        points = np.where(label.get_fdata() > 0.5)
        points = image.get_fdata()[points]
        self.clipping.append([np.percentile(points, 0.5), np.percentile(points, 99.5)])
        self.intensity_mean.append(np.mean(points))
        self.intensity_std.append(np.std(points))

    def get_kernels_strides(self, sizes, spacings):
        strides, kernels = [], []
        while True:
            spacing_ratio = [sp / min(spacings) for sp in spacings]
            stride = [
                2 if ratio <= 2 and size >= 8 else 1
                for (ratio, size) in zip(spacing_ratio, sizes)
            ]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break

            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)

        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return kernels, strides

    def get_supervision(self, strides):
        deep_supervision = len(strides) - 3
        weights = np.array([0.5**i for i in range(deep_supervision + 1)])
        weights = weights / np.sum(weights)
        return deep_supervision, weights

    def get_divisible(self, strides):
        d = [1] * len(strides[0])
        for stride in strides:
            d = [d[axis] * stride[axis] for axis in range(len(stride))]

        return d

    def calculate_median(self, item: List[Tuple], std: bool = False):
        item = list(zip(*item))
        if std == True:
            return (
                median(item[0]),
                median(item[1]),
                median(item[2]),
                stdev(item[0]),
                stdev(item[1]),
                stdev(item[2]),
            )
        else:
            return (median(item[0]), median(item[1]), median(item[2]))

    def calculate_bbox(self, data, relaxation=None):
        inds_x, inds_y, inds_z = np.where(data.get_fdata() > 0.5)

        if not relaxation:
            relaxation = [0, 0, 0]

        bbox = np.array(
            [
                [
                    np.min(inds_x) - relaxation[0],
                    np.min(inds_y) - relaxation[1],
                    np.min(inds_z) - relaxation[2],
                ],
                [
                    np.max(inds_x) + relaxation[0],
                    np.max(inds_y) + relaxation[1],
                    np.max(inds_z) + relaxation[2],
                ],
            ]
        )

        # Remove below zero and higher than shape because of relaxation
        bbox[bbox < 0] = 0
        largest_dimension = [
            int(x) if x <= data.shape[i] else data.shape[i]
            for i, x in enumerate(bbox[1])
        ]
        bbox = np.array([bbox[0].tolist(), largest_dimension])

        return bbox

    def calculate_new_shape(self, shape, spacing_ratio):
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def calculate_padded_shape(self, shape, padding=0.1, divisible_by=None):
        new_shape = [x + math.ceil(x * padding) for x in shape]
        if divisible_by:
            if len(divisible_by) == 1:
                divisible_by = [divisible_by] * len(new_shape)

            for axis in range(len(new_shape)):
                residue = new_shape[axis] % divisible_by[axis]
                if residue != 0:
                    new_shape[axis] = new_shape[axis] + residue

        return new_shape

    def crossval(self):
        if not self.seed:
            self.seed = random.randint(0, 2**32 - 1)

        split = []
        if self.leave_one_out:
            print("- Using leave one out crossval")

            unique_subtypes = [*set(self.classes)]
            if len(unique_types) == 1:
                raise ValueError(
                    "Either no classes or 1 class is provided in the metadata.json -> Therefore leave one out subtype out experiment not possible"
                )

            print(f"{len(unique_subtypes)} classes detected: {unique_subtypes}")

            for subtype in enumerate(unique_subtypes):
                idx_keep = [idx for idx, x in enumerate(self.classes) if x != subtype]
                names_keep = [self.names[idx] for idx in idx_keep]
                class_keep = [self.classes[idx] for idx in idx_keep]
                train, test = train_test_split(
                    names_keep,
                    test_size=1 / unique_subtypes,
                    stratify=class_keep,
                    shuffle=True,
                )
                split.append({"train": train, "val": test, "left out": subtype})

            values = [x for xs in metadata.values() for x in xs]
            if not all([name in values for name in self.names]):
                raise KeyError(
                    "names from metadata don't match names in fingerprinting"
                )

            self.folds = len(unique_subtypes)
        else:
            kf = StratifiedKFold(
                n_splits=self.folds, random_state=self.seed, shuffle=True
            )
            for train_index, val_index in kf.split(self.names, self.classes):
                split.append(
                    {
                        "train": [self.names[i] for i in train_index],
                        "val": [self.names[i] for i in val_index],
                    }
                )

        return split

    def save(self):
        d = {
            "Fingerprint": {
                "In mask": self.in_mask,
                "Anisotropic": self.anisotrophy.count(True)
                >= len(self.anisotrophy) / 2,
                "CT": self.ct,
                "Clipping": self.clipping,
                "Intensity_mean": self.intensity_mean,
                "Intensity_std": self.intensity_std,
                "Resampling": self.resample_strategy,
                "Target spacing": self.target_spacing,
                "Median size": self.median_dim,
                "Median size bbox": self.median_bbox,
                "Median size resampled": self.median_resampled_shape,
                "Median final shape": self.median_final_shape,
            },
            "Plans": {
                "kernels": self.kernels,
                "strides": self.strides,
                "deep supervision": self.deep_supervision,
                "deep supervision weights": self.supervision_weights,
                "padding": self.relax_bbox,
                "divisible by": self.divisible_by,
                "seed": self.seed,
                "number of folds": self.folds,
                "splits": self.splits,
            },
            "Cases": [
                {
                    "name": self.names[idx],
                    "dimensions": self.dim[idx],
                    "pixdims": self.pixdim[idx],
                    "orientations": self.orientation[idx],
                    "bbox": self.bbox[idx],
                    "resampled shape": self.resampled_shape[idx],
                    "final shape": self.final_shape[idx],
                }
                for idx in range(len(self.names))
            ],
        }

        with open(self.processed_path / "plans.json", "w") as jfile:
            json.dump(d, jfile, indent=4, sort_keys=True, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(description="InteractiveNet Fingerprinting")
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


if __name__ == "__main__":
    main()
