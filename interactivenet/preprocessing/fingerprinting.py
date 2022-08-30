from typing import List, Tuple, Union
from pathlib import Path, PosixPath
import math
import random
import json

from statistics import median, stdev
import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold
from interactivenet.utils.jsonencoders import NumpyEncoder

class FingerPrint(object):
    def __init__(
        self,
        task: str,
        relax_bbox:float=0.1,
        seed:Union[int, None]=None,
        folds:int=5,
        ct:bool=False,
        leave_one_out:bool=False,
    ) -> None:
        print("Initializing Fingerprinting")
        self.task = task
        self.get_files()

        self.relax_bbox = relax_bbox
        self.seed = seed
        self.folds = folds
        self.ct = ct
        self.leave_one_out = leave_one_out
        self.sanity_files()

        self.dim = []
        self.pixdim = []
        self.orientation = []
        self.anisotrophy = []
        self.names = []
        self.bbox = []
        self.clipping = []
        if self.ct:
            self.intensity_mean = []
            self.intensity_std = []
        else:
            self.intensity_mean = 0
            self.intensity_std = 0

    def __call__(self):
        print("Starting Fingerprinting: \n")
        print(f"Path: {self.images[0].parents[1]}")
        for img_path, mask_path, annot_path in zip(self.images, self.masks, self.annotations):
            name = mask_path.with_suffix('').stem
            print(
                f"File: {name}"
            )

            img = nib.load(img_path)
            mask = nib.load(mask_path)
            annot = nib.load(annot_path)
            self.sanity_same_metadata(img, mask, annot)

            self.dim.append(img.shape)
            spacing = img.header.get_zooms()
            self.pixdim.append(spacing)
            self.anisotrophy.append(self.check_anisotrophy(spacing))
            self.orientation.append(nib.orientations.aff2axcodes(img.affine))
            self.sanity_annotation_in_mask(mask, annot)
            self.get_normalization_strategy(img, mask, self.ct)

            bbox = self.calculate_bbox(mask)
            self.bbox.append(bbox[1] - bbox[0])
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
        self.target_spacing, self.resample_strategy = self.get_resampling_strategy(self.pixdim)
        self.spacing_ratios = [np.array(x) / np.array(self.target_spacing) for x in self.pixdim]
        print(f"- Resampling strategy: {self.resample_strategy}")
        print(f"- Target spacing: {self.target_spacing}")
        
        # Size
        self.median_dim = self.calculate_median(self.dim)
        self.median_bbox = self.calculate_median(self.bbox)
        print(f"- Median shape: {self.median_dim}")
        print(f"- Bounding box extracted based on: Mask")
        print(f"- Median shape of bbox: {self.median_bbox}")

        # Resampled shape -
        self.resampled_shape = [self.calculate_new_shape(x, y) for x, y in zip(self.bbox, self.spacing_ratios)]
        self.median_resampled_shape = self.calculate_median(self.resampled_shape)
        print(f"- Median shape of bbox after resampling: {self.median_resampled_shape}")

        # Experiment planning 
        self.kernels, self.strides = self.get_kernels_strides(self.median_resampled_shape, self.target_spacing)
        self.divisible_by = self.get_divisible(self.strides)
        print(f"- Network selection: {self.kernels} (kernels)")
        print(f"- Network selection: {self.strides} (strides)")

        # Get Final shape with right padding
        self.final_shape = [self.calculate_padded_shape(x, self.relax_bbox, self.divisible_by) for x in self.resampled_shape]
        self.median_final_shape = self.calculate_median(self.final_shape)
        print(f"- Median shape of bbox after padding: {self.median_final_shape} (final shape)")

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

    def get_files(self):
        self.exp = Path(os.environ["interactiveseg_raw"], self.task)
    
        self.images = sorted([x for x in (self.exp / "imagesTr").glob('**/*') if x.is_file()])
        self.masks = sorted([x for x in (self.exp / "labelsTr").glob('**/*') if x.is_file()])
        self.annotations = sorted([x for x in (self.exp / "interactionTr").glob('**/*') if x.is_file()])
        
        self.save_location = Path(os.environ["interactiveseg_processed"], self.task)
        self.save_location.mkdir(parents=True, exist_ok=True)

    def sanity_files(self):
        def check(a,b,c):
            return a == b == c

        len_mask = len(self.masks)
        if not len(self.images) % len_mask == len(self.annotations) % len_mask == 0:
            raise AssertionError("Length of database is not correct, e.g. more masks or annot than images")

        images_names = list(set(['_'.join(x.name.split("_")[:-1]) for x in self.images]))
        masks_names = list(set([x.with_suffix('').stem for x in self.masks]))
        annotations_names = list(set([x.with_suffix('').stem for x in self.annotations]))
        if all([check(a,b,c) for a,b,c in zip(images_names, masks_names, annotations_names)]) == False:
            raise AssertionError("images, masks and annotations do not have the correct names or are not ordered")

    def sanity_same_metadata(self, img, mask, annot):
        def check(a,b,c,all_check=True):
            if all_check == True:
                return np.logical_and((a==b).all(), (b==c).all())
            else:
                return np.logical_and((a==b), (b==c))
            
        if not check(img.affine, mask.affine, annot.affine) or not check(img.shape, mask.shape, annot.shape, False):
            raise AssertionError("Metadata of image, mask and or annotation do not match")

    def sanity_annotation_in_mask(self, mask, annot):
        _check = True
        for inds_x, inds_y, inds_z in np.column_stack((np.where(annot.get_fdata() > 0.5))):
            if not mask.dataobj[inds_x, inds_y, inds_z] == 1:
                _check = False
                warn.warning("Some annotations are not in the mask")

        self.in_mask = _check

    def check_anisotrophy(self, spacing:Tuple[int]):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing)

    def check_orientation(self, orientations:List[Tuple]):
        unique_orientations = list(set(orientations))
        if len(unique_orientations) == 1:
            orientation_message = f"All images have the same orientation: {unique_orientations[0]}"
        else:
            from collections import Counter
            unique_orientations = list(Counter(self.orientation).keys())
            orientation_message = f"Warning: Not all images have the same orientation, most are {unique_orientations[0]} but some also have {unique_orientations[1:]}\n  consider adjusting the orientation"

        return orientation_message

    def get_resampling_strategy(self, spacing:List[Tuple]):
        target_spacing = list(self.calculate_median(spacing))
        strategy = "Median"

        if self.anisotrophy.count(True) >= len(self.anisotrophy) / 2:
            index_max = np.argmax(target_spacing)
            target_spacing[index_max] = np.percentile(list(zip(*spacing))[index_max],10)
            strategy = "Anisotropic"

        return tuple(target_spacing), strategy

    def get_normalization_strategy(self, img, mask, ct:bool=False):
        if ct:
            points = np.where(mask.get_fdata() > 0.5)
            points = img.get_fdata()[points]
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
    
    def get_divisible(self, strides):
        d = [1] * len(strides[0])
        for stride in strides:
            d = [d[axis]*stride[axis] for axis in range(len(stride))]
            
        return d

    def calculate_median(self, item:List[Tuple], std:bool=False):
        item = list(zip(*item))
        if std == True:
            return ((median(item[0]), median(item[1]), median(item[2]), stdev(item[0]), stdev(item[1]), stdev(item[2])))
        else:
            return (median(item[0]), median(item[1]), median(item[2]))

    def calculate_bbox(self, data, relaxation=None):
        inds_x, inds_y, inds_z = np.where(data.get_fdata() > 0.5)

        if not relaxation:
            relaxation = [0, 0, 0]

        bbox = np.array([
            [
                np.min(inds_x) - relaxation[0],
                np.min(inds_y) - relaxation[1],
                np.min(inds_z) - relaxation[2]
                ],
            [
                np.max(inds_x) + relaxation[0],
                np.max(inds_y) + relaxation[1],
                np.max(inds_z) + relaxation[2]
            ]
        ])

        # Remove below zero and higher than shape because of relaxation
        bbox[bbox < 0] = 0
        largest_dimension = [int(x) if  x <= data.shape[i] else data.shape[i] for i, x in enumerate(bbox[1])]
        bbox = np.array([bbox[0].tolist(), largest_dimension])

        return bbox

    def calculate_new_shape(self, shape, spacing_ratio):
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def calculate_padded_shape(self, shape, padding=0.1, divisible_by=None):
        new_shape = [x + math.ceil(x*padding) for x in shape]
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
            metadata = self.exp / "types.json"
            if metadata.is_file():
                with open(metadata) as f:
                    metadata = json.load(f)
            else:
                raise KeyError("leave one out is True, but there is no types.json")

            keys = list(metadata.keys())
            for idx, key in enumerate(keys):
                train_keys = keys[:idx] + keys[idx+1 :]
                train_list = [metadata[k] for k in train_keys]
                train_list = [x for xs in train_list for x in xs]
                split.append({"train":train_list,"val":metadata[key], "left out":key})

            values = [x for xs in metadata.values() for x in xs]
            if not all([name in values for name in self.names]):
                raise KeyError("names from metadata don't match names in fingerprinting")

            self.folds = len(metadata.keys())
        else:
            kf = KFold(n_splits=self.folds, random_state=self.seed, shuffle=True)
            for train_index, val_index in kf.split(self.names):
                split.append({"train":[self.names[i] for i in train_index],"val":[self.names[i] for i in val_index]})

        return split

    def save(self):
        d = {
            "Fingerprint" : {
                "In mask": self.in_mask,
                "Anisotropic": self.anisotrophy.count(True) >= len(self.anisotrophy) / 2,
                "CT": self.ct,
                "Clipping" : self.clipping,
                "Intensity_mean" : self.intensity_mean,
                "Intensity_std" : self.intensity_std,
                "Resampling": self.resample_strategy,
                "Target spacing": self.target_spacing,
                "Median size": self.median_dim,
                "Median size bbox": self.median_bbox,
                "Median size resampled": self.median_resampled_shape,
                "Median final shape": self.median_final_shape,
            },
            "Plans": {
                "kernels": self.kernels,
                "strides" : self.strides,
                "padding": self.relax_bbox,
                "divisible by": self.divisible_by,
                "seed": self.seed,
                "number of folds": self.folds,
                "splits":self.splits,
            },
            "Cases" : [{
                "name": self.names[idx],
                "dimensions": self.dim[idx],
                "pixdims": self.pixdim[idx],
                "orientations": self.orientation[idx],
                "bbox":self.bbox[idx],
                "resampled shape":self.resampled_shape[idx],
                "final shape":self.final_shape[idx]
            } for idx in range(len(self.names))]
        }

        with open(self.save_location / "plans.json", "w") as jfile:
            json.dump(d, jfile, indent=4, sort_keys=True, cls=NumpyEncoder)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
             description="Preprocessing of "
         )
    parser.add_argument(
         "-t",
         "--task",
         nargs="?",
         default="Task710_STTMRI",
         help="Task name"
    )
    parser.add_argument(
        "-o",
        "--leave_one_out",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to do leave one out experiments?"
    )
    parser.add_argument(
        "-c",
        "--CT",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="is the data CT?"
    )
    args = parser.parse_args()

    fingerpint = FingerPrint(args.task, ct=args.CT, leave_one_out=args.leave_one_out)
    fingerpint()
