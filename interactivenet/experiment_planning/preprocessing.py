from typing import List, Tuple, Dict, Union
from pathlib import Path
import pickle

import numpy as np
import os
import argparse

from monai.data import Dataset as MonaiDataset

from interactivenet.utils.utils import read_dataset, read_metadata
from interactivenet.transforms.set_transforms import processing_transforms


class Preprocessing(MonaiDataset):
    def __init__(
        self,
        task: str,
        data: List[Dict[str, str]],
        target_spacing: Tuple[float],
        relax_bbox: Union[float, Tuple[float]] = 0.1,
        divisble_using: Union[int, Tuple[int]] = (16, 16, 8),
        clipping: List[float] = [],
        intensity_mean: float = 0,
        intensity_std: float = 0,
        ct: bool = False,
        verbose: bool = False,
    ) -> None:
        print("Initializing Preprocessing")
        self.task = task
        self.raw_path = Path(os.environ["interactivenet_raw"], task)
        self.processed_path = Path(os.environ["interactivenet_processed"], task)
        self.create_directories()

        self.data = data
        self.target_spacing = target_spacing
        self.relax_bbox = relax_bbox
        self.divisble_using = divisble_using
        self.clipping = clipping
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std
        self.ct = ct
        self.verbose = verbose
        self.transforms = processing_transforms(
            target_spacing=self.target_spacing,
            processed_path=self.processed_path,
            raw_path=self.raw_path,
            relax_bbox=self.relax_bbox,
            divisble_using=self.divisble_using,
            clipping=self.clipping,
            intensity_mean=self.intensity_mean,
            intensity_std=self.intensity_std,
            ct=self.ct,
            verbose=self.verbose,
        )

        super().__init__(self.data, self.transforms)

    def __call__(self) -> None:
        print("\nPreprocessing:\n")
        metainfo = {}
        for i, item in enumerate(self.data):
            name = Path(item["label"]).with_suffix("").stem
            print(f"File: {name}")
            import ipdb;
            ipdb.set_trace()
            item = self.__getitem__(i)
            metainfo[name] = self.create_metainfo(item)
            print("")

        with open(self.processed_path / "metadata.pkl", "wb") as handle:
            pickle.dump(metainfo, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_directories(self) -> None:
        self.input_folder = self.processed_path / "network_input"
        self.input_folder.mkdir(parents=True, exist_ok=True)

    def create_metainfo(self, item: Dict[str, Union[np.ndarray, dict]]) -> dict:
        metadata = item["image_meta_dict"]
        return {
            "filename_or_obj": metadata["filename_or_obj"],
            "org_size": metadata["org_dim"],
            "org_spacing": metadata["org_spacing"],
            "new_size": metadata["new_dim"],
            "new_spacing": metadata["new_spacing"],
            "ratio_size": metadata["new_dim"] / metadata["org_dim"],
            "ratio_spacing": metadata["new_spacing"] / metadata["org_spacing"],
            "resample_flag": metadata["resample_flag"],
            "anisotrophy_flag": metadata["anisotrophy_flag"],
            "bbox_location": metadata["bbox"],
            "bbox_size": metadata["bbox_shape"],
            "bbox_relaxation": metadata["bbox_relaxation"],
            "bbox_ratio": metadata["bbox_shape"] / metadata["new_dim"],
            "final_size": metadata["final_bbox"],
            "final_size": metadata["final_bbox_shape"],
        }


def main():
    parser = argparse.ArgumentParser(description="InteractiveNet Processing")
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do you want to run verbose and generate images?",
    )
    args = parser.parse_args()

    raw_path = Path(os.environ["interactivenet_raw"], args.task)
    data, modalities = read_dataset(raw_path)

    plans = Path(os.environ["interactivenet_processed"], args.task, "plans.json")
    plans = read_metadata(
        plans, error_message="Please run fingerprinting before processing data."
    )

    preprocess = Preprocessing(
        task=args.task,
        data=data,
        target_spacing=plans["Fingerprint"]["Target spacing"],
        relax_bbox=plans["Plans"]["padding"],
        divisble_using=plans["Plans"]["divisible by"],
        clipping=plans["Fingerprint"]["Clipping"],
        intensity_mean=plans["Fingerprint"]["Intensity_mean"],
        intensity_std=plans["Fingerprint"]["Intensity_std"],
        ct=plans["Fingerprint"]["CT"],
        verbose=args.verbose,
    )
    preprocess()

if __name__ == "__main__":
    main()