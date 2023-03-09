from typing import List, Tuple, Dict, Sequence, Optional, Callable, Union
from pathlib import Path, PosixPath
import json
import pickle

import numpy as np
import os
import argparse

from monai.data import Dataset as MonaiDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    DivisiblePadd
)

from interactivenet.transforms.transforms import Resamplingd, EGDMapd, BoudingBoxd, Visualized, NormalizeValuesd
from interactivenet.utils.utils import read_dataset

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
    ) -> None:
        print("Initializing Preprocessing")
        self.task = task
        self.raw_path = Path(os.environ["interactiveseg_raw"], task)
        self.processed_path = Path(os.environ["interactiveseg_processed"], task)
        self.create_directories()
        
        self.data = data
        self.relax_bbox = relax_bbox
        self.divisble_using = divisble_using
        self.clipping = clipping
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std
        self.ct = ct
        self.transforms = Compose(
            [
                # ADD PATH
                LoadImaged(keys=["image", "interaction", "label"]),
                EnsureChannelFirstd(keys=["image", "interaction", "label"]),
                Visualized(
                    keys=["image", "interaction", "label"],
                    save=self.processed_path / 'verbose' / 'raw',
                    interaction=True
                ),
                Resamplingd(
                    keys=["image", "interaction", "label"],
                    pixdim=target_spacing,
                ),
                BoudingBoxd(
                    keys=["image", "interaction", "label"],
                    on="label",
                    relaxation=self.relax_bbox,
                    divisiblepadd=self.divisble_using,
                ),
                NormalizeValuesd(
                    keys=["image"],
                    clipping=self.clipping,
                    mean=self.intensity_mean,
                    std=self.intensity_std,
                ),
                Visualized(
                    keys=["image", "interaction", "label"],
                    save=self.processed_path / 'verbose' / 'processed',
                    interaction=True
                ),
                EGDMapd(
                    keys=["interaction"],
                    image="image",
                    lamb=1,
                    iter=4,
                    logscale=True,
                    ct=self.ct
                ),
                DivisiblePadd(
                    keys=["image", "interaction", "label"],
                    k=self.divisble_using
                ),
                Visualized(
                    keys=["interaction", "label"],
                    save=self.processed_path / 'verbose' / 'Map',
                    distancemap=True,
                ),
                ]
        )
        super().__init__(self.data, self.transforms)

    def __call__(self) -> None:
        print("\nPreprocessing:\n")
        metainfo = {}
        for i, item in enumerate(self.data):
            name = item["label"].with_suffix('').stem
            print(f"File: {name}")
            item = self.__getitem__(i)

            metainfo[name] = self.create_metainfo(item)

            self.save_sample(item, name)
            print("")

        with open(self.processed_path / "metadata.pkl", 'wb') as handle:
            pickle.dump(metainfo, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_directories(self) -> None:
        self.input_folder = self.processed_path / "network_input"
        self.input_folder.mkdir(parents=True, exist_ok=True)

    def create_metainfo(self, item:Dict[str, Union[np.ndarray, dict]]) -> dict:
        metadata = item["image_meta_dict"]
        return {
            "filename_or_obj" : metadata["filename_or_obj"],
            "org_size" : metadata["org_dim"],
            "org_spacing" : metadata["org_spacing"],
            "new_size" : metadata["new_dim"],
            "new_spacing" : metadata["new_spacing"],
            "ratio_size": metadata["new_dim"] / metadata["org_dim"],
            "ratio_spacing": metadata["new_spacing"] / metadata["org_spacing"],
            "resample_flag" : metadata["resample_flag"],
            "anisotrophy_flag" : metadata["anisotrophy_flag"],
            "bbox_location" : metadata["bbox"],
            "bbox_size" : metadata["bbox_shape"],
            "bbox_relaxation" : metadata["bbox_relaxation"],
            "bbox_ratio": metadata["bbox_shape"] / metadata["new_dim"],
            "final_size" : metadata["final_bbox"],
            "final_size" : metadata["final_bbox_shape"],
        }

    def save_sample(self, item:Dict[str, Union[np.ndarray, dict]], name:str) -> None:
        keys = list(item.keys())
        objects = [False if x.endswith("_meta_dict") or x.endswith("_transforms") else True for x in keys]
        objects = sum(objects)

        np.savez(self.input_folder / name, **{key : item[key] for key in keys[:objects]})

        pickle_data = {
            key : item[key] for key in keys[objects:]
        }

        with open(self.input_folder.parent / "network_input" / (name + ".pkl"), 'wb') as handle:
            pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(description="InteractiveNet Processing")
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    args = parser.parse_args()

    raw_path = Path(os.environ["interactiveseg_raw"], args.task)
    data, modalities = read_dataset(raw_path)

    plans = Path(os.environ["interactiveseg_processed"], args.task, "plans.json")
    if not plans.is_file():
        raise KeyError("Please run fingerprinting before processing data.")
    
    with open(plans) as f:
        plans = json.load(f)

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
    )
    preprocess()


if __name__=="__main__":

    main()