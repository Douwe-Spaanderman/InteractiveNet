from typing import List, Tuple, Sequence, Optional, Callable
from pathlib import PosixPath

import numpy as np
import os

from monai.data import Dataset as _MonaiDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    DivisiblePadd,
)

from interactivenet.transforms.transforms import Resamplingd, EGDMapd, BoudingBoxd

class Preprocessing(_MonaiDataset):
    def __init__(
        self, 
        images: List[PosixPath],
        masks: List[PosixPath],
        annotations: List[PosixPath],
        median_shape: Tuple[float],
        target_spacing: Tuple[float], 
        task: int,
    ) -> None:

        data =[
            {"image": img_path, "mask": mask_path, "annotation": annot_path}
            for img_path, mask_path, annot_path in zip(images, masks, annotations)
        ]

        transforms = Compose(
            [
                LoadImaged(keys=["image", "annotation", "mask"]),
                EnsureChannelFirstd(keys=["image", "annotation", "mask"]),
                Resamplingd(
                    keys=["image", "annotation", "mask"],
                    pixdim=target_spacing,
                ),
                BoudingBoxd(
                    keys=["image", "annotation", "mask"],
                    on="annotation",
                    relaxation=(10, 10, 2),
                ),
                NormalizeIntensityd(
                    keys=["image"],
                    nonzero=True,
                    channel_wise=False,
                ),
                EGDMapd(
                    keys=["annotation"],
                    image="image",
                    lamb=1,
                    iter=4,
                ),
                DivisiblePadd(
                    keys=["image", "annotation", "mask"],
                    k=16
                ),
                ]
        )
        
        super().__init__(data, transforms)
        #name, file_location, padded_size, bbox,  = []
        print("Preprocessing:\n")
        for i, item in enumerate(self.data):
            name = item["image"].with_suffix('').stem
            print(f"File: {name}")
            item = self.__getitem__(i)

            # Save
            #p

        print(item["image_meta_dict"])

if __name__=="__main__":
    from pathlib import Path
    from interactivenet.preprocessing.fingerprinting import FingerPrint

    import os
    exp = os.environ["interactiveseg_raw"]
    task = "Task001_Lipo"
    images = [x for x in Path(exp, task, "imagesTr").glob('**/*') if x.is_file()]
    masks = [x for x in Path(exp, task, "labelsTr").glob('**/*') if x.is_file()]
    annotations = [x for x in Path(exp, task, "interactionTr").glob('**/*') if x.is_file()]
    results = FingerPrint(sorted(images)[:2], sorted(masks)[:2], sorted(annotations)[:2])
    results()
    Preprocessing(sorted(images)[:2], sorted(masks)[:2], sorted(annotations)[:2], results.dim, results.target_spacing, task)