import os
from typing import List, Tuple, Optional, Union

import numpy as np
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    DivisiblePadd,
    Compose,
    RandFlipd,
    RandScaleIntensityd,
    ConcatItemsd,
    ToTensord,
    RandGaussianNoised,
    RandGaussianSmoothd,
    CastToTyped,
    RandAdjustContrastd,
    RandZoomd,
    RandRotated,
    CopyItemsd,
)

from interactivenet.transforms.transforms import (
    Resamplingd,
    EGDMapd,
    BoundingBoxd,
    Visualized,
    NormalizeValuesd,
    AddDirectoryd,
    SavePreprocessed,
    LoadPreprocessed,
)


def processing_transforms(
    target_spacing: Tuple[float],
    processed_path: Union[str, os.PathLike],
    raw_path: Optional[Union[str, os.PathLike]] = None,
    relax_bbox: Union[float, Tuple[float]] = 0.1,
    divisble_using: Union[int, Tuple[int]] = (16, 16, 8),
    clipping: List[float] = [],
    intensity_mean: float = 0,
    intensity_std: float = 0,
    ct: bool = False,
    save: bool = True,
    verbose: bool = False,
    compose: bool = True,
):
    transforms = [
        AddDirectoryd(
            keys=["image", "interaction", "label"],
            directory=raw_path,
            convert_to_pathlib=True,
        ),
        LoadImaged(keys=["image", "interaction", "label"]),
        EnsureChannelFirstd(keys=["image", "interaction", "label"]),
    ]

    if verbose:
        transforms += [
            Visualized(
                keys=["image", "interaction", "label"],
                save=processed_path / "verbose" / "raw",
                interaction=True,
                CT=ct,
            ),
        ]

    transforms += [
        Resamplingd(
            keys=["image", "interaction", "label"],
            pixdim=target_spacing,
        ),
        BoundingBoxd(
            keys=["image", "interaction", "label"],
            on="label",
            relaxation=relax_bbox,
            divisiblepadd=divisble_using,
        ),
        NormalizeValuesd(
            keys=["image"],
            clipping=clipping,
            mean=intensity_mean,
            std=intensity_std,
        ),
    ]

    if verbose:
        transforms += [
            Visualized(
                keys=["image", "interaction", "label"],
                save=processed_path / "verbose" / "processed",
                interaction=True,
                CT=ct,
            ),
        ]

    transforms += [
        EGDMapd(
            keys=["interaction"], image="image", lamb=1, iter=4, logscale=True, ct=ct
        ),
        DivisiblePadd(keys=["image", "interaction", "label"], k=divisble_using),
    ]

    if verbose:
        transforms += [
            Visualized(
                keys=["interaction", "label"],
                save=processed_path / "verbose" / "map",
                distancemap=True,
                CT=ct,
            )
        ]

    if save:
        transforms += [
            SavePreprocessed(
                keys=["image", "interaction", "label"],
                save=processed_path / "network_input",
            )
        ]

    if compose:
        return Compose(transforms)
    else:
        return transforms

def training_transforms(seed: Optional[int] = None, validation: bool = False):
    if seed:
        set_determinism(seed=seed)

    transforms = [
        LoadPreprocessed(
            keys=["npz", "metadata"], new_keys=["image", "interaction", "label"]
        ),
    ]

    if not validation:
        transforms += [
            RandRotated(
                keys=["image", "interaction", "label"],
                range_x=180,
                range_y=180,
                mode=("bilinear", "bilinear", "nearest"),
                align_corners=(True, True, None),
                prob=0.2,
            ),
            RandZoomd(
                keys=["image", "interaction", "label"],
                min_zoom=0.7,
                max_zoom=1.4,
                mode=("trilinear", "trilinear", "nearest"),
                align_corners=(True, True, None),
                prob=0.2,
            ),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.2,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandAdjustContrastd(keys=["image"], gamma=(0.65, 1.5), prob=0.15),
            RandFlipd(
                keys=["image", "interaction", "label"], spatial_axis=[0], prob=0.5
            ),
            RandFlipd(
                keys=["image", "interaction", "label"], spatial_axis=[1], prob=0.5
            ),
            RandFlipd(
                keys=["image", "interaction", "label"], spatial_axis=[2], prob=0.5
            ),
        ]

    transforms += [
        CastToTyped(
            keys=["image", "interaction", "label"],
            dtype=(np.float32, np.float32, np.uint8),
        ),
        ConcatItemsd(keys=["image", "interaction"], name="image"),
        ToTensord(keys=["image", "label"]),
    ]

    return Compose(transforms)


def inference_transforms(
    metadata: dict,
    labels: bool = False,
    raw_path: Optional[Union[str, os.PathLike]] = None,
):
    transforms = []
    if labels:
        transforms += [
            AddDirectoryd(
                keys=["image", "interaction", "label"],
                directory=raw_path,
                convert_to_pathlib=True,
            ),
            LoadImaged(keys=["image", "interaction", "label"]),
            EnsureChannelFirstd(keys=["image", "interaction", "label"]),
            CopyItemsd(keys=["image"], names=["image_raw"]),
        ]
    else:
        transforms += [
            AddDirectoryd(
                keys=["image", "interaction"],
                directory=raw_path,
                convert_to_pathlib=True,
            ),
            LoadImaged(keys=["image", "interaction"]),
            EnsureChannelFirstd(keys=["image", "interaction"]),
        ]

    transforms += [
        Resamplingd(
            keys=["image", "interaction"],
            pixdim=metadata["Fingerprint"]["Target spacing"],
        ),
        BoundingBoxd(
            keys=["image", "interaction"],
            on="interaction",
            relaxation=metadata["Plans"]["padding"],
            divisiblepadd=metadata["Plans"]["divisible by"],
        ),
        NormalizeValuesd(
            keys=["image"],
            clipping=metadata["Fingerprint"]["Clipping"],
            mean=metadata["Fingerprint"]["Intensity_mean"],
            std=metadata["Fingerprint"]["Intensity_std"],
        ),
        EGDMapd(
            keys=["interaction"],
            image="image",
            lamb=1,
            iter=4,
            logscale=True,
            ct=metadata["Fingerprint"]["CT"],
        ),
    ]

    transforms += [
        CastToTyped(keys=["image", "interaction"], dtype=(np.float32, np.float32)),
        ToTensord(keys=["image", "interaction"]),
        ConcatItemsd(keys=["image", "interaction"], name="image"),
    ]

    return Compose(transforms)
