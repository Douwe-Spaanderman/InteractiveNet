import os
from typing import List, Tuple, Dict, Sequence, Optional, Callable, Union

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    DivisiblePadd
)

from interactivenet.transforms.transforms import Resamplingd, EGDMapd, BoudingBoxd, Visualized, NormalizeValuesd, AddDirectoryd

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
        verbose: bool = False,
        compose: bool = True,
    ):

    transforms = [
        AddDirectoryd(keys=["image", "interaction", "label"], directory=raw_path, convert_to_pathlib=True),
        LoadImaged(keys=["image", "interaction", "label"]),
        EnsureChannelFirstd(keys=["image", "interaction", "label"]),
    ]

    if verbose:
        transforms += [
            Visualized(
                keys=["image", "interaction", "label"],
                save=processed_path / 'verbose' / 'raw',
                interaction=True,
                CT=ct
            ),
        ]
    
    transforms += [
        Resamplingd(
            keys=["image", "interaction", "label"],
            pixdim=target_spacing,
        ),
        BoudingBoxd(
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
                save=processed_path / 'verbose' / 'processed',
                interaction=True,
                CT=ct
            ),
        ]
    
    transforms += [
        EGDMapd(
            keys=["interaction"],
            image="image",
            lamb=1,
            iter=4,
            logscale=True,
            ct=ct
        ),
        DivisiblePadd(
            keys=["image", "interaction", "label"],
            k=divisble_using
        ),
    ]

    if verbose:
        transforms += [
            Visualized(
                keys=["interaction", "label"],
                save=processed_path / 'verbose' / 'map',
                distancemap=True,
                CT=ct
            )
        ]

    if compose:
        return Compose(transforms)
    else:
        return transforms