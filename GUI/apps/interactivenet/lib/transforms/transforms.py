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

import logging
from typing import Union, List
from itertools import combinations

from monai.transforms.transform import MapTransform, Transform
from monai.transforms import (
    Flip,
    AsDiscrete,
    KeepLargestConnectedComponent,
    FillHoles,
)
from monai.data import MetaTensor
import numpy as np

import torch

from interactivenet.utils.resample import (
    resample_image,
    resample_label,
)

logger = logging.getLogger(__name__)


class EarlyCroppingd(MapTransform):
    """
    This transform class takes the bounding box of an object based on the mask or annotations. This is different than the BoundingBox transform
    as now the cropping is done with a larger margin and before resampling. So now instead of doing resampling on the whole image and than 
    extracting the bounding box, we now first extract a large bounding box in order to speed up resampling:
    non-fast (original)   : resampling -> bounding box
    fastR (this transform): large bounding box -> resampling -> bounding box 

    Same applies for the weight to orginal size at the end!
    """

    def __init__(
        self,
        keys: Union[str, List[str]],
        on: str,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.on = on

    def check_anisotrophy(self, spacing: List[float]):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)


    def calculate_bbox(self, data: np.ndarray):
        inds_x, inds_y, inds_z = np.where(data > 0.5)

        bbox = np.array(
            [
                [np.min(inds_x), np.min(inds_y), np.min(inds_z)],
                [np.max(inds_x), np.max(inds_y), np.max(inds_z)],
            ]
        )

        return bbox

    def calculate_relaxtion(self, bbox_shape: np.ndarray, anisotropic: bool = False):
        relaxation = []
        for i, axis in enumerate(range(len(bbox_shape))):

            if anisotropic and i == 2:  # This is only possible with Z on final axis
                relaxation.append(5)
            else:
                relaxation.append(15)

        return relaxation

    def relax_bbox(self, data: np.ndarray, bbox: np.ndarray, relaxation: List[int]):
        bbox = np.array(
            [
                [
                    bbox[0][0] - relaxation[0],
                    bbox[0][1] - relaxation[1],
                    bbox[0][2] - relaxation[2],
                ],
                [
                    bbox[1][0] + relaxation[0],
                    bbox[1][1] + relaxation[1],
                    bbox[1][2] + relaxation[2],
                ],
            ]
        )
        for axis in range(len(bbox[0])):
            if bbox[0, axis] == bbox[1, axis]:
                bbox[0, axis] = bbox[0, axis] - 2
                bbox[1, axis] = bbox[1, axis] + 2
                warnings.warn(
                    f"EarlyCroppingd box has the same size in {axis} axis so extending axis by 1 both direction"
                )

        # Remove below zero and higher than shape because of relaxation
        bbox[bbox < 0] = 0
        largest_dimension = [
            int(x) if x <= data.shape[i] else data.shape[i]
            for i, x in enumerate(bbox[1])
        ]
        bbox = np.array([bbox[0].tolist(), largest_dimension])

        return bbox

    def extract_bbox_region(
        self, data: np.ndarray, bbox: np.ndarray
    ):
        new_region = data[
            bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1], bbox[0][2] : bbox[1][2]
        ]

        return new_region

    def __call__(self, data):
        d = dict(data)
        list(self.key_iterator(d))

        bbox = self.calculate_bbox(d[self.on][0])
        bbox_shape = np.subtract(bbox[1], bbox[0])
        anisotrophy_flag = self.check_anisotrophy(d["image_meta_dict"]["pixdim"][1:4].tolist())
        relaxation = self.calculate_relaxtion(
            bbox_shape, anisotrophy_flag
        )

        final_bbox = self.relax_bbox(d[self.on][0], bbox, relaxation)
        final_bbox_shape = np.subtract(final_bbox[1], final_bbox[0])
        print(
            f"EarlyCroppingd box at location: {final_bbox[0]} and {final_bbox[1]} \t shape after cropping: {final_bbox_shape}"
        )
        for key in self.keys:
            if len(d[key].shape) == 4:
                new_dkey = []
                for idx in range(d[key].shape[0]):
                    new_dkey.append(
                        self.extract_bbox_region(d[key][idx], final_bbox)
                    )
                d[key] = np.stack(new_dkey, axis=0)
                final_size = d[key].shape[1:]
            else:
                d[key] = self.extract_bbox_region(d[key], final_bbox)
                final_size = d[key].shape

            d[f"{key}_meta_dict"]["EarlyCroppingd_bbox"] = final_bbox

        return d


class OriginalSized(Transform):
    """
    Return the label to the original image shape
    """

    def __init__(
        self,
        img_key,
        ref_meta,
        keep_key=None,
        label: bool = True,
        postprocess: bool = False,
        discreet: bool = True,
        fast:bool = False,
        device=None,
    ) -> None:
        self.img_key = img_key
        self.ref_meta = ref_meta
        self.keep_key = keep_key
        self.label = label
        self.postprocess = postprocess
        self.discreet = discreet
        self.fast = fast
        self.device = device
        self.as_discrete = AsDiscrete(argmax=True)

        self.largestcomponent = KeepLargestConnectedComponent()
        self.fillholes = FillHoles()

    def __call__(self, data):
        """
        Apply the transform to `img` using `meta`.
        """

        d = dict(data)

        img = d[self.img_key]
        meta = d[f"{self.ref_meta}_meta_dict"]

        if (np.array(img[0, :].shape) != np.array(meta["final_bbox_shape"])).all():
            raise ValueError(
                "image and metadata don't match so can't restore to original size"
            )

        new_size = tuple(meta["new_dim"])
        box_start = meta["final_bbox"]
        padding = [
            box_start[0],
            [
                new_size[0] - box_start[1][0],
                new_size[1] - box_start[1][1],
                new_size[2] - box_start[1][2],
            ],
        ]

        old_size = img.shape[1:]
        zero_padding = np.array(meta["zero_padding"])
        zero_padding = [
            [zero_padding[0][0], zero_padding[0][1], zero_padding[0][2]],
            [
                old_size[0] - zero_padding[1][0],
                old_size[1] - zero_padding[1][1],
                old_size[2] - zero_padding[1][2],
            ],
        ]

        if img.shape[0] > 1:
            method = [np.max(img[0])]
            for channel in img[1:]:
                method.append(np.min(channel))
        else:
            method = [np.min(img)]

        new_img = []
        for i, channel in enumerate(img):
            box = channel[
                zero_padding[0][0] : zero_padding[1][0],
                zero_padding[0][1] : zero_padding[1][1],
                zero_padding[0][2] : zero_padding[1][2],
            ]
            new_img.append(
                np.pad(
                    box,
                    (
                        (padding[0][0], padding[1][0]),
                        (padding[0][1], padding[1][1]),
                        (padding[0][2], padding[1][2]),
                    ),
                    constant_values=method[i],
                )
            )

        img = np.stack(new_img, axis=0)

        if img[0].shape != new_size:
            raise ValueError("New img and new size do know have the same size??")

        if self.keep_key:
            cache = img.copy()

        if self.discreet:
            img = self.as_discrete(img)

        if self.discreet and self.postprocess:
            img = self.largestcomponent(img)
            img = self.fillholes(img)
        elif self.postprocess:
            raise KeyError("label is not discreet, so cannot do postprocessing...")

        new_img = []
        for i, channel in enumerate(img):
            if self.label or self.discreet:
                new_img.append(
                    resample_label(
                        channel[None, :],
                        meta["org_dim"],
                        anisotrophy_flag=meta["anisotrophy_flag"],
                    )[0]
                )
            else:
                new_img.append(
                    resample_image(
                        channel[None, :],
                        meta["org_dim"],
                        anisotrophy_flag=meta["anisotrophy_flag"],
                    )[0]
                )

        new_img = np.stack(new_img, axis=0)

        if self.fast:
            fast_box = meta["EarlyCroppingd_bbox"]
            fast_padding = [
                fast_box[0],
                [
                    fast_box[1][0] - meta["org_dim"][0],
                    fast_box[1][1] - meta["org_dim"][1],
                    fast_box[1][2] - meta["org_dim"][2],
                ],
            ]

            # I think I should still check for the method...
            new_img = np.pad(
                new_img,
                (
                    (0, 0),
                    (fast_padding[0][0], fast_padding[1][0]),
                    (fast_padding[0][1], fast_padding[1][1]),
                    (fast_padding[0][2], fast_padding[1][2]),
                ),
                constant_values=0,
            )

        new_img = torch.tensor(new_img, dtype=torch.float32, device=self.device)
        d[self.img_key] = MetaTensor(new_img, meta.get("original_affine"))

        meta_dict = d.get(f"{self.img_key}_meta_dict")
        if meta_dict is None:
            meta_dict = dict()
            d[f"{self.img_key}_meta_dict"] = meta_dict
        meta_dict["affine"] = meta.get("original_affine")

        if self.keep_key:
            cache = np.stack(cache, axis=0)
            d[self.keep_key] = MetaTensor(
                torch.tensor(cache), meta.get("original_affine")
            )

        return d


class TestTimeFlippingd(MapTransform):
    """
    This transform class takes list of annotations to array.
    That code is in:
    """

    def __init__(
        self,
        keys,
        all_dimensions=True,
        back=False,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.all_dimensions = all_dimensions
        self.back = back

    def __call__(self, data):
        d = dict(data)
        data_type = None
        keys = list(self.key_iterator(d))
        for key in keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")

        if self.all_dimensions:
            spatial_axis = [0, 1, 2]
        else:
            spatial_axis = [0, 1]

        all_combinations = []
        for n in range(len(spatial_axis) + 1):
            all_combinations += list(combinations(spatial_axis, n))

        for key in keys:
            image = d[key]
            if not self.back:
                new_image = [image]
                for spatial_axis in all_combinations[1:]:
                    flipping = Flip(spatial_axis=spatial_axis)
                    new_image += flipping(image)[None, :]

                d[key] = torch.stack(new_image)
            else:
                if len(image.shape) == 5:
                    image = image[None, :]

                new_image = []
                for img in image:
                    new_image += [img[0]]
                    for idx, spatial_axis in enumerate(all_combinations[1:], 1):
                        flipping = Flip(spatial_axis=spatial_axis)
                        current_img = img[idx]
                        new_image += torch.stack(
                            [flipping(i[None, :]) for i in current_img], dim=1
                        )

                d[key] = torch.stack(new_image)

        return d


class AnnotationToChanneld(MapTransform):
    """
    This transform class takes list of annotations to array.
    That code is in:
    """

    def __init__(self, ref_image, guidance, method="interactivenet") -> None:
        super().__init__(guidance)
        self.ref_image = ref_image
        self.guidance = guidance
        self.method = method

    def __call__(self, data):
        d = dict(data)
        click_map = []

        for clicks in d[self.guidance]:  # pos and neg
            if clicks:
                if click_map and self.method == "interactivenet":
                    logger.info(
                        f"PRE - Transform (AnnotationToChanneld): Discarding negatives clicks because of method {self.method}"
                    )
                    continue

                annotation_map = torch.zeros(d[self.ref_image].shape)
                if len(clicks) < 1 and self.method == "interactivenet":
                    raise KeyError("please provide 6 interactions")

                for click in clicks:
                    annotation_map[click[0], click[1], click[2]] = 1

                click_map.append(annotation_map)

        d[self.guidance] = MetaTensor(
            torch.stack(click_map, dim=0),
            affine=d[self.ref_image].affine,
            meta=d[f"{self.ref_image}_meta_dict"],
        )
        d[f"{self.guidance}_meta_dict"] = d[f"{self.ref_image}_meta_dict"]

        return d
