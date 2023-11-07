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

from typing import List, Union
import numpy as np
import torch

from skimage.transform import resize
from nibabel import affines
import numpy.linalg as npl

from interactivenet.utils.utils import to_array


def resample_label(
    label: Union[np.ndarray, torch.Tensor], shape: List[int], anisotrophy_flag: bool
):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label)
    if anisotrophy_flag:
        shape_2d = shape[:-1]
        depth = label.shape[-1]
        reshaped_2d = np.zeros((*shape_2d, depth), dtype=np.uint8)

        for class_ in range(1, int(n_class) + 1):
            for depth_ in range(depth):
                mask = label[0, :, :, depth_] == class_
                resized_2d = resize(
                    mask.astype(float),
                    shape_2d,
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_
        for class_ in range(1, int(n_class) + 1):
            mask = reshaped_2d == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_
    else:
        for class_ in range(1, int(n_class) + 1):
            mask = label[0] == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_

    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


def resample_image(
    image: Union[np.ndarray, torch.Tensor], shape: List[int], anisotrophy_flag: bool
):
    resized_channels = []
    if anisotrophy_flag:
        for image_c in image:
            resized_slices = []
            for i in range(image_c.shape[-1]):
                image_c_2d_slice = image_c[:, :, i]
                image_c_2d_slice = resize(
                    image_c_2d_slice,
                    shape[:-1],
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_slices.append(image_c_2d_slice)
            resized = np.stack(resized_slices, axis=-1)
            resized = resize(
                resized,
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    else:
        for image_c in image:
            resized = resize(
                image_c,
                shape,
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    resized = np.stack(resized_channels, axis=0)
    return resized


def resample_interaction(
    image: Union[np.ndarray, torch.Tensor],
    affine: Union[np.ndarray, torch.Tensor],
    new_spacing: List[float],
    shape: List[int],
):
    resized_channels = []
    # Sanitize input
    affine = to_array(affine)
    for image_d in to_array(image):
        resized = np.zeros(shape)
        new_affine = affines.rescale_affine(
            affine, image_d.shape, new_spacing, new_shape=shape
        )

        inds_x, inds_y, inds_z = np.where(image_d > 0.5)
        for i, j, k in zip(inds_x, inds_y, inds_z):
            old_vox2new_vox = npl.inv(new_affine).dot(affine)
            new_point = np.rint(
                affines.apply_affine(old_vox2new_vox, [i, j, k])
            ).astype(int)

            for i in range(len(new_point)):
                if new_point[i] < 0:
                    new_point[i] = 0
                elif new_point[i] >= shape[i]:
                    new_point[i] = shape[i] - 1

            resized[new_point[0], new_point[1], new_point[2]] = 1

        resized_channels.append(resized)

    resized = np.stack(resized_channels, axis=0)
    return resized
