from typing import Optional, Union, Dict, List, Tuple
import os

import warnings
import pickle
import math
from pathlib import Path
import torch
from itertools import combinations

from monai.transforms.transform import MapTransform, Transform
from monai.transforms import NormalizeIntensity, GaussianSmooth, Flip
import numpy as np
import GeodisTK
from interactivenet.utils.utils import to_pathlib
from interactivenet.utils.resample import resample_image, resample_label, resample_interaction
from interactivenet.utils.visualize import ImagePlot

class TestTimeFlipping(Transform):
    """
        This transform class takes list of interactions to array.
        That code is in:
    """

    def __init__(
        self,
        back=False,
        all_dimensions=True,
    ) -> None:
        self.back = back
        self.all_dimensions = all_dimensions

    def __call__(self, image: torch.tensor) -> torch.tensor:
        if self.all_dimensions:
            spatial_axis = [0,1,2]
        else:
            spatial_axis = [0,1]

        all_combinations = []
        for n in range(len(spatial_axis) + 1):
            all_combinations +=  list(combinations(spatial_axis, n))

        new_image = [image[0]]
        if not self.back:
            for spatial_axis in all_combinations[1:]:
                flipping = Flip(spatial_axis=spatial_axis)
                new_image += torch.stack([flipping(i[None,:]) for i in image[0]], dim=1)

        else:
            for idx, spatial_axis in enumerate(all_combinations[1:], 1):
                flipping = Flip(spatial_axis=spatial_axis)
                img = image[idx]
                new_image += torch.stack([flipping(i[None,:]) for i in img], dim=1)

        img = torch.stack(new_image)
        print('Shape output')
        print(img.shape)
        print('')

        return img

class OriginalSize(Transform):
    """
    Return the label to the original image shape
    """

    def __init__(
        self,
        anisotrophic:bool,
        resample:bool = True,
    ) -> None:
        self.anisotrophic = anisotrophic
        self.resample =resample

    def __call__(self, img: np.ndarray, meta: Dict) -> np.ndarray:
        """
        Apply the transform to `img` using `meta`.
        """

        if (np.array(img[0,:].shape) != np.array(meta["final_bbox_shape"])).all():
            raise ValueError("image and metadata don't match so can't restore to original size")

        if self.resample:
            new_size = tuple(meta["new_dim"])
        else:
            new_size = tuple(meta["spatial_shape"])

        box_start = np.array(meta["final_bbox"])
        padding = [box_start[0], [
            new_size[0] - box_start[1][0],
            new_size[1] - box_start[1][1],
            new_size[2] - box_start[1][2]
        ]]

        old_size = img.shape[1:]
        zero_padding = np.array(meta["zero_padding"])
        zero_padding = [
            [zero_padding[0][0], zero_padding[0][1], zero_padding[0][2]],
            [old_size[0] - zero_padding[1][0], old_size[1] - zero_padding[1][1], old_size[2] - zero_padding[1][2]],
        ]

        new_img = []
        method = ["maximum"] + ["minimum"]*(img.shape[0]-1)
        for i, channel in enumerate(img):
            box = channel[zero_padding[0][0]:zero_padding[1][0],zero_padding[0][1]:zero_padding[1][1],zero_padding[0][2]:zero_padding[1][2]]
            new_img.append(np.pad(
                box, (
                    (padding[0][0], padding[1][0]),
                    (padding[0][1], padding[1][1]),
                    (padding[0][2], padding[1][2])),
                method[i])
            )

        new_img = np.stack(new_img, axis=0)

        if new_img[0].shape != new_size:
            raise ValueError("New img and new size do know have the same size??")

        if self.resample:
            new_img = resample_image(new_img, meta["org_dim"], anisotrophy_flag=meta["anisotrophy_flag"])

        return new_img

class LoadWeightsd(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        ref_image
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.ref_image = ref_image

    def __call__(self, data):
        d = dict(data)

        img = d[self.ref_image]
        for key in self.keys:
            weights = np.load(d[key])[key]
            if weights.shape[1:] != img.shape:
                raise ValueError(f"Something went wrong with the weights and image shape, as they don't match. (weights: {weights.shape}, image: {img.shape}")

            d[key] = weights
            d[f"{key}_meta_dict"] = d[f"{self.ref_image}_meta_dict"]

        return d

class AddDirectoryd(MapTransform):
    """
        This transform class appends the complete path to the objects in the dictionary.
    """
    def __init__(
        self,
        keys=Union[str, List[str]],
        directory: Optional[Union[str, os.PathLike]]=None,
        convert_to_pathlib: bool = True,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.directory = directory
        self.convert_to_pathlib = convert_to_pathlib

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            if self.directory:
                if isinstance(self.directory, os.PathLike):
                    value = self.directory / d[key]
                elif self.directory.endswith("/"):
                    value = self.directory + "/" + d[key]
                else:
                    value = self.directory + d[key]
            else:
                value = d[key]

            if self.convert_to_pathlib and value != "":
                value = Path(value)

            d[key] = value

        return d

class Visualized(MapTransform):
    """
        This transform class visualizes images at different timepoints of the interactivenet pipeline.
    """

    def __init__(
        self,
        keys,
        interaction=None,
        distancemap=False,
        CT=False,
        save=None,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.interaction = interaction
        self.additional = None
        self.distancemap = distancemap
        self.CT = CT
        self.save = save

    def __call__(self, data):
        if len(self.keys) < 1:
            raise KeyError(f"Please provide both an image and mask, now only {self.keys} is provided")

        d = dict(data)
        if self.interaction or self.distancemap:
            if len(self.keys) > 3:
                self.additional = self.keys[3:]
                self.additional = [d[f'{x}'] for x in self.additional]

            if self.distancemap:
                image, label = self.keys[:2]
                if self.interaction:
                    interaction = [d[f'{interaction}_backup']]
                else:
                    interaction = None
            else:
                image, interaction, label = self.keys[:3]
                interaction = [d[f"{interaction}"]]

        else:
            if len(self.keys) > 2:
                self.additional = self.keys[2:]
                self.additional = [d[x] for x in self.additional]

            image, label = self.keys[:2]
            interaction = None

        if self.save:
            save = self.save / Path(d[f'{label}_meta_dict']["filename_or_obj"]).name.split('.')[0]
        else:
            save = None

        image = d[f'{image}']
        label = d[f'{label}']

        ImagePlot(image, label, interaction=interaction, additional_scans=self.additional, CT=self.CT, save=save, show=False)

        return d

class Resamplingd(MapTransform):
    """
        This transform class takes NNUNet's resampling method and applies it to our data structure.
    """

    def __init__(
        self,
        keys:Union[str, List[str]],
        pixdim:List[float],
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.target_spacing = pixdim

    def calculate_new_shape(self, spacing_ratio:Union[np.ndarray, torch.Tensor], shape:Union[np.ndarray, torch.Tensor]):
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing:List[float]):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def sanity_in_mask(self, interaction:Union[np.ndarray, torch.Tensor], label:Union[np.ndarray, torch.Tensor]):
        sanity = []
        for i, interaction_d in enumerate(interaction):
            label_d = label[i]
            idx_x, idx_y, idx_z = np.where(interaction_d > 0.5)
            sanity_d = []
            for x, y, z in zip(idx_x, idx_y, idx_z):
                sanity_d.append(label_d[x, y, z] == 1)

            sanity.append(not any(sanity_d))

        return not any(sanity)

    def __call__(self, data):
        d = dict(data)

        if "image" in self.keys:
            message = "Resampling, image, "
            image = d["image"]
        else:
            raise KeyError("No image provided for resampling, this is not possible...")

        if "interaction" in self.keys:
            message += "interaction, "
            interaction = d["interaction"]
            interaction[interaction < 0] = 0
        else:
            warnings.warn("No interactions are provided...")

        if "label" in self.keys:
            message += "label, "
            label = d["label"]
            label[label < 0] = 0

        image_spacings = d["image_meta_dict"]["pixdim"][1:4].tolist()
        print(f"Original Spacing: {image_spacings} \t Target Spacing: {self.target_spacing}")

        # calculate shape
        original_shape = image.shape[1:]
        resample_flag = False
        anisotrophy_flag = False

        if self.target_spacing != image_spacings:
            print(message + "because current spacing != target spacing")
            resample_flag = True
            spacing_ratio = np.array(image_spacings) / np.array(self.target_spacing)
            resample_shape = self.calculate_new_shape(spacing_ratio, original_shape)
            print(f"Original Shape: {original_shape} \t Target Shape: {resample_shape}")

            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            print(f"Resampling anisotropic set to {anisotrophy_flag}")

            # Actual resampling
            image = resample_image(image, resample_shape, anisotrophy_flag)

            if "label" in self.keys:
                label = resample_label(label, resample_shape, anisotrophy_flag)

            if "interaction" in self.keys:
                interaction = resample_interaction(d["interaction"], d['interaction_meta_dict']["affine"], self.target_spacing, resample_shape)
                if "label" in self.keys:
                    d["interaction_meta_dict"]["in_mask"] = self.sanity_in_mask(interaction, label)
                    if d['interaction_meta_dict']["in_mask"] == False:
                        warnings.warn("interactions are outside of the mask, please fix this")

        new_meta = {
            "org_spacing": np.array(image_spacings),
            "org_dim": np.array(original_shape),
            "new_spacing": np.array(self.target_spacing),
            "new_dim": np.array(resample_shape),
            "resample_flag": resample_flag,
            "anisotrophy_flag": anisotrophy_flag,
        }

        d["image"] = image
        d["image_meta_dict"].update(new_meta)

        if "interaction" in self.keys:
            d["interaction"] = interaction
            d["interaction_meta_dict"].update(new_meta)

        if "label" in self.keys:
            d["label"] = label
            d["label_meta_dict"].update(new_meta)

        return d


class BoudingBoxd(MapTransform):
    """
        This transform class takes the bounding box of an object based on the mask or annotations.
    """

    def __init__(
        self,
        keys:Union[str, List[str]],
        on:str,
        relaxation:Union[float, Tuple[float]]=0,
        divisiblepadd:Optional[Union[int, Tuple[int]]]=None,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.on = on

        if isinstance(relaxation, float):
            relaxation = [relaxation] * 3

        if divisiblepadd:
            if isinstance(divisiblepadd, int):
                divisiblepadd = [divisiblepadd] * 3

        self.relaxation = relaxation
        self.divisiblepadd = divisiblepadd

    def calculate_bbox(self, data:np.ndarray):
        inds_x, inds_y, inds_z = np.where(data > 0.5)

        bbox = np.array([
            [
                np.min(inds_x),
                np.min(inds_y),
                np.min(inds_z)
                ],
            [
                np.max(inds_x),
                np.max(inds_y),
                np.max(inds_z)
            ]
        ])

        return bbox

    def calculate_relaxtion(self, bbox_shape:np.ndarray, anisotropic:bool=False):
        relaxation = [0] * len(bbox_shape)
        for i, axis in enumerate(range(len(bbox_shape))):
            relaxation[axis] = math.ceil(bbox_shape[axis] * self.relaxation[axis])

            if anisotropic and i == 2: # This is only possible with Z on final axis
                check = 3
            else:
                check = 8

            if relaxation[axis] < check:
                warnings.warn(f"relaxation was to small: {relaxation[axis]}, so adjusting it to {check}")
                relaxation[axis] = check

        return relaxation

    def relax_bbox(self, data:np.ndarray, bbox:np.ndarray, relaxation:List[int]):
        bbox = np.array([
            [
                bbox[0][0] - relaxation[0],
                bbox[0][1] - relaxation[1],
                bbox[0][2] - relaxation[2]
                ],
            [
                bbox[1][0] + relaxation[0],
                bbox[1][1] + relaxation[1],
                bbox[1][2] + relaxation[2],
            ]
        ])
        for axis in range(len(bbox[0])):
            if bbox[0,axis] == bbox[1,axis]:
                bbox[0,axis] = bbox[0,axis] - 1
                bbox[1,axis] = bbox[1,axis] + 1
                warnings.warn(f"Bounding box has the same size in {axis} axis so extending axis by 1 both direction")

        # Remove below zero and higher than shape because of relaxation
        bbox[bbox < 0] = 0
        largest_dimension = [int(x) if  x <= data.shape[i] else data.shape[i] for i, x in enumerate(bbox[1])]
        bbox = np.array([bbox[0].tolist(), largest_dimension])

        zeropadding = np.zeros(3)
        if self.divisiblepadd:
            for axis in range(len(self.divisiblepadd)):
                expand = True
                while expand == True:
                    bbox_shape = np.subtract(bbox[1][axis],bbox[0][axis])
                    residue = bbox_shape % self.divisiblepadd[axis]
                    if residue != 0:
                        residue = self.divisiblepadd[axis] - residue
                        if residue < 2:
                            neg = bbox[0][axis] - 1
                            if neg >= 0:
                                bbox[0][axis] = neg
                            else:
                                pos = bbox[1][axis] + 1
                                if pos <= data.shape[axis]:
                                    bbox[1][axis] = pos
                                else:
                                    zeropadding[axis] = zeropadding[axis] + residue
                                    warnings.warn(f"bbox doesn't fit in the image for axis {axis}, adding zero padding {residue}")
                                    expand = False
                        else:
                            neg = bbox[0][axis] - 1
                            if neg >= 0:
                                bbox[0][axis] = neg

                            pos = bbox[1][axis] + 1
                            if pos <= data.shape[axis]:
                                bbox[1][axis] = pos

                            if neg <= 0 and pos > data.shape[axis]:
                                zeropadding[axis] = zeropadding[axis] + residue
                                warnings.warn(f"bbox doesn't fit in the image for axis {axis}, adding zero padding {residue}")
                                expand = False
                    else:
                        expand = False

        padding = np.zeros((2,3), dtype=int)
        if any(zeropadding > 0):
            for idx, value in enumerate(zeropadding):
                x = int(value / 2)
                y = int(value - x)
                padding[0][idx] = x
                padding[1][idx] = y

        return bbox, padding

    def extract_bbox_region(self, data:np.ndarray, bbox:np.ndarray, padding:np.ndarray):
        new_region = data[
                bbox[0][0]:bbox[1][0],
                bbox[0][1]:bbox[1][1],
                bbox[0][2]:bbox[1][2]
                ]

        new_region = np.pad(
            new_region, (
                (padding[0][0], padding[1][0]),
                (padding[0][1], padding[1][1]),
                (padding[0][2], padding[1][2])),
            'constant')

        return new_region

    def __call__(self, data):
        d = dict(data)
        data_type = None
        keys = list(self.key_iterator(d))

        bbox = self.calculate_bbox(d[self.on][0])
        bbox_shape = np.subtract(bbox[1],bbox[0])
        relaxation = self.calculate_relaxtion(bbox_shape, d["image_meta_dict"]["anisotrophy_flag"])

        print(f"Original bouding box at location: {bbox[0]} and {bbox[1]} \t shape of bbox: {bbox_shape}")
        final_bbox, zeropadding = self.relax_bbox(d[self.on][0], bbox, relaxation)
        final_bbox_shape = np.subtract(final_bbox[1],final_bbox[0])
        print(f"Bouding box at location: {final_bbox[0]} and {final_bbox[1]} \t bbox is relaxt with: {relaxation} \t and zero_padding: {zeropadding} \t and made divisible with: {self.divisiblepadd} \t shape after cropping: {final_bbox_shape}")
        for key in self.keys:
            if len(d[key].shape) == 4:
                new_dkey = []
                for idx in range(d[key].shape[0]):
                    new_dkey.append(self.extract_bbox_region(d[key][idx], final_bbox, zeropadding))
                d[key] = np.stack(new_dkey, axis=0)
                final_size = d[key].shape[1:]
            else:
                d[key] = self.extract_bbox_region(d[key], final_bbox, zeropadding)
                final_size = d[key].shape

            d[f"{key}_meta_dict"]["bbox"] = bbox
            d[f"{key}_meta_dict"]["bbox_shape"] = bbox_shape
            d[f"{key}_meta_dict"]["bbox_relaxation"] = self.relaxation
            d[f"{key}_meta_dict"]["final_bbox"] = final_bbox
            d[f"{key}_meta_dict"]["final_bbox_shape"] = final_bbox_shape
            d[f"{key}_meta_dict"]["zero_padding"] = zeropadding
            d[f"{key}_meta_dict"]["final_size"] = final_size

        return d

class NormalizeValuesd(MapTransform):
    """
        This transform class takes NNUNet's normalization method and applies it to our data structure.
    """

    def __init__(
        self,
        keys:Union[str, List[str]],
        clipping:List[float]=[],
        mean:float=0,
        std:float=0,
        nonzero:bool=True,
        channel_wise:bool=True,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.clipping = clipping
        self.mean = mean
        self.std = std
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            image = d[key]
            if self.clipping:
                d[f"{key}_EGD"] = (image - self.mean) / self.std
                image = np.clip(image, self.clipping[0], self.clipping[1])
                image = (image - self.mean) / self.std
            else:
                image = self.normalize_intensity(image.copy())

            d[key] = image

        return d


class EGDMapd(MapTransform):
    """
        This transform class creates an exponetialized geodesic distance map, based on an image and annotations.
        For more information you can look into:
        https://github.com/taigw/GeodisTK
    """

    def __init__(
        self,
        keys:Union[str, List[str]],
        image:str,
        lamb:int=1,
        iter:int=4,
        logscale:bool=True,
        ct:bool=False,
        backup:bool=False,
        powerof:bool=False,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.image = image
        self.lamb = lamb
        self.iter = iter
        self.logscale = logscale
        self.backup = backup
        self.ct = ct
        self.powerof = powerof
        self.gaussiansmooth = GaussianSmooth(sigma=1)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            if self.backup:
                d[f"{key}_backup"] = d[key].copy()

            if "new_spacing" in d[f'{self.image}_meta_dict'].keys():
                spacing = d[f'{self.image}_meta_dict']["new_spacing"]
            else:
                spacing = np.asarray(d[f'{self.image}_meta_dict']["pixdim"][1:4])

            if f"{self.image}_EGD" in d.keys():
                image = d[f"{self.image}_EGD"]
                del d[f'{self.image}_EGD']
            else:
                image = d[self.image]

            if len(d[key].shape) == 4:
                for idx in range(d[key].shape[0]):
                    img = image[idx]

                    GD = GeodisTK.geodesic3d_raster_scan(img.astype(np.float32), d[key][idx].astype(np.uint8), spacing.astype(np.float32), self.lamb, self.iter)
                    if self.powerof:
                        GD = GD**self.powerof

                    if self.logscale == True:
                        GD = np.exp(-GD)

                    d[key][idx, :, :, :] = GD
            else:

                GD = GeodisTK.geodesic3d_raster_scan(image.astype(np.float32), d[key].astype(np.uint8), spacing.astype(np.float32), self.lamb, self.iter)
                if self.powerof:
                    GD = GD**self.powerof

                if self.logscale == True:
                    GD = np.exp(-GD)

                d[key] = GD

        print(f"Geodesic Distance Map with lamd: {self.lamb}, iter: {self.iter} and logscale: {self.logscale}")
        return d

class SavePreprocessed(MapTransform):
    """
        This transform class saves the preprocessed data to .npz and .pkl files
    """
    def __init__(
        self,
        keys: Union[str, List[str]],
        save: Optional[Union[str, os.PathLike]]=None,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.meta_keys = [key + "_meta_dict" for key in self.keys]
        self.save = to_pathlib(save)

    def __call__(self, data):
        d = dict(data)
        name = data[f"{self.meta_keys[-1]}"]["filename_or_obj"].split("/")[-1].split(".nii.gz")[0]

        np.savez(self.save / name, **{key :d[key] for key in self.keys})

        pickle_data = {
            key : d[key] for key in self.meta_keys
        }

        with open(self.save / (name + ".pkl"), 'wb') as handle:
            pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return d

class LoadPreprocessed(MapTransform):
    """
        This transform class loads the preprocessed .npz and .pkl files
    """

    def __init__(
        self,
        keys:Union[str, List[str]],
        new_keys:Union[str, List[str]],
     ) -> None:
        super().__init__(keys)
        self.keys = keys
        if len(self.keys) != 2:
            raise ValueError(f"LoadPreprocessed data assumes the data has 2 keys with npz and metadata, this is not the case as there are {len(self.keys)} provided")

        self.new_keys = new_keys
        self.meta_keys = [x + "_meta_dict" for x in new_keys]

    def read_pickle(self, filename:Optional[Union[str, os.PathLike]]):
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)

        return b

    def __call__(self, data):
        d = dict(data)
        new_d = {}
        for key in self.keys:
            current_data = d[key]
            if current_data.suffix == ".npz":
                image_data = np.load(d[key])
                old_keys = list(image_data.keys())
                if not len(old_keys) == len(self.new_keys):
                    raise KeyError("Old keys and new keys do not have the same length in preprocessed data loader")

                if old_keys == self.new_keys:
                    for new_key in self.new_keys:
                        new_d[new_key] = image_data[new_key]

                else:
                    warnings.warn("old keys do not match new keys, however are the right length so just applying it in order")
                    for old_key, new_key in zip(old_keys, self.new_keys):
                        new_d[new_key] = image_data[old_key]

            elif current_data.suffix == ".pkl":
                metadata = self.read_pickle(d[key])
                old_keys = list(metadata.keys())
                
                if not len(old_keys) == len(self.meta_keys):
                    raise KeyError("Old keys and new keys do not have the same length in preprocessed data loader")

                if old_keys == self.meta_keys:
                    for new_key in self.meta_keys:
                        new_d[new_key] = metadata[new_key]

                else:
                    warnings.warn("old keys do not match new keys, however are the right length so just applying it in order")
                    for old_key, new_key in zip(old_keys, self.meta_keys):
                        new_d[new_key] = metadata[old_key]
            else:
                raise ValueError("Neither npz or pkl in preprocessed loader")

        return new_d