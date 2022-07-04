import warnings
import pickle
import math
from pathlib import Path

from monai.transforms.transform import MapTransform
import numpy as np
import GeodisTK
from interactivenet.utils.resample import resample_image, resample_label, resample_annotation
from interactivenet.utils.visualize import ImagePlot

class Visualized(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        annotation=None,
        distancemap=False,
        CT=False,
        save=None
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.annotation = annotation
        self.additional = None
        self.distancemap = distancemap
        self.CT = CT
        self.save = save

    def __call__(self, data):
        if len(self.keys) < 1:
            raise KeyError(f"Please provide both an image and mask, now only {self.keys} is provided")

        d = dict(data)
        if self.annotation or self.distancemap:
            if len(self.keys) > 3:
                self.additional = self.keys[3:]
                self.additional = [d[f'{x}'] for x in self.additional]

            image, annotation, label = self.keys[:3]
            annotation = [d[f'{annotation}']]
        else:
            if len(self.keys) > 2:
                self.additional = self.keys[2:]
                self.additional = [d[x] for x in self.additional]

            image, label = self.keys[:2]
            annotation = None

        if self.save:
            save = self.save / Path(d[f'{label}_meta_dict']["filename_or_obj"]).name.split('.')[0]

        image = d[f'{image}']
        label = d[f'{label}']

        if self.distancemap:
            ImagePlot(image, label, annotation=annotation, additional_scans=self.additional, distancemap=self.distancemap, save=save)
        else:
            ImagePlot(image, label, annotation=annotation, additional_scans=self.additional, CT=self.CT, save=save)

        return d

class Resamplingd(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        pixdim,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.target_spacing = pixdim

    def calculate_new_shape(self, spacing_ratio, shape):
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def sanity_in_mask(self, annotation, label):
        sanity = []
        for i, annotation_d in enumerate(annotation):
            label_d = label[i]
            idx_x, idx_y, idx_z = np.where(annotation_d > 0.5)
            sanity_d = []
            for x, y, z in zip(idx_x, idx_y, idx_z):
                sanity_d.append(label_d[x, y, z] == 1)

            sanity.append(not any(sanity_d))

        return not any(sanity)

    def __call__(self, data):
        # load data

        # This is not really nice and could be better with sanitizing input
        if len(self.keys) == 3:
            image, annotation, label = self.keys
            nimg, npnt, nseg = image, annotation, label
        else:
            image = self.keys
            name = image

        d = dict(data)
        image = d[image]
        image_spacings = d[f"{nimg}_meta_dict"]["pixdim"][1:4].tolist()
        print(f"Original Spacing: {image_spacings} \t Target Spacing: {self.target_spacing}")

        if "annotation" in self.keys:
            annotation = d["annotation"]
            annotation[annotation < 0] = 0
        elif "point" in self.keys:
            annotation = d["point"]
            annotation[annotation < 0] = 0

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0
        elif "seg" in self.keys:
            label = d["seg"]
            label[label < 0] = 0
        elif "mask" in self.keys:
            label = d["mask"]
            label[label < 0] = 0

        # calculate shape
        original_shape = image.shape[1:]
        resample_flag = False
        anisotrophy_flag = False

        if self.target_spacing != image_spacings:
            # resample
            resample_flag = True
            spacing_ratio = np.array(image_spacings) / np.array(self.target_spacing)
            resample_shape = self.calculate_new_shape(spacing_ratio, original_shape)
            print(f"Original Shape: {original_shape} \t Target Shape: {resample_shape}")
            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            image = resample_image(image, resample_shape, anisotrophy_flag)

            if "label" in self.keys or "seg" in self.keys or "mask" in self.keys:
                label = resample_label(label, resample_shape, anisotrophy_flag)

            if "annotation" in self.keys:
                annotation = resample_annotation(d["annotation"], d['annotation_meta_dict']["affine"], self.target_spacing, resample_shape)
                if "label" in self.keys or "seg" in self.keys or "mask" in self.keys:
                    d["annotation_meta_dict"]["in_mask"] = self.sanity_in_mask(annotation, label)
                    if d['annotation_meta_dict']["in_mask"] == False:
                        warnings.warn("Annotations are outside of the mask, please fix this")
            elif "point" in self.keys:
                annotation = resample_annotation(d["point"], d['point_meta_dict']["affine"], self.target_spacing, resample_shape)
                if "label" in self.keys or "seg" in self.keys or "mask" in self.keys:
                    d['point_meta_dict']["in_mask"] = self.sanity_in_mask(annotation, label)
                    if d['point_meta_dict']["in_mask"] == False:
                        warnings.warn("Annotations are outside of the mask, please fix this")

        new_meta = {
            "org_spacing": np.array(image_spacings),
            "org_dim": np.array(original_shape),
            "new_spacing": np.array(self.target_spacing),
            "new_dim": np.array(resample_shape),
            "resample_flag": resample_flag,
            "anisotrophy_flag": anisotrophy_flag,
        }

        d[f"{nimg}"] = image
        d[f"{nimg}_meta_dict"].update(new_meta)

        if "annotation" in self.keys:
            d["annotation"] = annotation
            d["annotation_meta_dict"].update(new_meta)
        elif "point" in self.keys:
            d["point"] = annotation
            d["point_meta_dict"].update(new_meta)
            
        if "label" in self.keys:
            d["label"] = label
            d["label_meta_dict"].update(new_meta)
        elif "seg" in self.keys:
            d["seg"] = label
            d["seg_meta_dict"].update(new_meta)
        elif "mask" in self.keys:
            d["mask"] = label
            d["mask_meta_dict"].update(new_meta)

        return d

class EGDMapd(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        image,
        lamb=1,
        iter=4,
        logscale=True,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.image = image
        self.lamb = lamb
        self.iter = iter
        self.logscale = logscale

    def __call__(self, data):
        d = dict(data)
        output = []
        data_type = None
        keys = list(self.key_iterator(d))
        for key in keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])

        for key in self.keys:
            if len(d[key].shape) == 4:
                for idx in range(d[key].shape[0]):
                    GD = GeodisTK.geodesic3d_raster_scan(d[self.image][idx].astype(np.float32), d[key][idx].astype(np.uint8), d[f'{self.image}_meta_dict']["new_spacing"].astype(np.float32), self.lamb, self.iter)
                    if self.logscale == True:
                        d[key][idx, :, :, :] = np.exp(-GD)
            else:
                GD = GeodisTK.geodesic3d_raster_scan(d[self.image].astype(np.float32), d[key].astype(np.uint8), d[f'{self.image}_meta_dict']["new_spacing"].astype(np.float32), self.lamb, self.iter)
                if self.logscale == True:
                    d[key] = np.exp(-GD)
            
        print(f"Geodesic Distance Map with lamd: {self.lamb}, iter: {self.iter} and logscale: {self.logscale}")
        return d

class BoudingBoxd(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        on,
        relaxation=0,
        divisiblepadd=None,
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

    def calculate_bbox(self, data):
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

    def calculate_relaxtion(self, bbox_shape):
        relaxation = [0] * len(bbox_shape)
        for axis in range(len(bbox_shape)):
            relaxation[axis] = math.ceil(bbox_shape[axis] * self.relaxation[axis])

        return relaxation

    def relax_bbox(self, data, bbox, relaxation):
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
        # Remove below zero and higher than shape because of relaxation
        bbox[bbox < 0] = 0
        largest_dimension = [int(x) if  x <= data.shape[i] else data.shape[i] for i, x in enumerate(bbox[1])]
        bbox = np.array([bbox[0].tolist(), largest_dimension])

        if self.divisiblepadd:
            for axis in range(len(self.divisiblepadd)):
                expand = True
                while expand == True:
                    bbox_shape = np.subtract(bbox[1][axis],bbox[0][axis])
                    residue = bbox_shape % self.divisiblepadd[axis]
                    if residue != 0:
                        if residue < 2:
                            neg = bbox[0][axis] - 1
                            if neg >= 0:
                                bbox[0][axis] = neg
                            else:
                                pos = bbox[1][axis] + 1
                                if pos <= data.shape[axis]:
                                    bbox[1][axis] = pos
                                else:
                                    expand = False
                                    warnings.warn(f"bbox doesn't fit in the image for axis {axis}, adding zero padding")
                        else:
                            neg = bbox[0][axis] - 1
                            if neg >= 0:
                                bbox[0][axis] = neg

                            pos = bbox[1][axis] + 1
                            if pos <= data.shape[axis]:
                                bbox[1][axis] = pos

                            if neg <= 0 and pos >= data.shape[axis]:
                                expand = False
                                warnings.warn(f"bbox doesn't fit in the image for axis {axis}, adding zero padding")
                    else:
                        expand = False

        return bbox

    def extract_bbox_region(self, data, bbox):
        new_region = data[
                bbox[0][0]:bbox[1][0],
                bbox[0][1]:bbox[1][1],
                bbox[0][2]:bbox[1][2]
                ]

        return new_region

    def __call__(self, data):
        d = dict(data)
        output = []
        data_type = None
        keys = list(self.key_iterator(d))
        for key in keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])

        bbox = self.calculate_bbox(d[self.on][0])
        bbox_shape = np.subtract(bbox[1],bbox[0])
        relaxation = self.calculate_relaxtion(bbox_shape)

        print(f"Original bouding box at location: {bbox[0]} and {bbox[1]} \t shape of bbox: {bbox_shape}")
        final_bbox = self.relax_bbox(d[self.on][0], bbox, relaxation)
        final_bbox_shape = np.subtract(final_bbox[1],final_bbox[0])
        print(f"Bouding box at location: {final_bbox[0]} and {final_bbox[1]} \t bbox is relaxt with: {relaxation} \t and made divisible with: {self.divisiblepadd} \t shape after cropping: {final_bbox_shape}")
        for key in self.keys:
            if len(d[key].shape) == 4:
                new_dkey = []
                for idx in range(d[key].shape[0]):
                    new_dkey.append(self.extract_bbox_region(d[key][idx], final_bbox))
                d[key] = np.stack(new_dkey, axis=0)
            else:
                d[key] = self.extract_bbox_region(d[key], final_bbox)
            
            d[f"{key}_meta_dict"]["bbox"] = bbox
            d[f"{key}_meta_dict"]["bbox_shape"] = bbox_shape
            d[f"{key}_meta_dict"]["bbox_relaxation"] = self.relaxation
            d[f"{key}_meta_dict"]["final_bbox"] = final_bbox
            d[f"{key}_meta_dict"]["final_bbox_shape"] = final_bbox_shape

        return d

class LoadPreprocessed(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        new_keys,
     ) -> None:
        super().__init__(keys)
        self.keys = keys
        if len(self.keys) != 2:
            raise ValueError(f"LoadPreprocessed data assumes the data has 2 keys with npz and metadata, this is not the case as there are {len(self.keys)} provided")
        
        self.new_keys = new_keys
        self.meta_keys = [x + "_meta_dict" for x in new_keys] + [x + "_transforms" for x in new_keys]

    def read_pickle(self, filename):
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
                    raise KeyError("Old keys and new keys have not te same length in preprocessed data loader")

                if set(old_keys) == set(self.new_keys):
                    for new_key in self.new_keys:
                        new_d[new_key] = image_data[new_key]

                else:
                    warning.warn("old keys do not match new keys, however were the right length so just applying it in order")
                    for old_key, new_key in zip(old_keys, self.new_keys):
                        new_d[new_key] = image_data[old_key]

            elif current_data.suffix == ".pkl":
                metadata = self.read_pickle(d[key])
                old_keys = list(metadata.keys())
                if not len(old_keys) == len(self.meta_keys):
                    raise KeyError("Old keys and new keys have not te same length in preprocessed data loader")

                if set(old_keys) == set(self.meta_keys):
                    for new_key in self.meta_keys:
                        new_d[new_key] = metadata[new_key]

                else:
                    warning.warn("old keys do not match new keys, however were the right length so just applying it in order")
                    for old_key, new_key in zip(old_keys, self.meta_keys):
                        new_d[new_key] = metadata[old_key]
            else:
                raise ValueError("Neither npz or pkl in preprocessed loader")
            
        return new_d