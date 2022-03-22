from monai.transforms.transform import MapTransform
from monai.transforms import (
    NormalizeIntensity,
)
from skimage.transform import resize
import numpy as np
import GeodisTK

def resample_label(label, shape, anisotrophy_flag):
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

def resample_image(image, shape, anisotrophy_flag):
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

class PreprocessAnisotropic(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        clip_values,
        pixdim,
        normalize_values,
        model_mode,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.low = clip_values[0]
        self.high = clip_values[1]
        self.target_spacing = pixdim
        self.mean = normalize_values[0]
        self.std = normalize_values[1]
        self.training = False
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)
        if model_mode in ["train"]:
            self.training = True

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def __call__(self, data):
        # load data
        if len(self.keys) == 3:
            image, point, label = self.keys
            nimg, npnt, nseg = image, point, label
        else:
            image = self.keys
            name = image

        d = dict(data)
        image = d[image]
        point = d[point]
        image_spacings = d[f"{nimg}_meta_dict"]["pixdim"][1:4].tolist()

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0

        if "seg" in self.keys:
            label = d["seg"]
            label[label < 0] = 0

        # calculate shape
        original_shape = image.shape[1:]
        resample_flag = False
        anisotrophy_flag = False

        if self.target_spacing != image_spacings:
            # resample
            resample_flag = True
            resample_shape = self.calculate_new_shape(image_spacings, original_shape)
            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            image = resample_image(image, resample_shape, anisotrophy_flag)
            point = resample_image(point, resample_shape, anisotrophy_flag)


            if "label" in self.keys:
                label = resample_label(label, resample_shape, anisotrophy_flag)

            if "seg" in self.keys:
                label = resample_label(label, resample_shape, anisotrophy_flag)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag
        # clip image for CT dataset
        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d[f"{nimg}"] = image

        if "label" in self.keys:
            d["label"] = label

        if "seg" in self.keys:
            d["seg"] = label

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
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.image = image
        self.lamb = lamb
        self.iter = iter

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
                    GD = GeodisTK.geodesic3d_raster_scan(d[self.image][idx].astype(np.float32), d[key][idx].astype(np.uint8), d[f'{self.image}_meta_dict']["dim"][1:4], self.lamb, self.iter)
                    d[key][idx, :, :, :] = np.exp(-GD)
            else:
                GD = GeodisTK.geodesic3d_raster_scan(d[self.image].astype(np.float32), d[key].astype(np.uint8), d[f'{self.image}_meta_dict']["dim"][1:4], self.lamb, self.iter)
                d[key] = np.exp(-GD)

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
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.on = on
        self.relaxation = relaxation

        if len(self.relaxation) == 1:
            self.relaxation = [self.relaxation] * 3

    def calculate_bbox(self, data):
        inds_z, inds_y, inds_x = np.where(data > 0.5)
        bbox = np.array([
            [
                np.min(inds_z) - self.relaxation[2],
                np.min(inds_y) - self.relaxation[1],
                np.min(inds_x) - self.relaxation[0]
                ],
            [
                np.max(inds_z) + self.relaxation[2],
                np.max(inds_y) + self.relaxation[1],
                np.max(inds_x) + self.relaxation[0]
            ]
        ])
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

        for key in self.keys:
            if len(d[key].shape) == 4:
                print(d[key].shape)
                new_dkey = []
                bbox = self.calculate_bbox(d[self.on][0])
                for idx in range(d[key].shape[0]):
                    new_dkey.append(self.extract_bbox_region(d[key][idx], bbox))
                d[key] = np.stack(new_dkey, axis=0)
                print(d[key].shape)
            else:
                bbox = self.calculate_bbox(d[self.on][0])
                d[key] = self.extract_bbox_region(d[key], bbox)

        return d
