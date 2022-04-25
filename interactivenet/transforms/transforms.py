from monai.transforms.transform import MapTransform
from skimage.transform import resize
import numpy as np
import GeodisTK
from nibabel import affines
import numpy.linalg as npl

from interactivenet.utils.resample import resample_image, resample_label, resample_point

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

    def sanity_in_mask(self, point, label):
        sanity = []
        for i, point_d in enumerate(point):
            label_d = label[i]
            idx_x, idx_y, idx_z = np.where(point_d > 0.5)
            sanity_d = []
            for x, y, z in zip(idx_x, idx_y, idx_z):
                sanity_d.append(label_d[x, y, z] == 1)

            sanity.append(not any(sanity_d))

        return not any(sanity)

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
        image_spacings = d[f"{nimg}_meta_dict"]["pixdim"][1:4].tolist()

        if "point" in self.keys:
            point = d["point"]
            point[point < 0] = 0

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
            spacing_ratio = np.array(image_spacings) / np.array(self.target_spacing)
            resample_shape = self.calculate_new_shape(spacing_ratio, original_shape)
            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            image = resample_image(image, resample_shape, anisotrophy_flag)

            if "label" in self.keys or "seg" in self.keys:
                label = resample_label(label, resample_shape, anisotrophy_flag)

            if "point" in self.keys:
                point = resample_point(data["point"], data['point_meta_dict']["affine"], self.target_spacing, resample_shape)
                if "label" in self.keys or "seg" in self.keys:
                    d["points_in_mask"] = self.sanity_in_mask(point, label)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag

        new_meta = {
            "new_spacing": np.array(self.target_spacing),
            "new_dim": np.array(resample_shape)
        }

        d[f"{nimg}"] = image
        d[f"{nimg}_meta_dict"].update(new_meta)

        if "point" in self.keys:
            d["point"] = point
            d["point_meta_dict"].update(new_meta)

        if "label" in self.keys:
            d["label"] = label
            d["label_meta_dict"].update(new_meta)

        elif "seg" in self.keys:
            d["seg"] = label
            d["seg_meta_dict"].update(new_meta)

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
                    GD = GeodisTK.geodesic3d_raster_scan(d[self.image][idx].astype(np.float32), d[key][idx].astype(np.uint8), d[f'{self.image}_meta_dict']["new_spacing"].astype(np.float32), self.lamb, self.iter)
                    d[key][idx, :, :, :] = np.exp(-GD)
            else:
                GD = GeodisTK.geodesic3d_raster_scan(d[self.image].astype(np.float32), d[key].astype(np.uint8), d[f'{self.image}_meta_dict']["new_spacing"].astype(np.float32), self.lamb, self.iter)
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
        inds_x, inds_y, inds_z = np.where(data > 0.5)

        bbox = np.array([
            [
                np.min(inds_x) - self.relaxation[0],
                np.min(inds_y) - self.relaxation[1],
                np.min(inds_z) - self.relaxation[2]
                ],
            [
                np.max(inds_x) + self.relaxation[0],
                np.max(inds_y) + self.relaxation[1],
                np.max(inds_z) + self.relaxation[2]
            ]
        ])

        # Remove below zero and higher than shape because of relaxation
        bbox[bbox < 0] = 0
        largest_dimension = [int(x) if  x <= data.shape[i] else data.shape[i] for i, x in enumerate(bbox[1])]
        bbox = np.array([bbox[0].tolist(), largest_dimension])

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
        for key in self.keys:
            if len(d[key].shape) == 4:
                new_dkey = []
                for idx in range(d[key].shape[0]):
                    new_dkey.append(self.extract_bbox_region(d[key][idx], bbox))
                d[key] = np.stack(new_dkey, axis=0)
            else:
                d[key] = self.extract_bbox_region(d[key], bbox)
            
            d[f"{key}_meta_dict"]["bbox"] = bbox

        return d
