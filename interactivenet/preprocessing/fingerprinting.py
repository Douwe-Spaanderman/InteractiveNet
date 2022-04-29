from typing import List, Tuple
from pathlib import PosixPath

from statistics import median
import os
import numpy as np
import nibabel as nib

class FingerPrint(object):
    def __init__(
        self,
        images: List[PosixPath],
        masks: List[PosixPath],
        annotations: List[PosixPath]
    ) -> None:
        print("Initializing Fingerprinting")
        self.images = images
        self.masks = masks
        self.annotations = annotations
        self.sanity_files()

        self.dim = []
        self.pixdim = []
        self.orientation = []
        self.anisotrophy = []
        self.bbox = []

    def sanity_files(self):
        def check(a,b,c):
            return a == b == c

        len_mask = len(self.masks)
        if not len(self.images) % len_mask == len(self.annotations) % len_mask == 0:
            raise AssertionError("Length of database is not correct, e.g. more masks/annot than images")

        images_names = list(set(['_'.join(x.name.split("_")[:-1]) for x in self.images]))
        masks_names = list(set([x.with_suffix('').stem for x in self.masks]))
        annotations_names = list(set([x.with_suffix('').stem for x in self.annotations]))
        if all([check(a,b,c) for a,b,c in zip(images_names, masks_names, annotations_names)]) == False:
            raise AssertionError("images, masks and annotations do not have the correct names or are not ordered")

    def sanity_same_metadata(self, img, mask, annot):
        def check(a,b,c,all_check=True):
            if all_check == True:
                return np.logical_and((a==b).all(), (b==c).all())
            else:
                return np.logical_and((a==b), (b==c))
            
        if not check(img.affine, mask.affine, annot.affine) or not check(img.shape, mask.shape, annot.shape, False):
            raise AssertionError("Metadata of image, mask and or annotation do not match")

    def sanity_annotation_in_mask(self, mask, annot):
        _check = True
        for inds_x, inds_y, inds_z in np.column_stack((np.where(annot.get_fdata() > 0.5))):
            if not mask.dataobj[inds_x, inds_y, inds_z] == 1:
                _check = False
                warn.warning("Some annotations are not in the mask")

        self.in_mask = _check

    def check_anisotrophy(self, spacing:Tuple[int]):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def resampling_strategy(self, spacing:List[Tuple], anisotropic=False):
        target_spacing = list((median(spacing[0]), median(spacing[1]), median(spacing[2])))

        if anisotropic == True:
            index_max = np.argmax(target_spacing)
            target_spacing[index_max] = np.percentile(spacing[index_max],10)

        return tuple(target_spacing)

    def calculate_bbox(self, data, relaxation):
        inds_x, inds_y, inds_z = np.where(data.get_fdata() > 0.5)

        bbox = np.array([
            [
                np.min(inds_x) - relaxation[0],
                np.min(inds_y) - relaxation[1],
                np.min(inds_z) - relaxation[2]
                ],
            [
                np.max(inds_x) + relaxation[0],
                np.max(inds_y) + relaxation[1],
                np.max(inds_z) + relaxation[2]
            ]
        ])

        # Remove below zero and higher than shape because of relaxation
        bbox[bbox < 0] = 0
        largest_dimension = [int(x) if  x <= data.shape[i] else data.shape[i] for i, x in enumerate(bbox[1])]
        bbox = np.array([bbox[0].tolist(), largest_dimension])

        return bbox

    def calculate_new_shape(self, spacing_ratio, shape):
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def __call__(self):
        print("Starting Fingerprinting: \n")
        print(f"Path: {self.images[0].parents[1]}")
        for img_path, mask_path, annot_path in zip(self.images, self.masks, self.annotations):
            print(
                f"File: {mask_path.with_suffix('').stem}"
            )

            img = nib.load(img_path)
            mask = nib.load(mask_path)
            annot = nib.load(annot_path)
            self.sanity_same_metadata(img, mask, annot)

            self.dim.append(img.shape)
            spacing = img.header.get_zooms()
            self.pixdim.append(spacing)
            self.anisotrophy.append(self.check_anisotrophy(spacing))
            self.orientation.append(nib.orientations.aff2axcodes(img.affine))
            self.sanity_annotation_in_mask(mask, annot)

            bbox = self.calculate_bbox(mask, [10, 10, 2])
            self.bbox.append(bbox[1] - bbox[0])

        keep_dims = self.dim
        self.dim = list(zip(*self.dim))
        self.dim = (median(self.dim[0]), median(self.dim[1]), median(self.dim[2]))

        keep_pixdims = self.pixdim
        self.pixdim = list(zip(*self.pixdim))
        if self.anisotrophy.count(True) >= len(self.anisotrophy) / 2:
            resample_strategy = "Anistropic"
            self.pixdim = self.resampling_strategy(self.pixdim, True)
        else:
            resample_strategy = "Normal"
            self.pixdim = self.resampling_strategy(self.pixdim)

        unique_orientations = list(set(self.orientation))
        if len(unique_orientations) == 1:
            orientation_message = f"All images have the same orientation: {unique_orientations[0]}"
        else:
            from collections import Counter
            unique_orientations = list(Counter(self.orientation).keys())
            orientation_message = f"Not all images have the same orientation, most are {unique_orientations[0]} but some also have {unique_orientations[1:]}\n   Consider adjusting the orientation"

        keep_bbox = self.bbox
        self.bbox = list(zip(*self.bbox))
        self.bbox = (median(self.bbox[0]), median(self.bbox[1]), median(self.bbox[2]))

        spacing_ratios = [np.array(x) / np.array(self.pixdim) for x in keep_pixdims]
        final_shape = [self.calculate_new_shape(x, y) for x, y in zip(spacing_ratios, keep_bbox)]
        self.final_shape = list(zip(*final_shape))
        self.final_shape = (median(self.final_shape[0]), median(self.final_shape[1]), median(self.final_shape[2]))

        print("\nFingeprint:")
        print("- Database Structure: Correct")
        print(f"- All annotions in mask: {self.in_mask}")
        print(f"- Resampling strategy: {resample_strategy}")
        print(f"- All images anisotropic: {all(self.anisotrophy)}")
        print(f"- Target spacing: {self.pixdim}")
        print(f"- Median shape: {self.dim}")
        print(f"- Median shape of bbox: {self.bbox}")
        print(f"- Median shape of bbox after resampling: {self.final_shape} (final shape)")
        print(f"- {orientation_message}")

    def save(self):
        print('Currently not implemented')

if __name__=="__main__":
    from pathlib import Path

    import os
    exp = os.environ["interactiveseg_raw"]
    task = "Task001_Lipo"
    images = [x for x in Path(exp, task, "imagesTr").glob('**/*') if x.is_file()]
    masks = [x for x in Path(exp, task, "labelsTr").glob('**/*') if x.is_file()]
    annotations = [x for x in Path(exp, task, "interactionTr").glob('**/*') if x.is_file()]
    results = FingerPrint(sorted(images), sorted(masks), sorted(annotations))
    results()