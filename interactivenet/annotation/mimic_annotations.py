import os
import random
import itertools
import warnings
import json
import argparse
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from scipy import ndimage

from interactivenet.utils.visualization import myshow3d

class MaskItem(object):
    def __init__(self, Mask, Image):
        self.MaskName = Mask.name
        self.ImageName = Image.name
        self.AnnotationName = Mask.name
        self.Mask = self._from_simpleITK(self._read_Image(Mask))
        self.Image = self._read_Image(Image)
        self.Center = ndimage.center_of_mass(self.Mask)
        self.Spacing = self.Image.GetSpacing()
        self.Origin = self.Image.GetOrigin()
        self.Direction = self.Image.GetDirection()
        self.Image = self._from_simpleITK(self.Image)
        self.Dimensions = self.Mask.shape
        self.inds_z, self.inds_y, self.inds_x = np.where(self.Mask > 0.5)
        self.Cropped = False
        self.RandomPoints = None
        self.ExtremePoints = None
        self.CenterPoints = None
        self.BoudingBox = None
        self.NewMask = None
        self.ChangedMask = np.copy(self.Mask)
        self.ShowMask = np.copy(self.Mask)

    def _from_simpleITK(self, img):
        if img is not None:
            return sitk.GetArrayFromImage(img)

    def _to_simpleITK(self, img):
        if img is not None:
            img = sitk.GetImageFromArray(img)
            img.SetSpacing(self.Spacing)
            img.SetOrigin(self.Origin)
            img.SetDirection(self.Direction)
            return img

    def _read_Image(self, img):
        if img:
            return sitk.ReadImage(str(img), imageIO="NiftiImageIO")

    def _save_Image(self, img, name):
        if img:
            return sitk.WriteImage(img, name)

    def _assert_pad(self, used, pad=0):
        if type(pad) == int:
            pad_z, pad_y, pad_x = pad, pad, pad
        elif len(pad) == 3:
            pad_z, pad_y, pad_x = pad
        else:
            raise KeyError(f"{used} should either be an int or list of 3 int, not {pad}")

        return pad_z, pad_y, pad_x

    def _assert_in_bbox(self, points):
        if self.Cropped == True:
            raise KeyError("Unable to assert in bbox as already cropped to bbox, please crop afterwards")
        else:
            if self.BoudingBox is None:
                self.get_bbox(pad=[1,3,3])
                warnings.warn("Bounding box was created on the fly with padding = [1,3,3]")

            for point in points:
                if not (self.BoudingBox[0][0] <= point[0] <= self.BoudingBox[1][0]) or not (self.BoudingBox[0][1] <= point[1] <= self.BoudingBox[1][1]) or not (self.BoudingBox[0][2] <= point[2] <= self.BoudingBox[1][2]):
                    raise KeyError("Points do not fit in bouding box")

    def _find_point(self, id_z, id_y, id_x, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_z[sel_id], id_y[sel_id], id_x[sel_id]]

    def find_border(self, iterations=1) -> None:
        matrix = np.copy(self.ChangedMask)
        matrix = ndimage.binary_erosion(matrix, iterations=iterations).astype(matrix.dtype)
        self.ChangedMask = self.ChangedMask - matrix

    def add_border(self, iterations=1) -> None:
        matrix = np.copy(self.ChangedMask)
        matrix = ndimage.binary_dilation(matrix, iterations=iterations).astype(matrix.dtype)
        self.ChangedMask = matrix - self.ChangedMask
        self.inds_z, self.inds_y, self.inds_x = np.where(self.ChangedMask > 0.5)

    def get_bbox(self, pad=0) -> None:
        pad_z, pad_y, pad_x = self._assert_pad("bbox", pad)

        self.BoudingBox = np.array([
            [
                np.min(self.inds_z) - pad_z,
                np.min(self.inds_y) - pad_y,
                np.min(self.inds_x) - pad_x
                ],
            [
                np.max(self.inds_z) + pad_z,
                np.max(self.inds_y) + pad_y,
                np.max(self.inds_x) + pad_x
            ]
        ])

        # Check all dimensions - remove to small (below zero) and to large (above dimensions)
        self.BoudingBox[self.BoudingBox < 0] = 0
        largest_dimension = [int(x) if  x <= self.Dimensions[i] else self.Dimensions[i] for i, x in enumerate(self.BoudingBox[1])]
        self.BoudingBox = np.array([self.BoudingBox[0].tolist(), largest_dimension])

    def crop_from_bbox(self):
        self.ChangedMask = self.ChangedMask[
            self.BoudingBox[0][0]:self.BoudingBox[1][0],
            self.BoudingBox[0][1]:self.BoudingBox[1][1],
            self.BoudingBox[0][2]:self.BoudingBox[1][2]
            ]
        self.ShowMask = self.ShowMask[
            self.BoudingBox[0][0]:self.BoudingBox[1][0],
            self.BoudingBox[0][1]:self.BoudingBox[1][1],
            self.BoudingBox[0][2]:self.BoudingBox[1][2]
            ]

        if self.Image is not None:
            self.Image = self.Image[
                self.BoudingBox[0][0]:self.BoudingBox[1][0],
                self.BoudingBox[0][1]:self.BoudingBox[1][1],
                self.BoudingBox[0][2]:self.BoudingBox[1][2]
                ]

        self.Cropped = True
        self.Dimensions = self.Mask.shape
        self.inds_z, self.inds_y, self.inds_x = np.where(self.Mask > 0.5)

        if self.NewMask is not None:
            warnings.warn("Changing the new mask to fit bounding box, please only use for visualization")
            self.NewMask = self.NewMask[
                self.BoudingBox[0][0]:self.BoudingBox[1][0],
                self.BoudingBox[0][1]:self.BoudingBox[1][1],
                self.BoudingBox[0][2]:self.BoudingBox[1][2]
                ]
            
            return self.NewMask

    def points_in_mask(self, points, move):
        _error = False
        correct_points = np.empty(shape=(6,3), dtype=int)
        _indexinfo = [(0,-1),(0,1),(1,-1),(1,1),(2,-1),(2,1)] # this is needed
        for i, point in enumerate(points):
            _error = 0
            while True:
                if self.Mask[point[0], point[1], point[2]] != 1:
                    tmpinfo = _indexinfo[i]
                    _error += tmpinfo[1]
                    point[tmpinfo[0]] = point[tmpinfo[0]] + tmpinfo[1]
                    
                    if abs(_error) > move[tmpinfo[0]]:
                        raise KeyError("move is to big for points in masks")
                else:
                    correct_points[i] = point
                    break

        if _error != False:
            warnings.warn(f"Sample was created more outwards, because it dind't match mask ({_error})")

        return correct_points

    def extreme_points(self, move_internal=0):
        move_z, move_y, move_x = self._assert_pad("extreme_points", move_internal)

        self.ExtremePoints = np.array(
            [
                self._find_point(self.inds_z+move_z, self.inds_y, self.inds_x, np.where(self.inds_z <= np.min(self.inds_z))), # Z-low
                self._find_point(self.inds_z-move_z, self.inds_y, self.inds_x, np.where(self.inds_z >= np.max(self.inds_z))), # Z-high
                self._find_point(self.inds_z, self.inds_y+move_y, self.inds_x, np.where(self.inds_y <= np.min(self.inds_y))), # bottom
                self._find_point(self.inds_z, self.inds_y-move_y, self.inds_x, np.where(self.inds_y >= np.max(self.inds_y))), # top
                self._find_point(self.inds_z, self.inds_y, self.inds_x+move_x, np.where(self.inds_x <= np.min(self.inds_x))), # left
                self._find_point(self.inds_z, self.inds_y, self.inds_x-move_x, np.where(self.inds_x >= np.max(self.inds_x))) # right
            ])

        self._assert_in_bbox(self.ExtremePoints)
        self.ExtremePoints = self.points_in_mask(self.ExtremePoints, move=(move_z, move_y, move_x))

        return self.ExtremePoints

    def random_points(self, move=0, n=1):
        move_z, move_y, move_x = self._assert_pad("random_points", move)
        random_points = np.random.choice(len(self.inds_z), n, replace=False)

        self.RandomPoints = np.array(
            [
                [self.inds_z[x]+move_z, self.inds_y[x]+move_y, self.inds_x[x]+move_x] for x in random_points
            ])

        self._assert_in_bbox(self.RandomPoints)

        return self.RandomPoints

    def center_point(self):
        self.CenterPoints = np.array(
            [
                np.rint(self.Center).astype(int)
            ])

        self._assert_in_bbox(self.CenterPoints)

        return self.CenterPoints

    def combine_to_map(self, points=[], label=1):
        self.NewMask = np.zeros(shape=self.Dimensions)
        if label == 1:
            for point_map in points:
                for point in point_map:
                    #print(point)
                    self.NewMask[point[0], point[1], point[2]] = 1
        else:
            for i, point_map in enumerate(points):
                for point in point_map:
                    self.NewMask[point[0], point[1], point[2]] = label[i]

        return self.NewMask

    def overlay_points(self, overlap, color=[3]) -> None:
        for i, point_map in enumerate(overlap):
            for point in point_map:
                self.ShowMask[point[0], point[1], point[2]] = color[i]
        
    def show(self, overlap=None, xslices=[], yslices=[], zslices=[], enlarge=10, title=None, margin=0.05, dpi=300, show=True, save=None, gif=False) -> None:
        myshow3d(self.ShowMask, self.Image, overlap, xslices, yslices, zslices, enlarge, title, margin, dpi, show, save, gif)

    def save(self, location, mode) -> None:
        self._save_Image(self._to_simpleITK(self.Mask), str(location / f"labels{mode}" / self.MaskName))
        self._save_Image(self._to_simpleITK(self.Image), str(location / f"images{mode}" / self.ImageName))
        self._save_Image(self._to_simpleITK(self.NewMask), str(location / f"interaction{mode}" / self.AnnotationName))

        # Uncomment processed mask for dev
        #self._save_Image(self._to_simpleITK(self.ChangedMask), str(location / "processed_mask.nii.gz"))

def create_sample(input_mask, input_image=None, border=None, extreme_points=None, random_points=None, center_point=None, scribble=False, mode="Tr", save=None, plot=None, gif=False):
    data = MaskItem(input_mask, input_image)

    data.get_bbox(pad=[1,3,3])

    if border:
        data.find_border()

    overlap = []
    if extreme_points:
        overlap.append(data.extreme_points(move_internal=[int(x) for x in extreme_points]))

    if random_points:
        overlap.append(data.random_points(n=int(random_points)))

    if center_point:
        overlap.append(data.center_point())

    newmask = data.combine_to_map(overlap)
    # Saving data
    if save:
        for folder in [f"images{mode}", f"interaction{mode}", f"labels{mode}"]:
            tmp = save / folder
            tmp.mkdir(parents=True, exist_ok=True)

        data.save(save, mode=mode)
    
    if plot and save:
        newmask = data.crop_from_bbox()
        save = save / "images"
        save.mkdir(parents=True, exist_ok=True)
        # Plot image with mask
        data.show(zslices=list(range(0,newmask.shape[0],1)), show=False, save=str(save / "original"), gif=gif)
        # Plot image with newmask
        data.show(overlap=newmask, zslices=list(range(0,newmask.shape[0],1)), show=False, save=str(save / "newimage"), gif=gif)

        # Plot mask with newmask
        data.Image = None
        data.show(overlap=newmask, zslices=list(range(0,newmask.shape[0],1)), show=False, save=str(save / "newmask"), gif=gif)
    elif gif:
        raise KeyError("Can't use gif true and save false, gifs have to be saved")
    elif plot:
        newmask = data.crop_from_bbox()
        # Plot image with mask
        data.show(zslices=list(range(0,newmask.shape[0],1)), show=True, gif=gif)
        # Plot image with newmask
        data.show(overlap=newmask, zslices=list(range(0,newmask.shape[0],1)), show=True, gif=gif)

        # Plot mask with newmask
        data.Image = None
        data.show(overlap=newmask, zslices=list(range(0,newmask.shape[0],1)), show=True, gif=gif)

def main():
    parser = argparse.ArgumentParser(
		description="Create annotations from masks, in order to do minimally interactive segmentation experiments"
		)
    parser.add_argument(
        "-i",
        "--inpath",
        nargs="?",
        default="data/raw/LIPO/",
        help="Path to input folder"
        )
    parser.add_argument(
        "-o",
        "--outpath",
        nargs="?",
        default="data/exp/",
        help="Path to store output file"
        )
    parser.add_argument(
        "-n",
        "--name",
        nargs="?",
        default="Lipo",
        help="Name of task"
        )
    parser.add_argument(
        "-b",
        "--border",
        nargs="?",
        default=None,
        help="Do you want to look in the border only"
        )
    parser.add_argument(
        "-e",
        "--extreme_points",
        nargs="+",
        default=None,
        help="Do you want to get extreme points"
        )
    parser.add_argument(
        "-p",
        "--random_points",
        nargs="?",
        default=None,
        help="Do you want to get random points, if so how many"
        )
    parser.add_argument(
        "-c",
        "--center_point",
        nargs="?",
        default=None,
        help="Do you want to get the center point"
        )
    parser.add_argument(
        "-s",
        "--plot",
        nargs="?",
        default=None,
        help="Do you want to plot the images"
        )
    parser.add_argument(
        "-g",
        "--gif",
        nargs="?",
        default=None,
        help="Do you want to plot the images as gifs"
        )
    parser.add_argument(
        "-m",
        "--mask_name",
        nargs="?",
        default="masks",
        help="How are the masks called (can be masks or segmentation"
        )
    args = parser.parse_args()

    # This is stupid but whatever
    if args.extreme_points and len(args.extreme_points) != 3:
        raise KeyError(f"argument extreme_points (-e) should either be None or a list of 3 not: {args.extreme_points}")
    
    output_folder = Path(args.outpath, args.name)
    Dataset_numb = 0
    for mode in ["Tr", "Ts"]:
        labels = sorted([f for f in Path(args.inpath, "labels" + mode).glob('**/*') if f.is_file()])
        images = sorted([f for f in Path(args.inpath, "images" + mode).glob('**/*') if f.is_file()])

        for label, image in zip(labels, images):
            create_sample(
                input_mask = label,
                input_image = image,
                border = args.border,
                extreme_points = args.extreme_points,
                random_points = args.random_points,
                center_point = args.center_point,
                scribble=False,
                mode = mode,
                save = output_folder,
                plot = args.plot,
                gif = args.gif,
            )

        Dataset_numb += len(images)

    if output_folder:
        Dataset_name = Path(args.inpath).name
        with open(str(output_folder / 'metadata.json'), 'w') as f:
            json.dump({
                "Info" : "Metadata for scribble/point minimal interactive segmentation. Point have been drawn automatically using the mask of the image",
                "Dataset" : {
                    "Name" : f"{Dataset_name}",
                    "Number" : f"{Dataset_numb}"
                },
                "Output": {
                    "image": "file with the unchanged image (.nii.gz)",
                    "masks": "file with the unchanged mask (.nii.gz)",
                    "points": "file with points drawn using this experiment (.nii.gz)",
                },
                "Arguments" : {
                    "border" : f"{args.border}",
                    "extreme_points": f"{args.extreme_points}",
                    "random_points": f"{args.random_points}",
                    "center_point": f"{args.center_point}",
                    "plot": f"{args.plot}",
                    "gif": f"{args.gif}",
                },
                "Explanation" : {
                    "border" : f"If not None, all points have been drawn from the border. Otherwise the can be drawn from anywhere in the mask",
                    "extreme_points": f"If not None, extreme points are found for the mask, which in a 3D image are 6 points, i.e. two extremes for each axis. Also the list provided, e.g. [1,3,3] ([z,y,x]), specifies the relaxtion to the middle of the extreme points. This can be done in order to mimic errors from the clinicians, i.e. they won't find the optimal extreme points",
                    "random_points": f"If not None, random points are drawn from the mask or border (see argument border). Will be type(int) specifying the number of random points drawn",
                    "plot": f"If not None, this means that images are saved",
                    "gif": f"If not None, this means that images are saved as gifs",
                },
            }, f, indent=4)

if __name__ == "__main__":
    main()
