from typing import Union, Dict

import json
from pathlib import Path, PosixPath

import numpy as np
import torch

import nibabel as nib

def read_nifti(data:Dict, test:bool=False):
    loaded_data = {}
    for idx in data:
        idx_data = {}
        img = nib.load(idx["image"])
        name = "_".join(idx["image"].name.split('.')[0].split("_")[:-1])
        idx_data["image"] = img.get_fdata()
        idx_data["image_meta_dict"] = img.header
        if not test:
            masks = nib.load(idx["mask"])
            idx_data["masks"] = masks.get_fdata()
            idx_data["masks_meta_dict"] = masks.header

        loaded_data.update({name: idx_data})

    return loaded_data

def read_processed(datapath:Union[str, PosixPath]):
    if isinstance(datapath, str):
        datapath = Path(datapath)

    arrays = sorted([x for x in (datapath / "network_input").glob('**/*.npz') if x.is_file()])
    metafile = sorted([x for x in (datapath / "network_input").glob('**/*.pkl') if x.is_file()])

    if len(arrays) != len(metafile):
        raise ValueError("not the same number files for arrays and metafile")

    return [
            {"npz": npz_path, "metadata": metafile_path}
            for npz_path, metafile_path in zip(arrays, metafile)
        ]

def read_data(datapath:Union[str, PosixPath], test:bool=False):
    if isinstance(datapath, str):
        datapath = Path(datapath)

    images = sorted([x for x in (datapath / "imagesTs").glob('**/*') if x.is_file()])
    annotations = sorted([x for x in (datapath / "interactionTs").glob('**/*') if x.is_file()])

    if len(images) != len(annotations):
        raise ValueError("not the same number files for images and annotations")

    if not test:
        masks = sorted([x for x in (datapath / "labelsTs").glob('**/*') if x.is_file()])
        if not len(images) == len(masks) == len(annotations):
            raise ValueError("not the same number files for images, masks and annotations")
        
        return [
            {"image": img_path, "mask": mask_path, "annotation": annot_path}
            for img_path, mask_path, annot_path in zip(images, masks, annotations)
        ]
        
    else:
        return [
            {"image": img_path, "annotation": annot_path}
            for img_path, annot_path in zip(images, annotations)
        ]

def read_metadata(metapath:Union[str, PosixPath]):
    if isinstance(metapath, str):
        metapath = Path(metapath)

    if metapath.is_file():
        with open(metapath) as f:
            return json.load(f)
    else:
        raise KeyError(f"metadata does not exist at path: {metapath}")

def read_types(typespath:Union[str, PosixPath]):
    if isinstance(typespath, str):
        typespath = Path(typespath)

    if typespath.is_file():
        with open(typespath) as f:
            types = json.load(f)
            return {v: key for key, value in types.items() for v in value}
    else:
        raise KeyError(f"types does not exist at path: {typespath}")

def to_array(data:Union[np.ndarray, torch.Tensor]):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    return data

def to_torch(data:Union[np.ndarray, torch.Tensor]):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    return data