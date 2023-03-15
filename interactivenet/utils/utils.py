from typing import List, Tuple, Dict, Sequence, Optional, Callable, Union

import os
import json
from pathlib import Path

import numpy as np
import torch

import nibabel as nib
import SimpleITK as sitk

import mlflow
import uuid

import pickle

from monai.transforms import AsDiscrete

def save_niftis(mlflow, outputs:list, postprocessing:str):
    argmax = AsDiscrete(argmax=True)
    tmp_dir = Path("/tmp/", str(uuid.uuid4()))
    print(f"saving niftis to {tmp_dir} before moving to artifacts.")
    for output in outputs:
        name = Path(output[1][0]["filename_or_obj"]).name.split('.')[0]
        pred = output[0][0]
        meta = output[1][0]

        pred = argmax(pred)
        pred = ApplyPostprocessing(pred, postprocessing)
        
        tmp_dir.mkdir(parents=True, exist_ok=True)
        data_file = tmp_dir / f"{name}.nii.gz"

        output = nib.Nifti1Image(pred, meta["affine"])
        mlflow.log_artifact(str(data_file), artifact_path="niftis")
        data_file.unlink()
    
    tmp_dir.rmdir()

def save_weights(mlflow, outputs:list):
    tmp_dir = Path("/tmp/", str(uuid.uuid4()))
    print(f"saving weights to {tmp_dir} before moving to artifacts.")
    for output in outputs:
        name = Path(output[1][0]["filename_or_obj"]).name.split('.')[0]
        weights = output[0][0]
        meta = output[1][0]

        tmp_dir.mkdir(parents=True, exist_ok=True)
        data_file = tmp_dir / f"{name}.npz"

        np.savez(str(data_file), weights=weights)
        mlflow.log_artifact(str(data_file), artifact_path="weights")
        data_file.unlink()

        data_file = tmp_dir / f"{name}.pkl"
        with open(str(data_file), 'wb') as handle:
            pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        mlflow.log_artifact(str(data_file), artifact_path="weights")
        data_file.unlink()
    
    tmp_dir.rmdir()

def read_pickle(datapath:Union[str, os.PathLike]):
    datapath = to_pathlib(datapath)

    objects = []
    with (open(datapath, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    return objects

def read_dataset(datapath:Union[str, os.PathLike], mode="train", error_message=None):
    datapath = to_pathlib(datapath)

    datapath = datapath / "dataset.json"

    if datapath.is_file():
        with open(datapath) as f:
            dataset = json.load(f)
            return dataset[mode], dataset["modality"]
    else:
        if error_message:
            raise KeyError(error_message)
        else:
            raise KeyError(f"dataset.json does not exist at path: {datapath}")

def read_nifti(data:Dict, raw_path:Optional[Union[str, os.PathLike]], rename_image:Optional[str]="image_raw"):
    labels = all([x["label"] != "" for x in data])

    raw_path = to_pathlib(raw_path)
    loaded_data = {}
    for idx in data:
        idx_data = {}
        name = Path(idx["interaction"]).name.split(".")[0]
        
        img = nib.load(raw_path / idx["image"])
        if rename_image:
            idx_data[rename_image] = [img.get_fdata()[None, :]] # Adding a channel and in list to match batch output
        else:
            idx_data["image"] = [img.get_fdata()[None, :]]
            
        idx_data["image_meta_dict"] = [img.header]

        inter = nib.load(raw_path / idx["interaction"])
        idx_data["interaction"] = [inter.get_fdata()[None, :]]
        idx_data["interaction_meta_dict"] = [inter.header]

        idx_data["class"] = [idx["class"]]
        
        if labels:
            label = nib.load(raw_path / idx["label"])
            idx_data["label"] = [label.get_fdata()[None, :]]
            idx_data["label_meta_dict"] = [label.header]

        loaded_data.update({name: idx_data})

    return loaded_data, labels

def read_processed(datapath:Union[str, os.PathLike]):
    datapath = to_pathlib(datapath)

    arrays = sorted([x for x in (datapath / "network_input").glob('**/*.npz') if x.is_file()])
    metafile = sorted([x for x in (datapath / "network_input").glob('**/*.pkl') if x.is_file()])

    if len(arrays) != len(metafile):
        raise ValueError("not the same number files for arrays and metafile")

    return [
            {"npz": npz_path, "metadata": metafile_path}
            for npz_path, metafile_path in zip(arrays, metafile)
        ]

def read_metadata(metapath:Union[str, os.PathLike], error_message=None):
    metapath = to_pathlib(metapath)

    if metapath.is_file():
        with open(metapath) as f:
            return json.load(f)
    else:
        if error_message:
            raise KeyError(error_message)
        else:
            raise KeyError(f"metadata does not exist at path: {metapath}")

def read_types(typespath:Union[str, os.PathLike]):
    typespath = to_pathlib(typespath)

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

def to_sitk(data:Union[np.ndarray, torch.Tensor], meta:Dict):
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    else:
        raise KeyError(f"please provide array or tensor not: {type(data)}")

    data = sitk.GetImageFromArray(data, isVector=False)
    data.SetSpacing(np.array(meta["pixdim"][1:4], dtype='float32').tolist())
    return data

def to_pathlib(datapath:Union[str, os.PathLike]):
    if isinstance(datapath, str):
        datapath = Path(datapath)
    
    return datapath

def check_gpu():
    if torch.cuda.is_available():
        print("Using GPU!")
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Now on device: {current_device} which is a {device_name}")
        return "gpu", -1, 16
    else:
        print("YOU ARE CURRENTLY RUNNING WITHOUT GPU, THIS IS EXTREMELY SLOW!")
        return "cpu", None, "bf16"