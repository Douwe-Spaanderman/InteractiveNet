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

from typing import Union, Dict

import pandas as pd
import numpy as np
import torch

from monai.metrics import (
    compute_meandice,
    compute_average_surface_distance,
    compute_hausdorff_distance,
)

from interactivenet.utils.utils import to_torch, to_sitk

import seaborn as sns
import matplotlib.pyplot as plt

from radiomics.shape import RadiomicsShape


def CalculateScores(
    pred: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    include_background: bool = False,
):
    if isinstance(pred, np.ndarray):
        pred = to_torch(pred)

    if isinstance(mask, np.ndarray):
        mask = to_torch(mask)

    pred_shape = pred.shape
    mask_shape = mask.shape

    if not pred_shape == mask_shape:
        raise ValueError(
            f"Please provide equal sized tensors for comparing predictions and grounth truth, not {pred_shape} and {mask_shape}"
        )

    if len(pred_shape) == 3:
        if include_background == True:
            print("adding empty channel HWD -> CHWD")
            pred = pred[None, :]
            mask = mask[None, :]
            pred_shape = pred.shape
        else:
            raise ValueError(
                f"Predictions have shape {pred} and background == False, so should include atleast 2 channels (CHWD)"
            )

    if len(pred_shape) == 4:
        print("adding empty batch CHWD -> BCHWD")
        pred = pred[None, :]
        mask = mask[None, :]
    elif len(pred_shape) != 5:
        raise ValueError(
            f"Unrecognized number of channels {pred_shape}, either provide HWD, CHWD or BCHWD"
        )

    dice = compute_meandice(pred, mask, include_background=include_background)
    hausdorff_distance = compute_hausdorff_distance(
        pred, mask, include_background=include_background
    )
    surface_distance = compute_average_surface_distance(
        pred, mask, include_background=include_background
    )

    return dice, hausdorff_distance, surface_distance


def CalculateClinicalFeatures(
    image: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    meta: Dict,
):
    if pred.shape != mask.shape:
        raise ValueError(
            f"Please provide equal sized arrays for comparing predictions and grounth truth, not {pred.shape} and {mask.shape}"
        )

    if len(image.shape) != len(pred.shape) or len(image.shape) != len(mask.shape):
        raise ValueError(
            f"Please provide equal sized arrays for comparing predictions, grounth truth, and image, not {pred.shape}, {mask.shape} and {image.shape}"
        )

    if len(image.shape) == 4:
        image = image[0]
        pred = pred[1]
        mask = mask[1]

    image = to_sitk(image, meta)
    pred = to_sitk(pred, meta)
    mask = to_sitk(mask, meta)

    features_pred = RadiomicsShape(image, pred)
    features_gt = RadiomicsShape(image, mask)

    diameters_pred = {
        "Slice (axial)": features_pred.getMaximum2DDiameterSliceFeatureValue(),
        "Column (coronal)": features_pred.getMaximum2DDiameterColumnFeatureValue(),
        "Row (sagittal)": features_pred.getMaximum2DDiameterRowFeatureValue(),
    }

    diameters_gt = {
        "Slice (axial)": features_gt.getMaximum2DDiameterSliceFeatureValue(),
        "Column (coronal)": features_gt.getMaximum2DDiameterColumnFeatureValue(),
        "Row (sagittal)": features_gt.getMaximum2DDiameterRowFeatureValue(),
    }

    return (
        features_pred.getMeshVolumeFeatureValue(),
        features_gt.getMeshVolumeFeatureValue(),
        diameters_pred,
        diameters_gt,
    )


def ResultPlot(data, scorename="Dice", types=False, unseen=False):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    fig, ax = plt.subplots()

    data = pd.DataFrame.from_dict(data, orient="index")
    data.columns = [scorename]
    data["Names"] = data.index
    data = data.reset_index()

    if types:
        data["Types"] = data["Names"].map(types)
        if unseen:
            data["Seen in training"] = data["Types"].map(unseen)
            data.loc[data["Seen in training"] != False, "Seen in training"] = True
            sns.boxplot(
                x="Types",
                y=scorename,
                hue="Seen in training",
                dodge=False,
                data=data,
                ax=ax,
            )
        else:
            sns.boxplot(x="Types", y=scorename, data=data, ax=ax)
    else:
        sns.boxplot(y=scorename, data=data, ax=ax)

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig


def ComparePlot(data, hue=False):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    fig, ax = plt.subplots()

    data = pd.DataFrame.from_dict(data, orient="index")
    data.columns = ["GT", "Pred"]
    data["Names"] = data.index
    data = data.reset_index()

    if hue:
        sns.scatterplot(x="GT", y="Pred", data=data, linewidth=0, ax=ax)
    else:
        sns.scatterplot(x="GT", y="Pred", data=data, linewidth=0, ax=ax)

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig
