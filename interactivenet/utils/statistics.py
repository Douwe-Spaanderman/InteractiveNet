from typing import Union

import pandas as pd
import numpy as np
import torch

from monai.metrics import compute_meandice, compute_average_surface_distance, compute_hausdorff_distance

from interactivenet.utils.utils import to_torch

import seaborn as sns
import matplotlib.pyplot as plt

def CalculateScores(pred:Union[np.ndarray, torch.Tensor], mask:Union[np.ndarray, torch.Tensor], include_background:bool=False):
    pred = to_torch(pred)
    mask = to_torch(mask)

    pred_shape = pred.shape
    mask_shape = mask.shape

    if not pred_shape == mask_shape:
        raise ValueError(f"Please provide equal sized tensors for comparing predictions and grounth truth, not {pred_shape} and {mask_shape}")

    if len(pred_shape) == 3:
        if include_background == True:
            print('adding empty channel HWD -> CHWD')
            pred = pred[None, :]
            mask = mask[None, :]
            pred_shape = pred.shape
        else:
            raise ValueError(f"Predictions have shape {pred} and background == False, so should include atleast 2 channels (CHWD)")

    if len(pred_shape) == 4:
        print('adding empty batch CHWD -> BCHWD')
        pred = pred[None, :]
        mask = mask[None, :]
    elif len(pred_shape) != 5:
        raise ValueError(f"Unrecognized number of channels {pred_shape}, either provide HWD, CHWD or BCHWD")

    dice = compute_meandice(pred, mask, include_background=include_background)
    hausdorff_distance = compute_hausdorff_distance(pred, mask, include_background=include_background)
    surface_distance = compute_average_surface_distance(pred, mask, include_background=include_background)

    return dice, hausdorff_distance, surface_distance

def ResultPlot(data, scorename="Dice", types=False, unseen=False):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    fig, ax = plt.subplots()

    data = pd.DataFrame.from_dict(data, orient="index")
    data.columns = [scorename]
    data["Names"] = data.index
    data = data.reset_index()

    if types:
        data['Types'] = data['Names'].map(types)
        if unseen:
            data['Seen in training'] = data['Types'].map(unseen)
            data.loc[data['Seen in training'] != False, 'Seen in training'] = True
            sns.boxplot(x="Types", y=scorename, hue="Seen in training", dodge=False, data=data, ax=ax)
        else:
            sns.boxplot(x="Types", y=scorename, data=data, ax=ax)
    else:
        sns.boxplot(y=scorename, data=data, ax=ax)

    plt.xticks(rotation = 45, ha="right", rotation_mode="anchor")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig
