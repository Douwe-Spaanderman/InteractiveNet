import argparse

import os
from typing import List, Tuple, Dict, Sequence, Optional, Callable, Union

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from monai.transforms import AsDiscrete

from interactivenet.utils.visualize import ImagePlot
from interactivenet.utils.statistics import ResultPlot, CalculateScores, CalculateClinicalFeatures, ComparePlot
from interactivenet.utils.postprocessing import ApplyPostprocessing

def AnalyzeResults(mlflow, outputs:list, postprocessing:str, metadata:dict, labels:bool=False):
    # Transforms
    argmax = AsDiscrete(argmax=True)
    discrete = AsDiscrete(to_onehot=2)
    
    dices = {}
    hausdorff = {}
    surface = {}
    classes = {}
    volume = {}
    diameter = {}
    for output in outputs:
        name = Path(output[1][0]["filename_or_obj"]).name.split('.')[0]
        pred = output[0][0]

        pred = argmax(pred)
        pred = ApplyPostprocessing(pred, postprocessing)

        image = output[2]["image_raw"][0]
        if labels:
            mask = output[2]["label"][0]
            f = ImagePlot(image[0], mask[0], additional_scans=[pred[0]], CT=metadata["Fingerprint"]["CT"])
            
            pred = discrete(pred)
            mask = discrete(mask)
            dice, hausdorff_distance, surface_distance = CalculateScores(pred, mask)
            dices[name] = dice.item()
            hausdorff[name] = hausdorff_distance.item()
            surface[name] = surface_distance.item()
            classes[name] = output[2]["class"][0]

            volume_pred, volume_gt, diameter_pred, diameter_gt = CalculateClinicalFeatures(image, pred, mask, output[1][0])
            volume[name] = {"gt": volume_gt, "pred": volume_pred}
            diameter[name] = {"gt": diameter_gt, "pred": diameter_pred}
        else:
            f = ImagePlot(image[0], output[0], CT=metadata["Fingerprint"]["CT"])

        mlflow.log_figure(f, f"images/{name}.png")
    
    if labels:
        mlflow.log_metric("Mean dice", np.mean(list(dices.values())))
        mlflow.log_metric("Std dice", np.std(list(dices.values())))

        f = ResultPlot(dices, "Dice", classes)
        plt.close("all")
        mlflow.log_figure(f, f"dice.png")
        mlflow.log_dict(dices, "dice.json")

        f = ResultPlot(hausdorff, "Hausdorff Distance", classes)
        plt.close("all")
        mlflow.log_figure(f, f"hausdorff_distance.png")
        mlflow.log_dict(hausdorff, "hausdorff_distance.json")

        f = ResultPlot(surface, "Surface Distance", classes)
        plt.close("all")
        mlflow.log_figure(f, f"surface_distance.png")
        mlflow.log_dict(surface, "surface_distance.json")

        f = ComparePlot(volume)
        plt.close("all")
        mlflow.log_figure(f, f"volume.png")
        mlflow.log_dict(volume, "volume.json")
        mlflow.log_dict(diameter, "diameter.json")
