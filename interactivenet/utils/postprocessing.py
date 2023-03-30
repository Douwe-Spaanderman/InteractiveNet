import torch
import numpy as np
from monai.transforms import Compose, KeepLargestConnectedComponent, FillHoles


def ApplyPostprocessing(output, method):
    if method == "fillholes":
        postprocessing = FillHoles(applied_labels=None, connectivity=2)
    elif method == "largestcomponent":
        postprocessing = KeepLargestConnectedComponent(
            applied_labels=None, connectivity=2
        )
    elif method == "fillholes_and_largestcomponent":
        postprocessing = Compose(
            [
                FillHoles(applied_labels=None, connectivity=2),
                KeepLargestConnectedComponent(applied_labels=None, connectivity=2),
            ]
        )
    else:
        return output

    if len(output.shape) == 5:
        new_output = []
        for batch in output:
            new_output.append(np.stack([postprocessing(x) for x in batch], axis=0))

        output = np.stack(new_output, axis=0)
    elif len(output.shape) == 4:
        new_output = []
        for channel in output:
            new_output.append(postprocessing(channel))

        output = np.stack(new_output, axis=0)
    else:
        output = postprocessing(output)

    return output
