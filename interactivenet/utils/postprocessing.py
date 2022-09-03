import torch
from monai.transforms import (
    Compose,
    KeepLargestConnectedComponent,
    FillHoles
)

def ApplyPostprocessing(output, method):
    if method == "fillholes":
        postprocessing = FillHoles(applied_labels=None, connectivity=2)
    elif method == "largestcomponent":
        postprocessing = KeepLargestConnectedComponent(applied_labels=None, connectivity=2)
    elif method == "fillholes_and_largestcomponent":
        postprocessing = Compose([
            postprocessing = FillHoles(applied_labels=None, connectivity=2),
            postprocessing = KeepLargestConnectedComponent(applied_labels=None, connectivity=2)
        ])
    else:
        return output

    if len(output.shape) == 5:
        new_output = []
        for batch in output:
            new_output.append(torch.stack([postprocessing(x) for x in batch], dim=0))

        output = torch.stack(new_output, dim=0)
    elif len(output.shape) == 4:
        new_output = []
        for channel in output:
            new_output.append(postprocessing(channel))

        output = torch.stack(new_output, dim=0)
    else:
        output = postprocessing(output)

    return output