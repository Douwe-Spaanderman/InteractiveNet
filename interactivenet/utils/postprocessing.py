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
