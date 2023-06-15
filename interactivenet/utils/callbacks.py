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

##################################
##################################
## ALL CODE HERE IS OUTDATED!!! ##
##################################
##################################
##################################


from pytorch_lightning.callbacks import Callback

from monai.transforms import AsDiscrete

from interactivenet.utils.postprocessing import ApplyPostprocessing
from interactivenet.utils.statistics import CalculateScores


class AnalyzeResults(Callback):
    def __init__(self):
        super().__init__()

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        output, meta = outputs

        argmax = AsDiscrete(argmax=True)
        output = argmax(output[0])
        output = ApplyPostprocessing(output, pl_module.postprocessing["postprocessing"])

        batch["image_raw"][0]
        if pl_module.labels:
            label = batch["label"][0]

            discrete = AsDiscrete(to_onehot=2)
            label = discrete(label)
            output = discrete(output)

            dice, hausdorff_distance, surface_distance = CalculateScores(output, label)
            pl_module.logger.log_metric("Dice", dice)
            pl_module.logger.log_metric("HD", hausdorff_distance)
            pl_module.logger.log_metric("SD", surface_distance)
        else:
            print("No labels available, so cannot calculate metrics")
            # f = ImagePlot(image, label, additional_scans=output, CT=pl_module.metadata["Fingerprint"]["CT"])
