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
