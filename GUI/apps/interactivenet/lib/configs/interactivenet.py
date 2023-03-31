import logging
import os
import glob
from typing import Any, Dict, Union
from collections import Counter

import lib.infers
import lib.trainers
from monai.networks.nets import DynUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask

from interactivenet.utils.utils import read_metadata

logger = logging.getLogger(__name__)


class InteractiveNet(TaskConfig):
    def init(
        self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, studies: str, **kwargs
    ):
        super().init(name, model_dir, conf, planner, **kwargs)

        task_dir = model_dir / conf["models"]
        metadata = read_metadata(task_dir / "plans.json")

        # This should be somewhere in the plans.json file!
        self.labels = {
            "tumor": 1,
        }

        self.median_shape = metadata["Fingerprint"]["Median size"]
        self.target_spacing = metadata["Fingerprint"]["Target spacing"]
        self.relax_bbox = metadata["Plans"]["padding"]
        self.divisble_using = metadata["Plans"]["divisible by"]
        self.clipping = metadata["Fingerprint"]["Clipping"]
        self.intensity_mean = metadata["Fingerprint"]["Intensity_mean"]
        self.intensity_std = metadata["Fingerprint"]["Intensity_std"]
        self.ct = metadata["Fingerprint"]["CT"]
        self.kernels = metadata["Plans"]["kernels"]
        self.strides = metadata["Plans"]["strides"]

        self.tmp_folder = "/tmp/"

        postprocessing = read_metadata(task_dir / "postprocessings.json")
        postprocessing = [x["postprocessing"] for x in postprocessing.values()]
        postprocessing = Counter(postprocessing).most_common()[0][0]

        # Model Files
        #if "ensemble" in name:
        #    self.ensemble = True
        #    self.path = [
        #        model for model in (task_dir / "model").glob(f"*") if model.is_dir()
        #    ]
        #else:
        #    fold = "0"
        #    print(f"using fold {fold} as ensemble is not selected")
        #    self.ensemble = False
        #    self.path = task_dir / "model" / fold

        #if "tta" in name:
        #    self.tta = True
        #else:
        #    self.tta = False

        # All the above commented out code can be used to have a model without tta and ensembling
        # Now we just use both
        self.ensemble = True
        self.tta = True
        self.path = [
            model / "data" / "model.pth" for model in (task_dir / "model").glob(f"*") if model.is_dir()
        ]

        # Need to load model using torch_load -> model._model to get dynunet -> should just adjusted loading of the method :)

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.InteractiveNet(
            path=self.path,
            ensemble=self.ensemble,
            tta=self.tta,
            median_shape=self.median_shape,
            target_spacing=self.target_spacing,
            relax_bbox=self.relax_bbox,
            divisble_using=self.divisble_using,
            clipping=self.clipping,
            intensity_mean=self.intensity_mean,
            intensity_std=self.intensity_std,
            ct=self.ct,
            labels=self.labels,
            tmp_folder=self.tmp_folder
        )
        return task

    def trainer(self) -> None:
        return None
