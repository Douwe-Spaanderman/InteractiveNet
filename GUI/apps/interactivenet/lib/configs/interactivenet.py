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

import logging
from typing import Any, Dict, Union
from collections import Counter

import lib.infers
import lib.trainers

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask

from interactivenet.utils.utils import read_metadata

logger = logging.getLogger(__name__)


class InteractiveNet(TaskConfig):
    def init(
        self,
        name: str,
        model_dir: str,
        conf: Dict[str, str],
        planner: Any,
        studies: str,
        **kwargs,
    ):
        super().init(name, model_dir, conf, planner, **kwargs)

        if "+" in conf["models"]:
            task_dir = model_dir / conf["models"].split("+")[0]
        else:
            task_dir = model_dir / conf["models"]
        metadata = read_metadata(task_dir / "plans.json")

        # This should be somewhere in the plans.json file! Or this should be an option when launching
        self.labels = {
            "tumor" : 1,
            "metatase_1": 2,
            "metatase_2": 3,
            "metatase_3": 4,
            "meta_4": 5,
            "meta_5": 6,
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
        # if "ensemble" in name:
        #    self.ensemble = True
        #    self.path = [
        #        model for model in (task_dir / "model").glob(f"*") if model.is_dir()
        #    ]
        # else:
        #    fold = "0"
        #    print(f"using fold {fold} as ensemble is not selected")
        #    self.ensemble = False
        #    self.path = task_dir / "model" / fold

        # if "tta" in name:
        #    self.tta = True
        # else:
        #    self.tta = False

        # All the above commented out code can be used to have a model without tta and ensembling
        # Now we just use both

        # Newly introduced to provide faster pipeline
        # Note, impact on performance has been limited evaluated.
        if "fastR" in name:
            self.fast = True
        else:
            self.fast = False

        self.ensemble = True
        self.tta = True
        self.path = [
            model / "data" / "model.pth"
            for model in (task_dir / "model").glob(f"*")
            if model.is_dir()
        ]

        # Need to load model using torch_load -> model._model to get dynunet -> should just adjusted loading of the method :)

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.InteractiveNet(
            path=self.path,
            fast=self.fast,
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
            tmp_folder=self.tmp_folder,
        )
        return task

    def trainer(self) -> None:
        return None
