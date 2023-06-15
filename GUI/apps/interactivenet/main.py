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

import json
import logging
import os
from typing import Dict
from pathlib import Path

import lib.configs
import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import strtobool
from monailabel.utils.others.planner import HeuristicPlanner
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random


logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = Path(os.environ["interactivenet_results"], "models")

        c = get_class_names(lib.configs, "TaskConfig")
        if len(c) > 1:
            raise KeyError(
                "Multiple methods were found, I think you adjusted something from the original repo..."
            )
        else:
            c = c[0]

        configs = {}
        for name in [x.name for x in self.model_dir.glob("*") if x.is_dir()]:
            configs[name] = c
            # configs[f"{name}+ensemble"] = c
            # configs[f"{name}+tta"] = c
            # configs[f"{name}+ensemble+tta"] = c

        configs = {k: v for k, v in sorted(configs.items())}

        models = conf.get("models")
        if not models:
            print("")
            print(
                "---------------------------------------------------------------------------------------"
            )
            print("Provide --conf models <name>")
            print(
                "Following are the available models.  You can pass comma (,) seperated names to pass multiple"
            )
            print(f"    all, {', '.join(configs.keys())}")
            print(
                "---------------------------------------------------------------------------------------"
            )
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print(
                "---------------------------------------------------------------------------------------"
            )
            print(f"Invalid Model(s) are provided: {invalid}")
            print(
                "Following are the available models.  You can pass comma (,) seperated names to pass multiple"
            )
            print(f"    all, {', '.join(configs.keys())}")
            print(
                "---------------------------------------------------------------------------------------"
            )
            print("")
            exit(-1)

        # Use Heuristic Planner to determine target spacing and spatial size based on dataset+gpu
        spatial_size = json.loads(conf.get("spatial_size", "[48, 48, 32]"))
        target_spacing = json.loads(conf.get("target_spacing", "[1.0, 1.0, 1.0]"))
        self.heuristic_planner = strtobool(conf.get("heuristic_planner", "false"))
        self.planner = HeuristicPlanner(
            spatial_size=spatial_size, target_spacing=target_spacing
        )

        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, self.planner, studies)

        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - InteractiveNet ({monailabel.__version__})",
            description="Interactivenet",
            version=monailabel.__version__,
        )

    def init_datastore(self) -> Datastore:
        datastore = super().init_datastore()
        if self.heuristic_planner:
            self.planner.run(datastore)

        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v

        logger.info(infers)
        return infers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "random": Random(),
            "first": First(),
        }

        if strtobool(self.conf.get("skip_strategies", "true")):
            return strategies

        for n, task_config in self.models.items():
            s = task_config.strategy()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Strategy:: {k} => {v}")
                strategies[k] = v

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies
