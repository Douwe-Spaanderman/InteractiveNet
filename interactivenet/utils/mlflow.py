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

import mlflow
import os
from pathlib import Path


def mlflow_get_runs(name):
    results = Path(os.environ["interactivenet_results"], "mlruns")
    mlflow.set_tracking_uri(results)

    experiment_id = mlflow.get_experiment_by_name(name)
    if experiment_id == None:
        raise ValueError("Experiments not found, please first train models")
    else:
        experiment_id = experiment_id.experiment_id

    runs = mlflow.search_runs(experiment_id)
    return runs, experiment_id


def mlflow_get_id(name):
    results = Path(os.environ["interactivenet_results"], "mlruns")
    mlflow.set_tracking_uri(results)

    experiment_id = mlflow.get_experiment_by_name(name)
    if experiment_id == None:
        print(f"experiment_id not found will create {name}")
        experiment_id = mlflow.create_experiment(name)
    else:
        experiment_id = experiment_id.experiment_id

    return experiment_id
