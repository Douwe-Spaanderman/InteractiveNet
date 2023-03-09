import mlflow
import os
from pathlib import Path

def mlflow_get_runs(name):
    results = Path(os.environ["interactiveseg_results"], "mlruns")
    mlflow.set_tracking_uri(results)

    experiment_id = mlflow.get_experiment_by_name(name)
    if experiment_id == None:
        raise ValueError("Experiments not found, please first train models")
    else: experiment_id = experiment_id.experiment_id

    runs = mlflow.search_runs(experiment_id)
    return runs, experiment_id

def mlflow_get_id(name):
    results = Path(os.environ["interactiveseg_results"], "mlruns")
    mlflow.set_tracking_uri(results)

    experiment_id = mlflow.get_experiment_by_name(name)
    if experiment_id == None:
        print(f"experiment_id not found will create {name}")
        experiment_id = mlflow.create_experiment(name)
    else: experiment_id = experiment_id.experiment_id

    return experiment_id