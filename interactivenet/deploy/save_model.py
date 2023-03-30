from pathlib import Path
import argparse
import os
import json
import shutil


from interactivenet.utils.utils import (
    read_metadata,
)
from interactivenet.utils.mlflow import mlflow_get_runs


import mlflow.pytorch


def main():
    parser = argparse.ArgumentParser(
        description="Predict on the interactivenet network"
    )
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    args = parser.parse_args()

    exp = Path(os.environ["interactivenet_processed"], args.task)
    results = Path(os.environ["interactivenet_results"], "mlruns")
    metadata = read_metadata(exp / "plans.json")
    if "Cases" in metadata:
        del metadata["Cases"]
    if "splits" in metadata["Plans"]:
        del metadata["Plans"]["splits"]

    mlflow.set_tracking_uri(results)
    runs, experiment_id = mlflow_get_runs(args.task)

    models = {}
    postprocessings = {}
    for idx, run in runs.iterrows():
        if run["tags.Mode"] != "training":
            continue

        run["run_id"]
        fold = run["params.fold"]
        artifact_uri = Path(run["artifact_uri"].split("file://")[-1])
        postprocessing = read_metadata(
            artifact_uri / "postprocessing.json",
            error_message="postprocessing hasn't been run yet, please do this before predictions",
        )
        if postprocessing["using_checkpoint"]:
            models[fold] = artifact_uri / "model_checkpoint"
        else:
            models[fold] = artifact_uri / "model"

        postprocessings[fold] = postprocessing

    results = Path(os.environ["interactivenet_results"], "models", args.task)
    results.mkdir(parents=True, exist_ok=True)

    print(f"Saving models in {results}")
    with open(results / "plans.json", "w") as outfile:
        json.dump(metadata, outfile)

    with open(results / "postprocessings.json", "w") as outfile:
        json.dump(postprocessings, outfile)

    for fold, model in models.items():
        shutil.copytree(model, results / "model" / fold)


if __name__ == "__main__":
    main()
