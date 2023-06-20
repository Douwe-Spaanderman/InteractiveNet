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

import os
from pathlib import Path
import argparse
import requests
import zipfile

def get_available_models():
    available_models = {
        "Task800_WORC_MRI": {
            'description': "Soft Tissue Tumor (STT) segmentation on MRI. \n"
                           "Segmentation targets are tumor. \n"
                           "Input modalities are 0: T1. \n"
                           "Also see the WORC database, https://doi.org/10.1101/2021.08.19.21262238",
            'url': "https://zenodo.org/record/8054038/files/Task800_WORC_MRI.zip?download=1"
        },
        "Task801_WORC_CT": {
            'description': "Soft Tissue Tumor (STT) segmentation on CT. \n"
                           "Segmentation targets are tumor. \n"
                           "Input modalities are 0: CT. \n"
                           "Also see the WORC database, https://doi.org/10.1101/2021.08.19.21262238",
            'url': "https://zenodo.org/record/8054038/files/Task801_WORC_CT.zip?download=1"
        },
    }
    return available_models

def print_available_pretrained_models():
    print('\033[1m' + '\nThe following pretrained models are available:\n\n' + '\033[0m')
    models = get_available_models()
    for m in models.keys():
        print('\033[1m' + m + '\033[0m')
        print(models[m]['description'])
        print('')

def download_and_install_model():
    models = get_available_models()
    available_models = ["all"] + list(models.keys())

    parser = argparse.ArgumentParser(
        description="Download and install pretrained networks"
    )
    parser.add_argument(
        "-t", 
        "--task", 
        required=True, 
        choices=available_models,
        type=str,
        help="task name")
    args = parser.parse_args()

    if "interactivenet_results" not in os.environ:
        raise RuntimeError(f"Environment paths have not been set, this means InteractiveNet has not been correctly installed.")

    if args.task == "all":
        for model in models.keys():
            download_install(task=model, models=models)
    else:
        download_install(task=args.task, models=models)
    
def download_install(task:str, models:dict):
    print(f"\nDownloading and Installing {task}")
    results = Path(os.environ["interactivenet_results"], "models")
    results.mkdir(parents=True, exist_ok=True)

    if task not in models.keys():
        raise RuntimeError(f"The requested pretrained model ({task}) is not available.")

    url = models[task]['url']

    print("Downloading files")
    local = results / url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(str(local), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)

    print("Extracting files")
    with zipfile.ZipFile(str(local), 'r') as zip_ref:
        zip_ref.extractall(str(results / task))

    local.unlink(missing_ok=False)
    print('Done')

if __name__ == "__main__":
    print_available_pretrained_models()
