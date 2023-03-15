# InteractiveNet

Automatic segmentation has success in a variety of application throughout medical imaging. However, there could be a couple of reasons for automatic segmentation to fall short of providing accurate segmentations:
- Object are irregular and/or lobulated, which creates difficult segmentation task.
- It is impossible to encompass all heterogeneity from the patient population in a training dataset.
- It is difficult to create a (large) annotated dataset.

Here, we address this issue by using knowledge from a trained clinician to guide a 3D U-Net to improve segmentations. This is done using six interior margin points, i.e. extreme point in each axis. In theory, this approach should provide accurate segmentations for object or on modalities not seen during the training of the model. Currently, this has been assessed in various types of Soft-Tissue Tumors.

Inspired by [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), InteractiveNet uses dataset properties like imaging modality, voxel spacing, and image size to determine the best possible configuration for preprocessing, training, postprocessing, ensembling, and much more to provide a miminally interactive segmenation pipeline. By providing this framework, we hope to enable the use of minimally interactive segmentation for many more applications in medical imaging. **We provide the option to generate interior margin points [synthetically](documentation/synthetic_interactions.md) from ground truth segmentation**, which is ideal for pilot studies.

# Table of Contents
- [InteractiveNet](#interactivenet)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

# Installation
InteractiveNet has been tested on Linux (Ubuntu 20.04, and centOS), and MacOS (Big Sur). We do not provide support for other operating systems.

InteractiveNet requires a GPU for training, and while we recommend a GPU for inference, it is not required. Currently, we use an RTX 3060 (12GB) for soft-tissue tumors on MRI, and an NVIDIA A40 (48GB) for high dimensional CT data.

We recommend installing InteractiveNet in a [virtual environment](https://docs.python.org/3/library/venv.html). We tested InteractiveNet using Python 3.9.5, and do not provide support for earlier Python versions.

1. Install [Pytorch](https://pytorch.org/get-started/locally/) using pip or compile from source. Make sure to get the latest version which matches your [CUDA version](https://stackoverflow.com/questions/9727688/how-to-get-the-cuda-version), if you cannot find your CUDA version you can look for a previous version of pytorch [here](https://pytorch.org/get-started/previous-versions/). Note, that when a non matching version of Pytorch is installed, it will drastically slow down training and inference. Also, it is essential to install Pytorch before pip install InteractiveNet.
2. Install InteractiveNet:
    1. Install as is:
       ```
       pip install interactivenet
       ```
    2. Install for development (Only do this when you want to adjust/add code in the repository):
        ```
        git clone https://github.com/Douwe-Spaanderman/InteractiveNet.git
        cd InteractiveNet
        pip install -e .
        ```
3. InteractiveNet needs to know where you intent to save raw data, processed data and results. Follow [these](documentation/env_variables.md) instructions to set up environment paths.

# Usage
Interactivenet installs several new commands to your terminal, which are used to run the interactivenet pipeline. All commands have the prefix ```interactivenet_```. All commands have a ```-h``` of ```--help``` option to give you more information on how to use them.

## Running on a new dataset
Using InteractiveNet requires you to structure your dataset in a format closely following the data structure of [Medical Segmentation Decthlon](http://medicaldecathlon.com/). How to convert your dataset to be compatible with InteractiveNet can be found [here](documentation/dataset_conversion.md). Also, for InteractiveNet interior margin points are required, if it is not possible to create these interactions manual, we provide options to derive these interactions
[synthetically](documentation/synthetic_interactions.md)

## Fingerprinting and preprocessing
InteractiveNet uses fingerprinting of the dataset to determine the best strategy for preprocessing and determines best network configurations. you can run fingerprinting, experiment planning and processing in one go using:

```
interactivenet_plan_and_process -t TaskXXX_YOURTASK
```

This command creates and populates the interactivenet_processed/TaskXXX_MYTASK folder, with plans on experiment running, and preprocessed .npz and .pkl files for your data. This is done so training is significantly faster. Finally, using ```-v``` or ```--verbose``` with the above command will create snapshots of all the images at different timepoints of the processing pipeline. More specifically, it will create images from: raw data, the exponentialized geodesic map, and final processed data.

Using ```-h``` or ```--help``` for planning and processing gives you multiple options to adjust settings in the experiment. One option I would advise to use is setting the ```-s``` or ```--seed``` to a set value. This will make sure that you will be able to replicate your experiments. If you forgot to do this, don't worry the randomly generated seed is stored in plans.json file.

Note, that together, depending on how powerful your CPU is, running planning and preprocessing might take up to half an hour.

## MLflow 

We use MLflow to automatically log the interactivenet pipeline from training to testing your models. If you are not familiar with MLflow, please visit [here](https://mlflow.org/) for more information. Additionally, [here](documentation/mlflow.md) we have provided documentation to guide you to access MLflow. **Note that you cannot access MLflow without training atleast 1 model/fold**

## Training
InteractiveNet uses five fold cross-validation in order to use ensembling and define best postprocessing steps. You need to train all folds (default = 5), otherwise inference does not work.

To train interactivenet for all FOLDS in [0, 1, 2, 3, 4] (if default number of folds is selected), run:

```
interactivenet_train -t TaskXXX_YOURTASK -f FOLD
```

Additional options can be found under ```-h``` or ```--help```.

All experiments results as well as trained models can be found in the interactivenet_results folder. Tracking experiments and visualizing results is done in MLflow, for which documentation can be found [here](documentation/mlflow.md).

## Inference

## Running using a pretrained model

# GUI

# Roadmap 
- [ ] Example of running InteractiveNet
- [ ] Support lower memory graphic cards for high dimensional CT data
- [ ] Support multi-modality input
- [ ] Support multi labels, and multi interactions
- [ ] Resample label back instead of weights

# Acknowledgements

<img src="BIGR_logo.jpg" width="512px" />

InteractiveNet is developed by the [Biomedical Imaging Group Rotterdam (BIGR)](https://bigr.nl/)
