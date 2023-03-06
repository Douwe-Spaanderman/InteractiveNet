# InteractiveNet

Automatic segmentation has success in a variety of application throughout medical imaging. However, there could be a couple of reasons for automatic segmentation to fall short of providing accurate segmentations:
1. Object are irregular and/or lobulated, which creates difficult segmentation task.
2. It is impossible to encompass all heterogeneity from the patient population in a training dataset.
3. It is difficult to create a (large) annotated dataset.

Here, we address this issue by using knowledge from a trained clinician to guide a 3D U-Net to improve segmentations. This is done using six interior margin points, i.e. extreme point in each axis. In theory, this approach should provide accurate segmentations for object or on modalities not seen during the training of the model. Currently, this has been assessed in various types of Soft-Tissue Tumors.

Inspired by [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), InteractiveNet uses dataset properties like imaging modality, voxel spacing, and image size to determine the best possible configuration for preprocessing, training, postprocessing, ensembling, and much more to provide a miminally interactive segmenation pipeline. By providing this framework, we hope to enable the use of minimally interactive segmentation for many more applications in medical imaging. **We provide the option to generate interior margin points synthetically from ground truth segmentation**

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

## Running on a new dataset
Using InteractiveNet requires you to structure your dataset in a format closely following the data structure of [Medical Segmentation Decthlon](http://medicaldecathlon.com/). How to convert your dataset to be compatible with InteractiveNet can be found [here](documentation/dataset_conversion.md). Also, for InteractiveNet interior margin points are required, if it is not possible to create these interactions manual, we provide options to derive these interactions
[synthetically](documentation/synthetic_interactions.md)

### Fingerprinting and preprocessing
InteractiveNet uses fingerprinting of the dataset to determine the best strategy for preprocessing and determines best network configurations. you can run the preprocessing by:

```

```



### Training

### Inference

## Running using a pretrained model

### GUI

### Command line

# Roadmap 
- [ ] Example of running InteractiveNet
- [ ] Support lower memory graphic cards for high dimensional CT data
- [ ] Support multi-modality input
- [ ] Support multi labels, and multi interactions

# Acknowledgements

<img src="BIGR_logo.jpg" width="512px" />

InteractiveNet is developed by the [Biomedical Imaging Group Rotterdam (BIGR)](https://bigr.nl/)
