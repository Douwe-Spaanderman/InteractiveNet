# InteractiveNet Graphical User Interface

Here we provide the [monailabel](https://github.com/Project-MONAI/MONAILabel) application for running interactivenet in your 3D Slicer or OHIF viewer, to directly run inference on new samples. This will also allow you to easily make new interactions.

When the server has been setup, you can use the monailabel to annotate image with six interior margin points, and run the pipeline. An example in 3D slicer of this is shown here:

https://user-images.githubusercontent.com/39140958/229098680-5b9ce004-0274-47e9-938b-e2017e68934a.mp4

# Installation

There are two main ways to install the InteractiveNet GUI:
1. Install using container (e.g. [Docker](https://www.docker.com/)). This method is recommended for most users as it seamlessly works 'out of the box', eliminating the need for local software installations. This is also the best option when working on windows.
2. Install locally. This method is advised when your interested in improving the method, debugging or need some more flexibility with hosting.

Regardless of which installation method you pick, you will need to either install [3D slicer](https://www.slicer.org/) or the [OHIF](https://ohif.org/) viewer. The main difference of these two is that 3D slicer supports images in nifti format (.nii, .nii.gz), while OHIF support images in dicom format (.dcm). Finally, in both viewers you have to install the monailabel plugin as described [here](https://github.com/Project-MONAI/MONAILabel/README#Plugins).

## Install using container

The container is primarily build for [Docker](https://www.docker.com/), however should also work with [Podman](https://podman.io/) or [Apptainer](https://apptainer.org/). In order to use the docker image have one of the container managers installed.

Next build the dockerfile (ensure that you navigated to the directory with 'Dockerfile'):
```
docker build --tag "interactivenet" .
```

This is all you have to do in order to install the application. Note, that it might take a few minutes to completely build the image.

## Install locally

In order to setup the monailabel locally you need to first follow the installation guides for interactivenet [here](../README.md#installation). Next you will need to install the extra requirements, found in the requirements_extra.txt (or in requirements.txt). This can be easily done using pip:
```
pip install -r requirements_extra.txt
```

# Usage

Depending on how you installed the application (container vs locally) usage is also a bit different.

## Container

Running the monailabel server as a docker image can be done using:
```
docker run -p 8000:8000 -e studies="PATH_TO_DATA" -e models="TaskXXX_YOURTASK" "interactivenet" 
```

## Locally

Running the monailabel server locally can be done using:
```
monailabel start_server --app apps/interactivenet --studies PATH_TO_DATA --conf models TaskXXX_YOURTASK
```

## Arguments

As you notice, with either methods you have to define certain arguments:
- ```studies```: defines the location of your input images. Depending if you use [3D slicer](https://www.slicer.org/) or the [OHIF](https://ohif.org/) viewer the location of input images differs:
    - [3D slicer](https://www.slicer.org/): This is simply a path to your nifti files (.nii, .nii.gz). 
    - [OHIF](https://ohif.org/): requires you to run a dicom web server (e.g. [Orthanc](https://www.orthanc-server.com/)) and provide the adress on which this dicom server is hosted, e.g. http://orthanc:8042/dicom-web/. For more info, checkout [monailabel documents](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/ohif). Note, that when using OHIF + interactivenet installed using contained, you might have to expose the container to the dicom server port.
- ```models```: defined the models you can select in the viewer for inference. Note, that this command always needs to start with ```models```, following the Task ID you want to use for inference. This can be one model (```models TaskXXX_YOURTASK```), multiple models (```models 'TaskXXX_YOURTASK,TaskXXX_OTHERTASK'```), or all available models (```models all```).

Additional options can be found under ```-h``` or ```--help```. Importantly, monailabel does not search for an available port, but standard uses port 8000. If address is already in use, you can use ```-p``` or ```--port``` to define an available port.