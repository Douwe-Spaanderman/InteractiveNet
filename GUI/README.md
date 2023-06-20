# InteractiveNet Graphical User Interface

Here we provide the [monailabel](https://github.com/Project-MONAI/MONAILabel) application for running interactivenet in your 3D Slicer or OHIF viewer, to directly run inference on new samples. This will also allow you to easily make new interactions.

# Installation

In order to setup the monailabel server you need to follow the installation guides [here](../README.md#installation). Next you will need to install the extra requirements, found in the requirements_extra.txt (or in requirements.txt). This can be easily done using pip:
```
pip install -r requirements_extra.txt
```

Next, you will need to either install [3D slicer](https://www.slicer.org/) or the [OHIF](https://ohif.org/) viewer. Finally, in both viewers you have to install the monailabel plugin as described [here](https://github.com/Project-MONAI/MONAILabel/README#Plugins).

Alternatively, we provided a [docker](https://www.docker.com/) image in order to setup the monailabel application, and everything required. Note that this doesn't include the viewer.

# Usage

In order to use interactivenet in a graphical user interface (gui) / viewer you have to run a monailabel server. You can use the following command to run the server:
```
monailabel start_server --app apps/interactivenet --studies PATH_TO_DATA --conf models TaskXXX_YOURTASK
```

You have to define the location of your input images using ```--studies```. These should be in nifti format (.nii, .nii.gz). Next, you can define the models you can select in the viewer for inference, using ```--conf```. Note, that this command always needs to start with ```models```, following the Task ID you want to use for inference. This can be one model (```models TaskXXX_YOURTASK```), multiple models (```models 'TaskXXX_YOURTASK,TaskXXX_OTHERTASK'```), or all available models (```models all```).

Additional options can be found under ```-h``` or ```--help```. Importantly, monailabel does not search for an available port, but standard uses port 8000. If address is already in use, you can use ```-p``` or ```--port``` to define an available port.

When the server has been setup, you can use the monailabel to annotate image with six interior margin points, and run the pipeline. An example of this is shown here:

https://user-images.githubusercontent.com/39140958/229098680-5b9ce004-0274-47e9-938b-e2017e68934a.mp4
