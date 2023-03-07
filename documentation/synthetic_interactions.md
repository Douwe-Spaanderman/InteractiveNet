# Synthetic interior margin points
Deriving interactions for each sample is a time-consuming process. As a pilot you could opt for using the ground truth segmentations to create the interior margin points synthetically. It is important to adhere to the dataset format described [here](dataset_conversion.md).

We provide a 'mimic interaction' script to create synthetic interactions. For the most basic configuration please use the following command:
```
interactivenet_mimic_interactions -t TaskXXX_MYTASK
```

This will create six extreme points, and move them inwards by five pixels, and 1 pixel if the image is anisotropic (in the lowest resolution plane).  

The 'mimic interaction' script includes other options, such as identifying the center of mass, drawing random points, and adjusting extreme points more in/outwards. For these options please use:
```
interactivenet_mimic_interactions -h
```
Note, other settings for synthetic interactions have been mimimally tested, and could provide very different results.