# Dataset Conversion
Using InteractiveNet requires you to structure your dataset in a format closely following the data structure of [Medical Segmentation Decthlon](http://medicaldecathlon.com/).

The data should be present in the interactiveseg_raw folder (which is needed to be specified like [this](env_variables.md)). Each dataset is stored as a 'Task', which is associated to a specific Task ID, a three digit number, and a Task name, e.g. Task800_WORC_MRI. Please use a unique Task ID, to be safe use task IDs between 400-700, as MSD tasks already use 0-300 and our tasks already use 800-900. InteractiveNet/raw data folder should look like this:

    InteractiveNet/raw
    ├── Task001_BrainTumour
    ├── Task002_Heart
    ├── ...
    ├── Task600_YOURTASK
    ├── ...
    ├── Task800_WORC_MRI
    ├── ...
 
Within each task folder, the following structure is expected:

    Task800_WORC_MRI/
    ├── imagesTr
    ├── (imagesTs)
    ├── interactionsTr
    ├── (interactionsTs)
    ├── labelsTs
    └── (labelsTs)

imagesTr contain the images for the training samples. Based on these images, InteractiveNet is configured, and trained using cross-validation. imagesTs is optional and should contain test samples not present in imagesTr. interactionsTr contain the interior margin points, while interactionsTs are optional for test samples. **Note that interactions can be derived synthetically, which is described [here](synthetic_interactions.md)**  Finally, labelsTr contain ground truth segmentation maps for the training cases. labelsTs can also be defined but are optional, and can be used to compare results on a seperate test set.

Each sample should have an unique identifier, so the images, interactions and labels can be matched. **All files should be 3D nifti files (.nii.gz)**. The image can have any scalar pixel type. Label should contain only 0 or 1 integers, where 0 is considered background. Currenlty, multiple labels, e.g. 1 tumor and 2 edema is not supported. Interactions should also only contain 0 or 1 integers, where 1 represents the interior margin points. As images can have multiple modalities, we specifiy these by an extra suffix of four digits at the end of the filename.

Here is an example of the first task of the MSD: BrainTumor. In which each image has four modalities (0000), T1w (0001), T1gd (0002) and T2w (0003):

    InteractiveNet/raw/Task001_BrainTumour
    ├── imagesTr
    │   ├── BRATS_001_0000.nii.gz
    │   ├── BRATS_001_0001.nii.gz
    │   ├── BRATS_001_0002.nii.gz
    │   ├── BRATS_001_0003.nii.gz
    │   ├── BRATS_002_0000.nii.gz
    │   ├── BRATS_002_0001.nii.gz
    │   ├── BRATS_002_0002.nii.gz
    │   ├── BRATS_002_0003.nii.gz
    │   ├── BRATS_003_0000.nii.gz
    │   ├── BRATS_003_0001.nii.gz
    │   ├── BRATS_003_0002.nii.gz
    │   ├── BRATS_003_0003.nii.gz
    │   ├── BRATS_004_0000.nii.gz
    │   ├── BRATS_004_0001.nii.gz
    │   ├── BRATS_004_0002.nii.gz
    │   ├── BRATS_004_0003.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── BRATS_485_0000.nii.gz
    │   ├── BRATS_485_0001.nii.gz
    │   ├── BRATS_485_0002.nii.gz
    │   ├── BRATS_485_0003.nii.gz
    │   ├── BRATS_486_0000.nii.gz
    │   ├── BRATS_486_0001.nii.gz
    │   ├── BRATS_486_0002.nii.gz
    │   ├── BRATS_486_0003.nii.gz
    │   ├── BRATS_487_0000.nii.gz
    │   ├── BRATS_487_0001.nii.gz
    │   ├── BRATS_487_0002.nii.gz
    │   ├── BRATS_487_0003.nii.gz
    │   ├── BRATS_488_0000.nii.gz
    │   ├── BRATS_488_0001.nii.gz
    │   ├── BRATS_488_0002.nii.gz
    │   ├── BRATS_488_0003.nii.gz
    │   ├── ...
    ├── interactionsTr
    │   ├── BRATS_001.nii.gz
    │   ├── BRATS_002.nii.gz
    │   ├── BRATS_003.nii.gz
    │   ├── BRATS_004.nii.gz
    │   ├── ...
    ├── interactionsTs
    │   ├── BRATS_485.nii.gz
    │   ├── BRATS_486.nii.gz
    │   ├── BRATS_487.nii.gz
    │   ├── BRATS_488.nii.gz
    │   ├── ...
    ├── labelsTr
    │   ├── BRATS_001.nii.gz
    │   ├── BRATS_002.nii.gz
    │   ├── BRATS_003.nii.gz
    │   ├── BRATS_004.nii.gz
    │   ├── ...
    └── labelsTs
        ├── BRATS_485.nii.gz
        ├── BRATS_486.nii.gz
        ├── BRATS_487.nii.gz
        ├── BRATS_488.nii.gz
        ├── ...

Note that imagesTs, interactionsTs, and labelsTs are optional and not required for configuring InteractiveNet.
