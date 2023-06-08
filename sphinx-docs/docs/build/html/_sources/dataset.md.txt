# Dataset creation


The idea is that the framework serves as a research toolbox to train and run inference on medical image segmentation models. The toolbox supports inference on already pre-trained models as well as enables an easy creation of new models by inheriting from and configuring an existing architecture.

Currently we are working on a model for dermatoscopic image segmentation trained on the PH2 dataset (see details below).

This document describes the structure and format of data expected by the framework as well as general steps necessary to prepare this data for training and, ultimately, inference.  


## Data

### General requirements

Currently, the framework expects images and their segmented counterparts (i.e. labels) to be provided in the following format:

* Images are saved as png or bmp. For each *image* there is a corresponding *mask*. 
* Following the UNet architecture, currently used to build our segmentation model, the sampling of images must be no less than 572x572 pixels and the sampling of masks no less than 388x388 pixels. 
* The data can either be pre-processed by users and provided at the specified samplings or can be pre-processed by the framework prior to their use. It is planned to have a dedicated framewrok for dataset creation, inclusing any necessary processing steps.
* Each set of data has a unique identifier *sampleID* and resides in a separate folder with this particular ID name *sampleID*. The *sampleID* folder contains two other folders - one with an *image*, another with a respective *mask*:

```
root_folder
├── folder_with_images
│    ├── sampleID1
│    │    ├── sampleID1_image
│    │    │    └── image1.bmp
│    │    └── sampleID1_mask
│    │         └── mask1.bmp
│    ├── sampleID2
│    │    ├── sampleID2_image
│    │    │    └── image2.bmp
│    │    └── sampleID2_mask
│    │         └── mask2.bmp
│    ├──...
├── ...
```


### Example: PH2 Data


As a specific example, that is also used to build a dermatoscopic segmentation model in this framewrok, let us take the `PH2 dermoscopic image database <https://www.fc.up.pt/addi/ph2%20database.html>` acquired at the Dermatology Service of Hospital Pedro Hispano, Matosinhos, Portugal. They are 8-bit RGB color images with a resolution of 768x560 pixels of lesions (we refer to them as *images*) and their segmentations available as binary masks (we refer to them as *masks*). In total the database comprises 200 datasets and corresponding files with metadata:


* *PH2 Dataset images* folder contains the image data.
* *PH2 Dataset.xlsx* file contains the classification of all images according to the dermoscopic criteria that are evaluated in the PH2 database.
* *PH2 Dataset.txt* file contains the classification of all images according to the dermoscopic criteria that are evaluated in the PH2 database.

For more details please refer to the original website. The file structure for PH2 segmentation follows the structure described in general requirements above:

```
PH2Dataset
├── PH_Dataset_images
│    ├── IMD002
│    │    ├── IMD002_Dermoscopic_Image
│    │    │    └── IMD002.bmp
│    │    └── IMD002_lesion
│    │         └── IMD002_lesion.bmp
│    ├── IMD003
│    │    ├── IMD003_Dermoscopic_Image
│    │    │    └── IMD003.bmp
│    │    └── IMD003_lesion
│    │         └── IMD003_lesion.bmp
│    ├──...
├── PH2_dataset.txt
└── PH2_dataset.xlsx
```

* The data are resized to the expected number of pixels prior to their use in training as described in the further sections.


## Pytorch datasets

Once the data are available, a new dataset class and a dataloader are required to read images and parse them to the model for training and testing. 


### UNetDataset class to read images and masks for UNet-based segmentation


As a template to create custom datasets for segmentation with the UNet, the framework provides the ```UNet.UNet.data_handling.unetdataset.UNetDataset``` class derived from the base ```UNet.UNet.data_handling.base.BaseDataset``` class.


* The UNetDataset class is a subclass of the BaseDataset class and is designed to read images and masks for the UNet-based image segmentation task. It takes a list of image paths and mask paths as input during initialization.
* It expects data to have the structure as for the PH2 dataset. When you access an instance of the UNetDataset class with an index using the __getitem__ method, it reads the corresponding image and mask from the provided paths. It returns a dictionary with keys 'image' and 'mask', where the values are the loaded image and mask, respectively. Optionally, if a transform function is provided during initialization, it applies the transform on the sample before returning it.
* The class is initialized as follows:

```python

class UNetDataset(BaseDataset):

    def __init__(self,
                 required_image_size: Tuple[int, int],
                 required_mask_size: Tuple[int, int],
                 images_list: List[str],
                 masks_list: List[str],
                 resize_required: bool):
        """
        Initializes class to read images for UNet image segmentation into a dataset. Expects the full paths of images
        and masks to be provided at initialization.

        The file structure follows:
        * root_dir
            * sample1
                * sample1_images_tag
                    * sample1_optional_images_subtag.bmp
                * sample1_masks_tag
                    * sample1_optional_masks_subtag.bmp
            * sample2
                * sample2_images_tag
                    * sample1_optional_images_subtag.bmp
                * sample2_masks_tag
                    * sample2_optional_masks_subtag.bmp

        :param required_image_size: Image size as required by model
        :param required_mask_size: Mask size as required by model
        :param images_list: List of full paths to images
        :param masks_list: List of full paths to masks
        :param resize_required: If True, input images and masks will be resized

        :return sample: A dictionary with keys 'image' and 'mask' containing an image and a mask, respectively
        """

```



### Creating dataset-specific classes. Example: PH2Dataset class

Whilst UNetDataset class provides a generic template to work with data for UNet-based segmentation by reading and returning images and masks arranged under specific file structure, it requires a list of actual paths to these images and masks. Hence, for a new dataset one needs to create a new parent class that handles the dataset-specific namings in the file structure and, at this point, also any possible transformations. 

Specifically, for the PH2 dataset that has the following file structure

```
PH2Dataset
├── PH_Dataset_images
│    ├── IMD002
│    │    ├── IMD002_Dermoscopic_Image
│    │    │    └── IMD002.bmp
│    │    └── IMD002_lesion
│    │         └── IMD002_lesion.bmp
│    ├── IMD003
│    │    ├── IMD003_Dermoscopic_Image
│    │    │    └── IMD003.bmp
│    │    └── IMD003_lesion
│    │         └── IMD003_lesion.bmp
│    ├──...
├── PH2_dataset.txt
└── PH2_dataset.xlsx
```

we create a new class, ```UNet.UNet.data_handling.ph2dataset.PH2Dataset```, inheriting from ```UNet.UNet.data_handling.unetdataset.UNetDataset```. 

* The new class is specific to the PH2 data in a way that it sets names for expected images and masks folders, collects respective paths and sets a type of transform applied in the course of training. 

* To make a new class instance, provide the root directory argument setting the path to data, e.g. "/PH2Dataset/PH_Dataset_images/" for the example above and required sizes of images and masks as a tuple, e.g. ```(578, 578)```  to make sure they have shapes expected by the model. In future, we plan to introduce a separate class to pre-process images prior to training. Currently, the images are resized using ```ResizeData``` class upon their read-out from the disk in ```UNetDataset```.

```python
from UNet.data_handling.ph2dataset import PH2Dataset
import torch.nn as nn

self.ph2_dataset = PH2Dataset(root_dir=self.datapath,
                                required_image_size=self.size_images,
                                required_mask_size=self.size_masks,
                                resize_required=True)
```


## Pytorch dataloaders

Dataloaders provide functionality for initializing and managing data for training, validation, and testing. 

### BaseDataLoader class

The base```UNet.UNet.data_handling.base.BaseDataLoader``` class provides functionality to make custom dataloaders from your data. 

### Example: PH2 dataloader

* Given PH2 dataset obtained as above, we can now create a respective dataloader

```python
from UNet.data_handling.base import BaseDataLoader

self.data_loader = BaseDataLoader(dataset=self.ph2_dataset,
                                    batch_size=10,
                                    validation_split=0.2,
                                    test_split=0.1,
                                    shuffle_for_split=True,
                                    random_seed_split=42)
```

In here, the batch size is set to 10 images and the data are split randomly (with prior shuffling) 20% for validation, 10% for testing and the remaining 70% for training (the latter not specified). The random seed split is set to 42 for reproducibility purposes. 


