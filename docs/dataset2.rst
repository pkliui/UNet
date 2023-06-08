Dataset creation
=====

This document describes the dataset format used by SkinSeg framework for segmentation of dermatoscopic images. After creating the dataset one can keep it locally or upload to AzureML blob storage.

PH2 Dataset
------------

As an example for training, the framework uses the `PH2 dermoscopic image database <https://www.fc.up.pt/addi/ph2%20database.html>`_ acquired at the Dermatology Service of Hospital Pedro Hispano, Matosinhos, Portugal. They are 8-bit RGB color images with a resolution of 768x560 pixels of lesions (we refer to them as *images*) and their segmentations available as binary masks (we refer to them as *masks*). In total the database comprises 200 datasets and corresponding files with metadata:

* *PH2 Dataset images* folder contains the image data.
* *PH2 Dataset.xlsx* file contains the classification of all images according to the dermoscopic criteria that are evaluated in the PH2 database.
* *PH2 Dataset.txt* file contains the classification of all images ccording to the dermoscopic criteria that are evaluated in the PH2 database.

For more details please refer to the original website. The PH2 dataset sets requirements for the file structure currently supported by the framework. It is possible to use any other dataset if it fulfills the following requirements:

* Images are saved in png or bmp formats. For each image there is a corresponding mask. 
* Following the architecture of UNet used for segmentation, the sampling of images must be no less than 572x572 pixels and the sampling of masks no less than 388x388 pixels. The data are then automatically pre-processed in terms of their size accordingly prior to their use by framework.
* Each set of data has a unique identifier *Image Name* and resides in a separate folder with this particular ID name.

Currently, the file structure for segmentation using the PH2 database looks like:

.. code-block:: console

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


Making a pytorch dataset
------------


UNetDataset
~~~~~~~~~~~~~~~~~


Example: PH2 dataset
~~~~~~~~~~~~~~~~~

* Create a new dataset using PH2 data from the ```PH2Dataset``` class as follows:


.. code-block:: python

    DATAPATH = '/path/to/PH_Dataset_images/'
    SIZE_IMAGES = (572,572) # follows the unet guidelines
    SIZE_MASKS = (388, 388) 
    ph2_dataset = PH2Dataset(
        root_dir=DATAPATH,
        transform=nn.Sequential(Resize(SIZE_IMAGES, SIZE_MASKS)))

* The ```PH2Dataset``` class inherits from the UNetDataset class. The ```UNetDataset``` class is specific to the UNet segmentation for the file structure as above. It returns a dictionary 

* 



