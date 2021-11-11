import unittest
from ddt import ddt

import shutil, tempfile

from matplotlib import pyplot as plt

from UNet.classes.unetdataset import UNetDataset
from UNet.classes.preprocess import Resize, SplitDataLoader

import numpy as np
from torchvision import transforms

@ddt
class TestUNetDateset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # create an instance of UNetDataset class
        self.und = UNetDataset()
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_arguments(self):
        """
        test input arguments are existing and are either None or equal to expected default values
        """
        for var in ["root_dir", "images_folder", "masks_folder", "transform"]:
            self.assertIn(var, self.und.__dict__)
            self.assertEqual(self.und.__dict__[var], None)

    def test_making_dataset_1(self):
        """
        test making a dataset by reading images and masks from a directory and transforming them
        expects RGB images and RGB masks
        directory structure 1
        """
        # specify the roo dir, images' and masks' folders and the image size
        root_dir = "/UNet/data/test_reading_files/"
        images_folder = "images"
        masks_folder = "masks"
        im_size = 100
        #
        # specify an image transform function
        transform = transforms.Compose([
            Resize((im_size,im_size),
                   (im_size,im_size))
        ])
        #
        # read masks and images into a dataset
        unet_data = UNetDataset(root_dir=root_dir, images_folder=images_folder,
                                masks_folder=masks_folder, transform=transform, files_structure=1)
        #
        # I expect to see a list of dictionaries,
        # where each dictionary has an "image" and a "mask" keys and he corresponding arrays as values
        # I don't check here if the pixel values were correctly read, just the overall output structure
        #
        # check there are 8 entries in the list (8 different dictionaries with masks and images)
        #print(list(unet_data))
        #print("list length: ", len(list(unet_data)))
        self.assertEqual(len(list(unet_data)), 8)
        # and each dictionary contains 2 entries
        #print([len(ii) for ii in list(unet_data)])
        self.assertEqual([len(ii) for ii in list(unet_data)], [2]*8)


    def test_making_dataset_2(self):
        """
        test making a dataset by reading images and masks from a directory and transforming them
        expect RGB images and RGB masks
        directory structure 2
        """
        images_folder = "Dermoscopic_Image"
        masks_folder = "lesion"
        root_dir = "/UNet/data/PH2_Dataset_images/"
        im_size = 100
        #
        transform = transforms.Compose([
            Resize((im_size,im_size),
                   (im_size,im_size))
        ])
        #
        # note here the file_structure is 2!
        unet_data = UNetDataset(root_dir=root_dir, images_folder=images_folder,
                                masks_folder=masks_folder, transform=transform, files_structure=2)
        #
        #
        # I expect to see a list of dictionaries,
        # where each dictionary has an "image" and a "mask" keys and he corresponding arrays as values
        # I don't check here if the pixel values were correctly read, just the overall output structure
        #
        # check there are 10 entries in the list (10 different dictionaries with masks and images)
        #print(list(unet_data))
        #print("list length: ", len(list(unet_data)))
        self.assertEqual(len(list(unet_data)), 10)
        # and each dictionary contains 2 entries
        #print([len(ii) for ii in list(unet_data)])
        self.assertEqual([len(ii) for ii in list(unet_data)], [2]*10)
