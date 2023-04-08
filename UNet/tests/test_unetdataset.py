import unittest
from ddt import ddt

import shutil, tempfile

from matplotlib import pyplot as plt

from UNet.data_handling.unetdataset import UNetDataset
from UNet.classes.preprocess import Resize
from UNet.data_handling.base import BaseDataLoader
from torch.utils.data import DataLoader, SubsetRandomSampler

from torch.utils.data.dataset import Dataset

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

    # def setUp(self):
    #     # create an instance of UNetDataset class
    #     self.und = UNetDataset()
    #     # create a temporary directory
    #     self.test_dir = tempfile.mkdtemp()
    #
    # def tearDown(self):
    #     # remove temporary directory after the test
    #     shutil.rmtree(self.test_dir)
    #
    # def test_arguments(self):
    #     """
    #     test input arguments are existing and are either None or equal to expected default values
    #    """
    #     for var in ["transform"]:
    #         self.assertIn(var, self.und.__dict__)
    #         self.assertEqual(self.und.__dict__[var], None)

    def test_making_dataset_train(self):
        """
        test creating a dataloader for training data
        """
        # specify the roo dir, images' and masks' folders and the image size
        extension = "*.bmp"
        #extension = "*.jpeg"
        root_dir = "/Users/Pavel/Documents/repos/UNet/docs/data/PH2_Dataset_images/"
        #root_dir = "/Users/Pavel/Documents/repos/UNet/docs/data/test_reading_files/"
        images_folder = "images"
        masks_folder = "masks"
        im_size = 10
        batch_size = 2
        #
        # specify an image transform function
        transform = transforms.Compose([Resize((im_size,im_size), (im_size,im_size))])
        #
        # read masks and images
        unet_data = UNetDataset(root_dir, images_folder, masks_folder, extension, transform=transform)
        # make a dataloader from the dataset
        data_loader = BaseDataLoader(dataset=unet_data,
        batch_size=batch_size,
        validation_split=0,
        shuffle_for_split=True,
        random_seed_split=0)

        # check the batch size
        for batch in data_loader.train_loader:
            self.assertEqual(batch['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(batch['mask'].shape, (batch_size, im_size, im_size))


    def test_making_dataset_train_val_50_50(self):
        """
        test creating a 50-50 dataloader for training and validation data
        """
        # specify the root dir, images' and masks' folders and the image size
        extension = "*.bmp"
        #extension = "*.jpeg"
        root_dir = "/Users/Pavel/Documents/repos/UNet/docs/data/PH2_Dataset_images/"
        images_folder = "images"
        masks_folder = "masks"
        im_size = 100
        batch_size = 2
        validation_split = 0.5
        shuffle_for_split = True
        random_seed_split = 0
        #
        # specify an image transform function
        transform = transforms.Compose([Resize((im_size,im_size), (im_size,im_size))])
        #
        # read masks and images
        unet_data = UNetDataset(root_dir, images_folder, masks_folder, extension, transform=transform)
        # make a dataloader from the dataset
        data_loader = BaseDataLoader(dataset=unet_data,
                                     batch_size=batch_size,
                                     validation_split=validation_split,
                                     shuffle_for_split=shuffle_for_split,
                                     random_seed_split=random_seed_split)

        # check the batch size of the train data
        for batch in data_loader.train_loader:
            self.assertEqual(batch['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(batch['mask'].shape, (batch_size, im_size, im_size))

        # check the batch size of the validation data
        for batch in data_loader.val_loader:
            self.assertEqual(batch['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(batch['mask'].shape, (batch_size, im_size, im_size))

    def test_making_dataset_train_val_90_10(self):
        """
        test creating a 90-10 dataloader for training and validation data
        """
        # specify the root dir, images' and masks' folders and the image size
        extension = "*.bmp"
        #extension = "*.jpeg"
        root_dir = "/Users/Pavel/Documents/repos/UNet/docs/data/PH2_Dataset_images/"
        images_folder = "images"
        masks_folder = "masks"
        im_size = 100
        batch_size = 2
        validation_split = 0.1
        shuffle_for_split = True
        random_seed_split = 0
        #
        # specify an image transform function
        transform = transforms.Compose([Resize((im_size,im_size), (im_size,im_size))])
        #
        # read masks and images
        unet_data = UNetDataset(root_dir, images_folder, masks_folder, extension, transform=transform)
        #
        # make a dataloader from the dataset
        # assert there is a value error as the batch size is larger than the validation set size
        with self.assertRaises(ValueError):
            _ = BaseDataLoader(dataset=unet_data,
                                         batch_size=batch_size,
                                         validation_split=validation_split,
                                         shuffle_for_split=shuffle_for_split,
                                         random_seed_split=random_seed_split)

    def test_making_dataset_train_val_10_90(self):
        """
        test creating a 10-90 dataloader for training and validation data
        """
        # specify the root dir, images' and masks' folders and the image size
        extension = "*.bmp"
        #extension = "*.jpeg"
        root_dir = "/Users/Pavel/Documents/repos/UNet/docs/data/PH2_Dataset_images/"
        images_folder = "images"
        masks_folder = "masks"
        im_size = 100
        batch_size = 2
        validation_split = 0.9
        shuffle_for_split = True
        random_seed_split = 0
        #
        # specify an image transform function
        transform = transforms.Compose([Resize((im_size,im_size), (im_size,im_size))])
        #
        # read masks and images
        unet_data = UNetDataset(root_dir, images_folder, masks_folder, extension, transform=transform)
        #
        # make a dataloader from the dataset
        # assert there is a value error as the batch size is larger than the train set size
        with self.assertRaises(ValueError):
            _ = BaseDataLoader(dataset=unet_data,
                               batch_size=batch_size,
                               validation_split=validation_split,
                               shuffle_for_split=shuffle_for_split,
                               random_seed_split=random_seed_split)
