import unittest
from ddt import ddt

import shutil
import tempfile
import os
import numpy as np
from PIL import Image

from UNet.data_handling.base import BaseDataLoader
from UNet.data_handling.unetdataset import UNetDataset


@ddt
class TestUNetDateset(unittest.TestCase):
    """
    test UNetDataset class and functions therein
    """

    def setUp(self):
        """
        create some temporary image data

        Test instance variables created in this method:

        test_dir: temporary directory to keep data
        images_list: list of paths to created images
        masks_list: list of paths to created masks
        """
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        #
        # Create dummy image and mask folders and files
        sample_data = []
        num_images = 10
        image_size_linear = 574
        mask_size_linear = 390

        for i in range(1, num_images + 1):
            sample_id = f"sample_id{i}"
            image_file = f"image{i}.png"
            mask_file = f"mask{i}.png"
            sample_data.append({"sample_id": sample_id,
                                "image_file": image_file,
                                "mask_file": mask_file})

            # make sample folder containing images and masks
            sample_folder = os.path.join(self.test_dir, sample_id)
            os.makedirs(sample_folder)

            # make image folder and create a dummy image in it
            image_folder = os.path.join(sample_folder, f"{sample_id}_image")
            os.makedirs(image_folder)
            image_file_path = os.path.join(image_folder, image_file)
            random_array = np.random.randint(low=0, high=256, size=(image_size_linear, image_size_linear, 3),
                                             dtype=np.uint8)
            image = Image.fromarray(random_array, 'RGB')
            image.save(image_file_path, "PNG")

            # make mask folder and create a dummy mask in it
            mask_folder = os.path.join(sample_folder, f"{sample_id}_mask")
            os.makedirs(mask_folder)
            mask_file_path = os.path.join(mask_folder, mask_file)
            random_array = np.random.randint(low=0, high=1, size=(mask_size_linear, mask_size_linear), dtype=np.uint8)
            mask = Image.fromarray(random_array)
            mask.save(mask_file_path, "PNG")

        # get paths to images and masks
        self.images_list = [os.path.join(self.test_dir,
                                         sample["sample_id"],
                                         f"{sample['sample_id']}_image",
                                         sample["image_file"]) for sample in sample_data]
        self.masks_list = [os.path.join(self.test_dir,
                                        sample["sample_id"],
                                        f"{sample['sample_id']}_mask",
                                        sample["mask_file"]) for sample in sample_data]

    def tearDown(self):
        """remove temporary directory after the test"""
        shutil.rmtree(self.test_dir)

    def test_making_dataset_train(self):
        """
        test creating a dataloader for training data
        """
        batch_size = 2

        image_size_linear = 572
        mask_size_linear = 388

        # make unetdataset
        unet_data = UNetDataset(images_list=self.images_list,
                                masks_list=self.masks_list,
                                resize_required=True,
                                required_image_size=(image_size_linear, image_size_linear),
                                required_mask_size=(mask_size_linear, mask_size_linear))

        # make a dataloader from the dataset
        data_loader = BaseDataLoader(dataset=unet_data,
                                     batch_size=batch_size,
                                     validation_split=0,
                                     shuffle_for_split=True,
                                     random_seed_split=0)

        # check the batch size
        for batch in data_loader.train_loader:
            self.assertEqual(batch['image'].shape, (2, image_size_linear, image_size_linear, 3))
            self.assertEqual(batch['mask'].shape, (2, mask_size_linear, mask_size_linear))

    def test_making_dataset_train_val_60_40(self):
        """
        test creating a 60-40  dataloader for training and validation data
        from 10 images
        6-4 split in terms if number of images in training-validation sets, respectively
        3-2 split in terms if batches in training-validation sets, respectively
        """
        batch_size = 2

        image_size_linear = 572
        mask_size_linear = 388

        unet_data = UNetDataset(images_list=self.images_list,
                                masks_list=self.masks_list,
                                resize_required=True,
                                required_image_size=(image_size_linear, image_size_linear),
                                required_mask_size=(mask_size_linear, mask_size_linear))

        # make a dataloader from the dataset
        data_loader = BaseDataLoader(dataset=unet_data,
                                     batch_size=batch_size,
                                     validation_split=0.4,
                                     shuffle_for_split=True,
                                     random_seed_split=0)
        #
        # check the number of batches
        self.assertEqual(len(data_loader.train_loader), 3)
        self.assertEqual(len(data_loader.val_loader), 2)
        #
        # check individual batch sizes
        for batch in data_loader.train_loader:
            self.assertEqual(batch['image'].shape, (batch_size, image_size_linear, image_size_linear, 3))
            self.assertEqual(batch['mask'].shape, (batch_size, mask_size_linear, mask_size_linear))
        for batch in data_loader.val_loader:
            self.assertEqual(batch['image'].shape, (batch_size, image_size_linear, image_size_linear, 3))
            self.assertEqual(batch['mask'].shape, (batch_size, mask_size_linear, mask_size_linear))

    def test_making_dataset_train_val_90_10(self):
        """
        test creating a 90-10  dataloader for training and validation data
        from 10 images
        9-1 split in terms if number of images in training-validation sets, respectively

        :raise ValueError as the batch size (2) is larger than the validation set size (1)
        """
        batch_size = 2

        image_size_linear = 572
        mask_size_linear = 388

        unet_data = UNetDataset(images_list=self.images_list,
                                masks_list=self.masks_list,
                                resize_required=True,
                                required_image_size=(image_size_linear, image_size_linear),
                                required_mask_size=(mask_size_linear, mask_size_linear))
        # make a dataloader from the dataset
        with self.assertRaises(ValueError):
            _ = BaseDataLoader(dataset=unet_data,
                               batch_size=batch_size,
                               validation_split=0.1,
                               shuffle_for_split=True,
                               random_seed_split=0)

    def test_making_dataset_train_val_test_70_20_10(self):
        """
        test creating a 70-20-10  dataloader for training, validation and testing data
        from 10 images
        7-2-1 split in terms if number of images in training-validation-testing sets, respectively

        :raise ValueError as the batch size (2) is larger than the test set size (1)
        """
        batch_size = 2

        image_size_linear = 572
        mask_size_linear = 388

        unet_data = UNetDataset(images_list=self.images_list,
                                masks_list=self.masks_list,
                                resize_required=True,
                                required_image_size=(image_size_linear, image_size_linear),
                                required_mask_size=(mask_size_linear, mask_size_linear))
        # make a dataloader from the dataset
        with self.assertRaises(ValueError):
            _ = BaseDataLoader(dataset=unet_data,
                               batch_size=batch_size,
                               validation_split=0.2,
                               test_split=0.1,
                               shuffle_for_split=True,
                               random_seed_split=0)
