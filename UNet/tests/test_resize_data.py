import unittest
from ddt import ddt

import shutil
import tempfile
import os
import numpy as np
from PIL import Image

from UNet.data_handling.base import BaseDataLoader
from UNet.data_handling.unetdataset import UNetDataset
from UNet.utils.resize_data import resizer, ResizeData


@ddt
class TestResizeData(unittest.TestCase):
    """
    test ResizeData class and related functions
    """

    def setUp(self):
        """
        create a temp dir

        Test instance variables created in this method:
        test_dir: temporary directory to keep data
        """
        self.test_dir = tempfile.mkdtemp()

    def create_data(self, image_size_linear = 574, mask_size_linear = 390):
        """
        create some temporary image data
        """
        # Create dummy image and mask folders and files
        sample_data = []
        sample_id = "sample_id"
        image_file = "image.png"
        mask_file = "mask.png"
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
        image = np.random.randint(low=0, high=256, size=(image_size_linear, image_size_linear, 3),
                                         dtype=np.uint8)

        # make mask folder and create a dummy mask in it
        mask_folder = os.path.join(sample_folder, f"{sample_id}_mask")
        os.makedirs(mask_folder)
        mask_file_path = os.path.join(mask_folder, mask_file)
        mask = np.random.randint(low=0, high=1, size=(mask_size_linear, mask_size_linear), dtype=np.uint8)

        return image, mask


    def test_resizer_even_pxl_number_larger(self):
        """
        test resizing when input has odd linear number of pixels larger than the new number of pxls
        """
        image, _ = self.create_data(image_size_linear=574, mask_size_linear=390)

        image_resized = resizer(image, (572, 572))
        self.assertEqual(image_resized.shape, (572, 572, 3))

    def test_resizer_odd_pxl_number_larger(self):
        """
        test resizing when input has even linear number of pixels larger than the new number of pxls
        """
        image, _ = self.create_data(image_size_linear=573, mask_size_linear=390)

        image_resized = resizer(image, (572, 572))
        self.assertEqual(image_resized.shape, (572, 572, 3))

    def test_resizer_even_pxl_number_smaller(self):
        """
        test resizing when input has smaller linear number of pixels  than the new number of pixels
        """
        image, _ = self.create_data(image_size_linear=570, mask_size_linear=390)

        with self.assertRaises(ValueError):
            image_resized = resizer(image, (572, 572))
            self.assertEqual(image_resized.shape, (572, 572, 3))

    def test_resizedata_class_even_pxl_number_larger(self):
        """
        test resizing when input has even linear number of pixels larger than the new number of pxls
        """
        image, mask = self.create_data(image_size_linear=574, mask_size_linear=390)
        required_image_size = (572, 572)
        required_image_size_for_comparison = (572, 572, 3)
        required_mask_size = (388, 388)
        resizedata = ResizeData(required_image_size, required_mask_size)
        data_resized = resizedata({'image': image, 'mask': mask})

        self.assertEqual(data_resized['image'].shape, required_image_size_for_comparison)

    def test_resizedata_class_odd_pxl_number_larger(self):
        """
        test resizing when input has odd linear number of pixels larger than the new number of pxls
        """
        image, mask = self.create_data(image_size_linear=573, mask_size_linear=390)
        required_image_size = (572, 572)
        required_image_size_for_comparison = (572, 572, 3)
        required_mask_size = (388, 388)
        resizedata = ResizeData(required_image_size, required_mask_size)
        data_resized = resizedata({'image': image, 'mask': mask})

        self.assertEqual(data_resized['image'].shape, required_image_size_for_comparison)

    def test_resizedata_class_even_pxl_number_smaller(self):
        """
        test resizing when input has even linear number of pixels smaller than the new number of pxls
        """
        image, mask = self.create_data(image_size_linear=571, mask_size_linear=390)
        required_image_size = (572, 572)
        required_image_size_for_comparison = (572, 572, 3)
        required_mask_size = (388, 388)

        with self.assertRaises(ValueError):
            resizedata = ResizeData(required_image_size, required_mask_size)
            data_resized = resizedata({'image': image, 'mask': mask})
