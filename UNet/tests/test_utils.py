import unittest
from ddt import ddt

import shutil
import tempfile
import os
import numpy as np
from PIL import Image

from UNet.data_handling.utils import get_list_of_data


@ddt
class TestUtils(unittest.TestCase):
    """
    test utils functions: get_list_of_data
    """

    def setUp(self):
        """
        create temporary image data having two pairs of images and masks

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
        num_images = 2
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
            #
            # add 2nd image to a folder to test raising error later
            if i == 3:
                image.save(os.path.join(image_folder, f"image{i}.png"), "PNG")
                image.save(os.path.join(image_folder, f"image{i}_2nd.png"), "PNG")
            else:
                image.save(image_file_path, "PNG")

    def test_list_files_in_dir_one_image(self):
        """check images are being append one by one"""
        list_of_images = []
        folder_with_image1 = os.path.join(self.test_dir, "sample_id1/sample_id1_image")
        list_of_images = get_list_of_data(folder_with_image1, 'png', list_of_images)
        self.assertEqual(list_of_images, [os.path.join(self.test_dir, "sample_id1/sample_id1_image/image1.png")])

        folder_with_image2 = os.path.join(self.test_dir, "sample_id2/sample_id2_image")
        list_of_images = get_list_of_data(folder_with_image2, 'png', list_of_images)
        self.assertEqual(list_of_images, [os.path.join(self.test_dir, "sample_id1/sample_id1_image/image1.png"),
                                          os.path.join(self.test_dir, "sample_id2/sample_id2_image/image2.png")])

    def test_list_files_in_dir_two_images(self):
        """check raising error if two images are in a folder"""
        list_of_images = []
        folder_with_image3 = os.path.join(self.test_dir, "sample_id3/sample_id3_image")
        with self.assertRaises(ValueError):
            _ = get_list_of_data(folder_with_image3, 'png', list_of_images)

