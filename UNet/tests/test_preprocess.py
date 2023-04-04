import unittest
import shutil
import tempfile

from ddt import ddt
import numpy as np

from UNet.classes.preprocess import Resize


@ddt
class TestResize(unittest.TestCase):

    def setUp(self):
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)


    def test_resize_dataset_1(self):
        """
        test resizing a dataset by nearest neighbour interpolation
        """
        # the  size to resize to
        new_size = (4,4)
        #
        # input images amd masks
        images = [
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[0.999, 0.999], [0.999, 1.0]]),
            np.array([[1.0, 0.6], [1.0, 0.6]]),
            np.array([[1.0, 0.001], [1.0, 0.001]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[0.6, 0.0], [0.3, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]])
            ]
        masks = [
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]])
        ]
        #
        # expected ground truth images and masks
        images_gt = [
            np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.999, 0.999,0.999, 0.999], [0.999, 0.999, 0.999, 0.999], [0.999, 0.999, 1.0, 1.0], [0.999, 0.999, 1.0, 1.0]]),
            np.array([[1.0, 1.0, 0.6, 0.6], [1.0, 1.0, 0.6, 0.6], [1.0, 1.0, 0.6, 0.6], [1.0, 1.0, 0.6, 0.6]]),
            np.array([[1.0, 1.0, 0.001, 0.001], [1.0, 1.0, 0.001, 0.001], [1.0, 1.0, 0.001, 0.001], [1.0, 1.0, 0.001, 0.001]]),
            np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
            np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        ]
        masks_gt = [
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]),
            np.array([[0.0, 0.0, 0, 0], [0.0, 0.0, 0, 0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]),
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]),
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]),
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [0.0, 0.0, 0, 0], [0.0, 0.0, 0, 0]]),
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        ]

        for image, mask, image_gt, mask_gt in zip(images, masks, images_gt, masks_gt):
            #
            # make a dictionary from original data
            original = {"image": image, "mask": mask}
            #
            # pass the dictionary
            rsz = Resize(new_size, new_size)
            resized = rsz.__call__(original)
            #
            # check if resized originals match ground truth
            self.assertTrue(np.allclose(image_gt, resized["image"]))
            self.assertTrue(np.allclose(mask_gt, resized["mask"]))
