import unittest
from ddt import ddt

import shutil, tempfile

from UNet.classes.readandprocess import ReadAndProcess

import numpy as np

@ddt
class TestReadAndProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # create an instance of ReadAndProcess class
        self.readproc = ReadAndProcess()
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_arguments(self):
        """
        test input arguments are existing and are either None or equal to expected default values
        """
        for var in ["datapath", "images_folder", "masks_folder", "images", "masks"]:
            self.assertIn(var, self.readproc.__dict__)
            self.assertEqual(self.readproc.__dict__[var], None)
        for var in ["size_X"]:
            self.assertIn(var, self.readproc.__dict__)
            self.assertEqual(self.readproc.__dict__[var], (572,572))
        for var in ["size_Y"]:
            self.assertIn(var, self.readproc.__dict__)
            self.assertEqual(self.readproc.__dict__[var], (388,388))

    def test_read_data(self):
        """
        test missing positional arguments
        test to read some non-existing data
        """
        with self.assertRaises(TypeError):
            self.readproc.read_data()
        with self.assertRaises(ValueError):
            self.readproc.read_data(datapath="some_nonsense_path", images_folder="nonsense_images", masks_folder="nonsense_masks")

    def test_read_data(self):
        datapath = "/Users/Pavel/Documents/repos/machine-learning/stepik-deep-learning/16-HW-semantic-segmentation/UNet/UNet/data/test_reading_files"
        images_folder = "images"
        masks_folder = "masks"
        self.readproc.read_data(datapath, images_folder, masks_folder)

    def test_resize_data(self):
        """
        test resizing images and masks
        """
        #
        # initialize new class instance
        self.readproc = ReadAndProcess()
        # input images amd masks
        self.readproc.images = [
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[0.999, 0.999], [0.999, 1.0]]),
            np.array([[1.0, 0.6], [1.0, 0.6]]),
            np.array([[1.0, 0.001], [1.0, 0.001]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[0.6, 0.0], [0.3, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]])
            ]

        self.readproc.masks = [
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]])
        ]

        # resize
        size_X = (4,4)
        size_Y = (4,4)
        self.readproc.resize_data(size_X, size_Y)
        #
        # expected ground truth
        gt_resized_images = [
            np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.999, 0.999,0.999, 0.999], [0.999, 0.999, 0.999, 0.999], [0.999, 0.999, 1.0, 1.0], [0.999, 0.999, 1.0, 1.0]]),
            np.array([[1.0, 1.0, 0.6, 0.6], [1.0, 1.0, 0.6, 0.6], [1.0, 1.0, 0.6, 0.6], [1.0, 1.0, 0.6, 0.6]]),
            np.array([[1.0, 1.0, 0.001, 0.001], [1.0, 1.0, 0.001, 0.001], [1.0, 1.0, 0.001, 0.001], [1.0, 1.0, 0.001, 0.001]]),
            np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
            np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        ]
        gt_resized_masks = [
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]),
            np.array([[0.0, 0.0, 0, 0], [0.0, 0.0, 0, 0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]),
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]),
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]),
            np.array([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0], [0.0, 0.0, 0, 0], [0.0, 0.0, 0, 0]]),
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        ]

        #print("self.readproc.images ", self.readproc.images)
        #print("self.readproc.masks ", self.readproc.masks)
        #print("images_resized ", gt_resized_images)
        #print("masks_resized ", gt_resized_masks)
        #
        # compare ground truth images with the returned images of resize_data
        self.assertTrue(np.allclose(gt_resized_images, self.readproc.images))
        self.assertTrue(np.allclose(gt_resized_masks, self.readproc.masks))

    def test_split_data(self):
        """
        tests splitting data into train-val-test sets
        """
        #
        # initialize new class instance
        self.readproc = ReadAndProcess()
        #
        # generate 200 new images and masks
        self.readproc.images = [np.ones((4, 4, 3)) for _ in range(200)]
        self.readproc.masks = [np.ones((4, 4)) for _ in range(200)]
        #
        # resize before splitting
        self.readproc.resize_data((2,2), (2,2))
        #
        # split and check for batch size =  positive integer
        self.readproc.split_data(100, 50, 50, 1)
        self.assertEqual(len(self.readproc.train), 100)
        self.assertEqual(len(self.readproc.val), 50)
        self.assertEqual(len(self.readproc.test), 50)
        #
        self.readproc.split_data(100, 50, 50, 2)
        self.assertEqual(len(self.readproc.train), 50)
        self.assertEqual(len(self.readproc.val), 25)
        self.assertEqual(len(self.readproc.test), 25)
        #
        self.readproc.split_data(100, 50, 50, 3)
        self.assertEqual(len(self.readproc.train), 33)
        self.assertEqual(len(self.readproc.val), 16)
        self.assertEqual(len(self.readproc.test), 16)
        #
        # check for batch size = 0
        with self.assertRaises(ValueError):
            self.readproc.split_data(100, 50, 50, 0)


    # def test_run_all(self):
    #
    #
    #     images1 = [np.ones((4, 4, 3)) for _ in range(0,200)]
    #     masks1 = [np.ones((4, 4)) for _ in range(0,200)]
    #     #
    #     #
    #     # will run all the def's in the same time
    #     self.readproc = ReadAndProcess(images = images1, masks = masks1,
    #                             size_X = (2,2), size_Y=(2,2),
    #                             train_share = 100, val_share=50, test_share=50, batch_size=1)
    #     self.assertEqual(len(self.readproc.train), 100)


        #self.assertEqual(len(self.readproc.val), 50)
        #self.assertEqual(len(self.readproc.test), 50)
        #
        #self.readproc = ReadAndProcess()
        #self.readproc = ReadAndProcess(images = images1, masks = masks1,
        #                        size_X = (2,2), size_Y=(2,2),
        #                        train_share = 100, val_share=50, test_share=50, batch_size=3)

        #self.assertEqual(len(self.readproc.train), 33)
        #self.assertEqual(len(self.readproc.val), 16)
        #self.assertEqual(len(self.readproc.test), 16)
