import unittest
from ddt import ddt

import shutil, tempfile

from UNet.classes.unetdataset import UNetDataset
from UNet.classes.preprocess import Resize, SplitDataLoader

import numpy as np
from torchvision import transforms

@ddt
class TestSplitDataLoader(unittest.TestCase):

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

    def test_splitting_dataset_1(self):
        """
        test splitting dataset
        directory structure 1
        """
        root_dir = "/UNet/data/test_reading_files/"
        images_folder = "images"
        masks_folder = "masks"
        files_structure = 1
        batch_size = 1

        im_size = 100
        transform = transforms.Compose([
            Resize((im_size,im_size),
                   (im_size,im_size))
        ])
        tr_size = 3
        vl_size = 3
        ts_size = 2
        #
        # read data into a dataset
        unet_data = UNetDataset(root_dir=root_dir, images_folder=images_folder,
                                masks_folder=masks_folder, transform=transform,
                                files_structure = files_structure)
        #
        #
        # make a dataloader from the dataset
        sdl = SplitDataLoader(dataset=unet_data,
                              batch_size=batch_size,
                              tr=tr_size, vl=vl_size, ts=ts_size)
        train_loader, val_loader, test_loader = sdl.split_and_load()

        import matplotlib.pyplot as plt
        for batch_idx, sample in enumerate(train_loader):
            print(batch_idx)
            print(sample["image"].shape)
            print(sample["mask"].shape)
            self.assertEqual(sample['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(sample['mask'].shape, (batch_size, im_size, im_size, 3))

        for batch_idx, sample in enumerate(val_loader):
            print(batch_idx)
            print(sample["image"].shape)
            print(sample["mask"].shape)
            self.assertEqual(sample['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(sample['mask'].shape, (batch_size, im_size, im_size, 3))


        for batch_idx, sample in enumerate(test_loader):
            print(batch_idx)
            print(sample["image"].shape)
            print(sample["mask"].shape)
            self.assertEqual(sample['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(sample['mask'].shape, (batch_size, im_size, im_size, 3))

    def test_splitting_dataset_2(self):
        """
        test splitting dataset
        directory structure 2
        """
        images_folder = "Dermoscopic_Image"
        masks_folder = "lesion"
        root_dir = "/UNet/data/PH2_Dataset_images/"
        files_structure = 2
        batch_size = 1

        im_size = 100
        transform = transforms.Compose([
            Resize((im_size, im_size),
                   (im_size, im_size))
        ])
        tr_size = 4
        vl_size = 4
        ts_size = 2
        #
        # read data into a dataset
        unet_data = UNetDataset(root_dir=root_dir, images_folder=images_folder,
                                masks_folder=masks_folder, transform=transform,
                                files_structure=files_structure)
        #
        #
        # make a dataloader from the dataset
        sdl = SplitDataLoader(dataset=unet_data,
                              batch_size=batch_size,
                              tr=tr_size, vl=vl_size, ts=ts_size)
        train_loader, val_loader, test_loader = sdl.split_and_load()

        import matplotlib.pyplot as plt
        for batch_idx, sample in enumerate(train_loader):
            print(batch_idx)
            print(sample["image"].shape)
            print(sample["mask"].shape)
            self.assertEqual(sample['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(sample['mask'].shape, (batch_size, im_size, im_size))

        for batch_idx, sample in enumerate(val_loader):
            print(batch_idx)
            print(sample["image"].shape)
            print(sample["mask"].shape)
            self.assertEqual(sample['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(sample['mask'].shape, (batch_size, im_size, im_size))

        for batch_idx, sample in enumerate(test_loader):
            print(batch_idx)
            print(sample["image"].shape)
            print(sample["mask"].shape)
            self.assertEqual(sample['image'].shape, (batch_size, im_size, im_size, 3))
            self.assertEqual(sample['mask'].shape, (batch_size, im_size, im_size))

    @ddt
    class TestResize(unittest.TestCase):

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
