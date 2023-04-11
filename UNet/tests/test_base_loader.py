import os
import unittest
from ddt import ddt
import shutil, tempfile


from UNet.data_handling.base import BaseDataLoader
from UNet.data_handling.unetdataset import UNetDataset


@ddt
class TestBaseDataLoader(unittest.TestCase):

    def setUp(self):
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_arguments(self):
        """
        test input arguments are existing and are either None or equal to expected default values
        """

        self.bdl = BaseDataLoader(dataset=["somedummydata", "somedummydata"],
                 batch_size=1,
                 validation_split=0,
                 shuffle_for_split=True,
                 random_seed_split=0)
        vals = [["somedummydata", "somedummydata"], 1, 0]
        print(self.bdl.__dict__)
        for idx, var in enumerate(["dataset", "batch_size", "validation_split"]):
            self.assertIn(var, self.bdl.__dict__)
            self.assertEqual(self.bdl.__dict__[var], vals[idx])

    def test_read_data(self):
        """
        test reading some non-existing data
        """
        # instantiate class
        self.bdl = BaseDataLoader(dataset=["somedummydata", "somedummydata"],
                                  batch_size=1,
                                  validation_split=0.5,
                                  shuffle_for_split=True,
                                  random_seed_split=0)
        self.assertEqual(self.bdl.dataset, ["somedummydata", "somedummydata"])
        self.assertEqual(self.bdl.batch_size, 1)
        self.assertEqual(self.bdl.validation_split, 0.5)

    def test_read_batch(self):
        """
        test reading batches of existing data
        """
        # set args
        DATAPATH = os.path.abspath("/Users/Pavel/Documents/repos/UNet/docs/data/PH2_Dataset_images/")
        images_folder = "images"
        masks_folder = "masks"
        extension = "*.bmp"
        transform = None
        # read masks and images
        unet_data = UNetDataset(extension=extension,
                                root_dir=DATAPATH,
                                images_folder=images_folder,
                                masks_folder=masks_folder,
                                transform=transform)

        #
        # set args for dataloader
        batch_size = 1
        validation_split = 0.75
        # create the corresponding dataloader for training and validation
        self.bdl = BaseDataLoader(dataset=unet_data,
                                     batch_size=batch_size,
                                     validation_split=validation_split,
                                  shuffle_for_split=True,
                                  random_seed_split=0)
        # check the batch size
        self.assertEqual(self.bdl.batch_size, 1)
        # check the validation split
        self.assertEqual(self.bdl.validation_split, 0.75)
        # check the number of batches for validation
        self.assertEqual(len(self.bdl.val_loader), 3)
        # check the number of batches for training
        self.assertEqual(len(self.bdl.train_loader), 1)



