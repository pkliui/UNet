import unittest
from ddt import ddt
import numpy as np
import torch

import shutil, tempfile


from UNet.data_handling.base import BaseDataLoader


@ddt
class TestBaseDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # create an instance of BaseDataLoader class
        #self.bdl = BaseDataLoader()
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
        test missing positional arguments
        test to read some non-existing data
        """
        with self.assertRaises(TypeError):
            self.bdl.read_data()
        with self.assertRaises(ValueError):
            self.bdl.read_data(datapath="some_nonsense_path", images_folder="nonsense_images", masks_folder="nonsense_masks")

