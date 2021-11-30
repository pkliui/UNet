import unittest
from ddt import ddt
import numpy as np
import torch

import shutil, tempfile


from UNet.models.unet import UNet


@ddt
class TestUNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # create an instance of BaseTrainer class
        self.unet = UNet()
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_arguments(self):
        """
        test input arguments are existing and are either None or equal to expected default values
        """
        for var in []:
            self.assertIn(var, self.unet.__dict__)

    def test_shape(self):
        """
        test the shapes of the image in intermediate layers
        """
        num_batches = 2
        x = torch.randn(num_batches, 3, 572, 572, dtype=torch.float32)
        _ = self.unet(x)

        self.assertEqual((self.unet.e0_fm.shape[1], self.unet.e0_fm.shape[2], self.unet.e0_fm.shape[3]), (64, 568, 568))
        self.assertEqual((self.unet.e0_pool_fm.shape[1], self.unet.e0_pool_fm.shape[2], self.unet.e0_pool_fm.shape[3]), (64, 284, 284))
        self.assertEqual((self.unet.e0_crop_fm.shape[1], self.unet.e0_crop_fm.shape[2], self.unet.e0_crop_fm.shape[3]), (64, 392, 392))
        self.assertEqual((self.unet.idx0_crop_fm.shape[1], self.unet.idx0_crop_fm.shape[2], self.unet.idx0_crop_fm.shape[3]), (64, 196, 196))

        self.assertEqual((self.unet.e1_fm.shape[1], self.unet.e1_fm.shape[2], self.unet.e1_fm.shape[3]), (128, 280, 280))
        self.assertEqual((self.unet.e1_pool_fm.shape[1], self.unet.e1_pool_fm.shape[2], self.unet.e1_pool_fm.shape[3]), (128, 140, 140))
        self.assertEqual((self.unet.e1_crop_fm.shape[1], self.unet.e1_crop_fm.shape[2], self.unet.e1_crop_fm.shape[3]), (128, 200, 200))
        self.assertEqual((self.unet.idx1_crop_fm.shape[1], self.unet.idx1_crop_fm.shape[2], self.unet.idx1_crop_fm.shape[3]), (128, 100, 100))

        self.assertEqual((self.unet.e2_fm.shape[1], self.unet.e2_fm.shape[2], self.unet.e2_fm.shape[3]), (256, 136, 136))
        self.assertEqual((self.unet.e2_pool_fm.shape[1], self.unet.e2_pool_fm.shape[2], self.unet.e2_pool_fm.shape[3]), (256, 68, 68))
        self.assertEqual((self.unet.e2_crop_fm.shape[1], self.unet.e2_crop_fm.shape[2], self.unet.e2_crop_fm.shape[3]), (256, 104, 104))
        self.assertEqual((self.unet.idx2_crop_fm.shape[1], self.unet.idx2_crop_fm.shape[2], self.unet.idx2_crop_fm.shape[3]), (256, 52, 52))

        self.assertEqual((self.unet.e3_fm.shape[1], self.unet.e3_fm.shape[2], self.unet.e3_fm.shape[3]), (512, 64, 64))
        self.assertEqual((self.unet.e3_pool_fm.shape[1], self.unet.e3_pool_fm.shape[2], self.unet.e3_pool_fm.shape[3]), (512, 32, 32))
        self.assertEqual((self.unet.e3_crop_fm.shape[1], self.unet.e3_crop_fm.shape[2], self.unet.e3_crop_fm.shape[3]), (512, 56, 56))
        self.assertEqual((self.unet.idx3_crop_fm.shape[1], self.unet.idx3_crop_fm.shape[2], self.unet.idx3_crop_fm.shape[3]), (512, 28, 28))

        self.assertEqual((self.unet.bneck_fm.shape[1], self.unet.bneck_fm.shape[2], self.unet.bneck_fm.shape[3]), (1024, 28, 28))

        self.assertEqual((self.unet.d3_upconv_fm.shape[1], self.unet.d3_upconv_fm.shape[2], self.unet.d3_upconv_fm.shape[3]), (512, 28, 28))
        self.assertEqual((self.unet.d3_upsample_fm.shape[1], self.unet.d3_upsample_fm.shape[2], self.unet.d3_upsample_fm.shape[3]), (512, 56, 56))
        self.assertEqual((self.unet.d3_concat_fm.shape[1], self.unet.d3_concat_fm.shape[2], self.unet.d3_concat_fm.shape[3]), (1024, 56, 56))
        self.assertEqual((self.unet.d3_fm.shape[1], self.unet.d3_fm.shape[2], self.unet.d3_fm.shape[3]), (512, 52, 52))

        self.assertEqual((self.unet.d2_upconv_fm.shape[1], self.unet.d2_upconv_fm.shape[2], self.unet.d2_upconv_fm.shape[3]), (256, 52, 52))
        self.assertEqual((self.unet.d2_upsample_fm.shape[1], self.unet.d2_upsample_fm.shape[2], self.unet.d2_upsample_fm.shape[3]), (256, 104, 104))
        self.assertEqual((self.unet.d2_concat_fm.shape[1], self.unet.d2_concat_fm.shape[2], self.unet.d2_concat_fm.shape[3]), (512, 104, 104))
        self.assertEqual((self.unet.d2_fm.shape[1], self.unet.d2_fm.shape[2], self.unet.d2_fm.shape[3]), (256, 100, 100))

        self.assertEqual((self.unet.d1_upconv_fm.shape[1], self.unet.d1_upconv_fm.shape[2], self.unet.d1_upconv_fm.shape[3]), (128, 100, 100))
        self.assertEqual((self.unet.d1_upsample_fm.shape[1], self.unet.d1_upsample_fm.shape[2], self.unet.d1_upsample_fm.shape[3]), (128, 200, 200))
        self.assertEqual((self.unet.d1_concat_fm.shape[1], self.unet.d1_concat_fm.shape[2], self.unet.d1_concat_fm.shape[3]), (256, 200, 200))
        self.assertEqual((self.unet.d1_fm.shape[1], self.unet.d1_fm.shape[2], self.unet.d1_fm.shape[3]), (128, 196, 196))

        self.assertEqual((self.unet.d0_upconv_fm.shape[1], self.unet.d0_upconv_fm.shape[2], self.unet.d0_upconv_fm.shape[3]), (64, 196, 196))
        self.assertEqual((self.unet.d0_upsample_fm.shape[1], self.unet.d0_upsample_fm.shape[2], self.unet.d0_upsample_fm.shape[3]), (64, 392, 392))
        self.assertEqual((self.unet.d0_concat_fm.shape[1], self.unet.d0_concat_fm.shape[2], self.unet.d0_concat_fm.shape[3]), (128, 392, 392))
        self.assertEqual((self.unet.d0_fm.shape[0], self.unet.d0_fm.shape[1], self.unet.d0_fm.shape[2], self.unet.d0_fm.shape[3]), (num_batches, 1, 388, 388))
