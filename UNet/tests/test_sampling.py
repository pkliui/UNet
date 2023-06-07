import numpy as np
import unittest
import shutil
import tempfile
import torch
import torch.testing
from ddt import ddt

from UNet.utils.sampling import zero_pad_masks, unpad_transformed_masks


@ddt
class TestSampling(unittest.TestCase):

    def setUp(self):
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_zero_pad_masks_odd_diff(self):
        """
        test zero-padding of a mask for odd difference in number of pixels
        :return:
        """
        images_batch = np.array([[[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]],

                                 [[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]],

                                 [[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]]])
        # make it batch of size 1
        images_batch = torch.tensor(images_batch[np.newaxis, :, :, :])

        masks_batch = np.array([[[0., 0., 0., 0.],
                                 [0., 0., 1., 1.],
                                 [0., 0., 1., 1.],
                                 [0., 0., 0., 0.]]])
        # make it batch of size 1
        masks_batch = torch.tensor(masks_batch[np.newaxis, :, :])

        # created expected zero-padded mask
        mask_0_expected = torch.tensor([[[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]],

                                        [[0., 0., 0., 0., 0.],
                                         [0., 0., 1., 1., 0.],
                                         [0., 0., 1., 1., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]],

                                        [[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]]], dtype=torch.float64)
        # print("mask before  ", masks_batch[0])
        # print("mask before ", masks_batch[0].shape)
        mask_0_pad = zero_pad_masks(images_batch[0], masks_batch[0])
        # print(" mask pad  ", mask_0_pad)

        torch.testing.assert_allclose(mask_0_pad, mask_0_expected)


    def test_zero_pad_masks_even_diff(self):
        """
        test zero-padding of a mask for even difference in number of pixels
        :return:
        """
        # create image batch
        images_batch = np.array([[[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]],

                                 [[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]],

                                 [[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]]])
        # make it batch of size 1
        images_batch = torch.tensor(images_batch[np.newaxis, :, :, :])

        # create mask batch
        masks_batch = np.array([[[0., 0., 0.],
                                 [0., 1., 1.],
                                 [0., 1., 1.]]])
        # make it batch of size 1
        masks_batch = torch.tensor(masks_batch[np.newaxis, :, :])

        # created expected zero-padded mask
        mask_0_expected = torch.tensor([[[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]],

                                        [[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 1., 1., 0.],
                                         [0., 0., 1., 1., 0.],
                                         [0., 0., 0., 0., 0.]],

                                        [[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]]], dtype=torch.float64)

        # print("mask before  ", masks_batch[0])
        # print("mask before ", masks_batch[0].shape)
        mask_0_pad = zero_pad_masks(images_batch[0], masks_batch[0])
        # print(" mask pad  ", mask_0_pad)

        torch.testing.assert_allclose(mask_0_pad, mask_0_expected)


    def test_unpad_transformed_masks_odd_diff(self):
        """
        test unpadding zero-padding of a mask for odd difference in number of pixels
        :return:
        """
        # create image batch
        images_batch = np.array([[[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]],

                                 [[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]],

                                 [[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]]])
        # make it batch of size 1
        images_batch = torch.tensor(images_batch[np.newaxis, :, :, :])

        # create mask batch
        masks_pad_batch = torch.tensor([[[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]],

                                        [[0., 0., 0., 0., 0.],
                                         [0., 0., 1., 1., 0.],
                                         [0., 0., 1., 1., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]],

                                        [[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]]], dtype=torch.float64)
        # make it batch of size 1
        masks_pad_batch = torch.tensor(masks_pad_batch[np.newaxis, :, :])

        # created expected unpadded mask
        mask_0_expected = np.array([[[0., 0., 0., 0.],
                                 [0., 0., 1., 1.],
                                 [0., 0., 1., 1.],
                                 [0., 0., 0., 0.]]])

        mask_0 = unpad_transformed_masks(images_batch[0], mask_0_expected, masks_pad_batch[0])
        print("mask 0 ", mask_0)

        torch.testing.assert_allclose(mask_0, mask_0_expected)


    def test_unpad_transformed_masks_even_diff(self):
        """
        test unpadding zero-padding of a mask for even difference in number of pixels
        :return:
        """
        # create image batch
        images_batch = np.array([[[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]],

                                 [[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]],

                                 [[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]]])
        # make it batch of size 1
        images_batch = torch.tensor(images_batch[np.newaxis, :, :, :])

        # create mask batch
        masks_pad_batch = torch.tensor([[[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]],

                                        [[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 1., 1., 0.],
                                         [0., 0., 1., 1., 0.],
                                         [0., 0., 0., 0., 0.]],

                                        [[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]]], dtype=torch.float64)
        # make it batch of size 1
        masks_pad_batch = torch.tensor(masks_pad_batch[np.newaxis, :, :])

        # created expected unpadded mask
        mask_0_expected = np.array([[[0., 0., 0.],
                                     [0., 1., 1.],
                                     [0., 1., 1.]]])

        mask_0 = unpad_transformed_masks(images_batch[0], mask_0_expected, masks_pad_batch[0])

        torch.testing.assert_allclose(mask_0, mask_0_expected)
