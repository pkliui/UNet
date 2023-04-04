#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes to process data prior to training, validation and testing

"""
import cv2
import numpy as np

# load data using dataloader
from torch.utils.data import DataLoader, SubsetRandomSampler


class Resize(object):
    """
    Resize the image in a sample to a given size
    Normalize to 1
    Convert to float32
    """

    def __init__(self, new_image_size, new_mask_size):
        """
        Initializes Resize class to resize images for UNet image segmentation
        ---
        Parameters
        ---
        new_image_size: tuple of int
            Desired output image's size
        new_mask_size: tuple of int
            Desired output mask's size
        """
        assert isinstance(new_image_size, tuple)
        assert isinstance(new_mask_size, tuple)
        self.new_image_size = new_image_size
        self.new_mask_size = new_mask_size

    def __call__(self, originals_dictionary):
        """
        Applies the resizer function to an input dictionary of images and masks
        ---
        Parameters
        ---
        originals_dictionary: dict
            Dictionary with keys "image" and "mask" containing images and their respective masks
        ---
        Return
        ---
        A dictionary of resized and normalized images and masks
        """
        # a dictionary to keep resized data
        resized_dict = {}
        # go through the content of the input dictionary and resize
        for ii in originals_dictionary:
            resized_dict[ii] = self.resizer(originals_dictionary[ii], (ii == "image") * self.new_image_size + (ii == "mask") * self.new_mask_size)
        #
        return resized_dict

    def resizer(self, original, new_size):
        """
        resizes  an input image by nearest neighbor interpolation
        normalizes  it to 0...1 range
        ---
        Parameters
        ---
        original: np.array
            an input image to be resized and normalized
        new_size: tuple of int
            a new size to resize the input to
        ---
        Return
        ---
        resized: np.array
            resized input image normalized to 0...1 range and converted to float32
        """

        norm_factor = np.max(original) * (np.max(original) != 0) + 1 * (np.max(original) == 0)
        resized = cv2.resize(original, new_size, interpolation=cv2.INTER_NEAREST) / norm_factor
        return np.array(resized, np.float32)
