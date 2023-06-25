#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch.nn as nn
from typing import Tuple


class ResizeData(nn.Module):
    """
    Class to resize images and masks and normalize pixel values to 1
    """

    def __init__(self,
                 new_image_size: Tuple[int, int],
                 new_mask_size: Tuple[int, int]):
        """
        :param new_image_size: New output image size specified as tuple of 2 integers (no channel size)
            Example: (572, 572)
        :param new_mask_size: New output mask size specified as tuple of 2 integers (no channel size)
            Example: (388, 388)
        """
        super().__init__()
        if len(new_image_size)!=2 or len(new_mask_size)!=2:
            raise ValueError(f"New output size must be specified as a tuple of 2 integers, "
                             f"current new_image_size {new_image_size}, current new_mask_size {new_mask_size}")
        else:
            self.new_image_size = new_image_size
            self.new_mask_size = new_mask_size


    def __call__(self, orig_sample: dict) -> dict:
        """
        Resizes and normalizes original images and masks

        :param orig_sample: Dictionary with keys "image" and "mask" containing images and their respective masks

        :return resized_sample: Dictionary with resized images and masks whose pixel values are normalized
        """
        #
        # validate new input size
        self._validate_input_size(orig_sample['image'], orig_sample['mask'])

        # a dictionary to keep resized data
        resized_sample = {'image': resizer(orig_sample['image'], self.new_image_size),
                          'mask': resizer(orig_sample['mask'], self.new_mask_size)}
        #
        print("resized dict len ", len(resized_sample))
        print("resized dict image shape ", resized_sample['image'].shape)
        print("resized dict mask shape ", resized_sample['mask'].shape)
        return resized_sample

    def _validate_input_size(self, original_image, original_mask):
        """
        Validate input size with respect to the set new size
        """

        if not (isinstance(self.new_image_size, tuple)
                and isinstance(self.new_image_size[0], int) and isinstance(self.new_image_size[1], int)):
            raise ValueError("Specified new image size must be a tuple of integers")

        if not (isinstance(self.new_mask_size, tuple)
                and isinstance(self.new_mask_size[0], int) and isinstance(self.new_mask_size[1], int)):
            raise ValueError("Specified new mask size must be a tuple of integers")

        if len(original_image.shape) == 3 and original_image.shape[-3] == 3:
            if (self.new_image_size[-1] > original_image.shape[-1]) or (self.new_image_size[-2] > original_image.shape[-2]):
                raise ValueError("Specified new image size exceeds the original image dimensions. "
                                 f"Shape of a current input image is ({original_image.shape})."
                                 f"New image size {self.new_image_size}.")
        elif len(original_image.shape) == 3 and original_image.shape[-1] == 3:
            if (self.new_image_size[-1] > original_image.shape[-2]) or (self.new_image_size[-2] > original_image.shape[-3]):
                raise ValueError("Specified new image size exceeds the original image dimensions. "
                                 f"Shape of a current input image is ({original_image.shape})."
                                 f"New image size {self.new_image_size}.")
        else:
            raise ValueError(f"Input image must of of shape (3, n_image, n_image) or (n_image, n_image, 3)"
                             f"Current shape of input image is ({original_image.shape})")

        if len(original_mask.shape) == 3 and original_mask.shape[-3] == 1:
            if (self.new_mask_size[-1] > original_mask.shape[-1]) or (self.new_mask_size[-2] > original_mask.shape[-2]):
                raise ValueError("Specified new mask size exceeds the original mask dimensions. "
                                 f"Shape of a current input mask is ({original_mask.shape})."
                                 f"New mask size {self.new_mask_size}.")
        elif len(original_mask.shape) == 3 and original_mask.shape[-1] == 1:
            if (self.new_mask_size[-1] > original_mask.shape[-2]) or (self.new_mask_size[-2] > original_mask.shape[-3]):
                raise ValueError("Specified new mask size exceeds the original mask dimensions. "
                                 f"Shape of a current input mask is ({original_mask.shape})."
                                 f"New mask size {self.new_mask_size}.")

def resizer(original_image: np.ndarray,
            new_size: Tuple[int, int]):
    """
    Resizes  an original image or mask by nearest neighbor interpolation and normalizes it to 0...1 range
    ---
    Parameters
    ---
    original_image: np.array
        an input image to be resized and normalized
    new_size: tuple of int
        a new size to resize the input to
    ---
    Return
    ---
    resized: np.array
        resized input image normalized to 0...1 range and converted to float32
    """

    if len(original_image.shape) == 3 and original_image.shape[-3] == 3:
        if (new_size[-1] > original_image.shape[-1]) or (new_size[-2] > original_image.shape[-2]):
            raise ValueError("Specified new size exceeds the original dimensions. "
                             f"Shape of a current input is ({original_image.shape})."
                             f"New size {self.new_size}.")

    if len(original_image.shape) == 3 and original_image.shape[-1] == 3:
        if (new_size[-1] > original_image.shape[-2]) or (new_size[-2] > original_image.shape[-3]):
            raise ValueError("Specified new size exceeds the original dimensions. "
                             f"Shape of a current input is ({original_image.shape})."
                             f"New size {new_size}.")

    print(" originaal type ", type(original_image))
    print("original shape", original_image.shape)
    norm_factor = np.max(original_image) * (np.max(original_image) != 0) + 1 * (np.max(original_image) == 0)
    resized = cv2.resize(original_image, new_size, interpolation=cv2.INTER_NEAREST) / norm_factor
    return np.array(resized, np.float32)