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


class SplitDataLoader(object):
    """
    A class to split datasets into tran, val and test
    """
    def __init__(self, dataset,
                 batch_size=2,
                 tr=None, vl=None, ts=None):
        """
        Initializes SplitDataLoader class
        ---
        Parameters
        ---
        dataset:
            Output of UNetDataset() class
        batch_size: int
            Batch size
            Default is 2
        tr: int
            Number of samples for training
            Default is None
        vl: int
            Number of samples for validation
            Default is None
        ts: int
            Number of sample for testing
            Default is None
        tr + vl+ ts MUST add up to the total number of samples in the folder!  - may need to be changed to a fraction
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.tr = tr
        self.vl = vl
        self.ts = ts
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        # split into train-val-test
        if self.train_loader is None or self.val_loader is None or self.test_loader is None:
            self.split_and_load()

    def split_and_load(self):
        """
        Splits incoming dataset into train-val-test
        ---
        Returns
        ---
         train, val. and test. dataloaders
        """

        # generate len(self.images) random indices as if it were np.arange(len(self.images))
        # False stands for number generation without replacement (no repetitions)
        idx = np.random.choice(len(list(self.dataset)), len(list(self.dataset)), False)
        #
        # split generated indices to train-val-test sets
        tr_idx, vl_idx, ts_idx = np.split(idx, [self.tr, self.tr + self.vl])
        assert (len(tr_idx), len(vl_idx), len(ts_idx)) == (self.tr, self.vl, self.ts)

        #print("tr_idx", tr_idx)
        #print("vl_idx", vl_idx)
        #print("ts_idx", ts_idx)
        #
        # set random samplers to get random indices for each batch in dataloaders
        train_sampler = SubsetRandomSampler(tr_idx)
        val_sampler = SubsetRandomSampler(vl_idx)
        test_sampler = SubsetRandomSampler(ts_idx)
        #
        self.train_loader = DataLoader(
            self.dataset, self.batch_size, sampler=train_sampler)
        self.val_loader = DataLoader(
            self.dataset, self.batch_size, sampler=val_sampler)
        self.test_loader = DataLoader(
            self.dataset, self.batch_size, sampler=test_sampler)

        return self.train_loader, self.val_loader, self.test_loader
