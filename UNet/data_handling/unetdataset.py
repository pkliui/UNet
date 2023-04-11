#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, fnmatch
from skimage.io import imread
import numpy as np

from UNet.classes.base import BaseDataset
from UNet.data_handling import utils

"""
This class contains a class to read UNet data
"""


class UNetDataset(BaseDataset):

    def __init__(self, root_dir=None, images_folder=None, masks_folder=None, extension=None, transform=None):
        """
        Initializes class to read images for UNet image segmentation
        Assumes the following structure of files:
            * root dir
                * images
                    * image_1.jpg (or any other format)
                    * image 2.jpg
                    ...
                * masks
                    * mask_1.jpg
                    * mask_2.jpg
                    ...
                * may be some other directory or directories
        ---
        Args:
        ---
        root_dir: str
            Directory with images_folder and masks_fodler
        images_folder: str
            Folder with images
        masks_folder: str
            Folder with masks
        extension: str
            Extension of images and masks
        transform : callable, optional
            Optional transform to be applied on a sample
            Default: None
        """
        self.extension = extension
        self.root_dir = root_dir
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.transform = transform

        if os.path.exists(os.path.join(self.root_dir, self.images_folder)):
            self.images_list = np.asarray(utils.list_files_in_dir(os.path.join(self.root_dir, self.images_folder), self.extension))
        else:
            raise ValueError("Invalid images_folder! It must be a valid directory!")
        if os.path.exists(os.path.join(self.root_dir, self.masks_folder)):
            self.masks_list = np.asarray(utils.list_files_in_dir(os.path.join(self.root_dir, self.masks_folder), self.extension))
        else:
            raise ValueError("Invalid masks_folder! It must be a valid directory!")

    def __getitem__(self, item):
        """
        Supports indexing such that dataset[i] can be used to get the i-th sample
        Reads images and masks one by one
        ---
        Args:
        ---
        item: int
            Index of the sample
        ---
        Returns:
        ---
        sample: dict
            Dictionary with image and mask
        """
        # supported file types
        filetypes = ["*.jpeg", "*.jpg", "*.bmp", "*.png"]
        #
        # read images and masks
        if self.extension in (tuple(filetypes)):
            image_name = self.images_list[item]
            image = imread(os.path.join(self.root_dir, self.images_folder, image_name))
            mask_name = self.masks_list[item]
            mask = imread(os.path.join(self.root_dir, self.masks_folder, mask_name))
            #mask = mask > 0.5
            sample = {'image': image, 'mask': mask}
            #
            # transform if needed
            if self.transform:
                sample = self.transform(sample)
        else:
            raise ValueError("Invalid extension! It must be one of the following: {}".format(filetypes))
        return sample

    def __len__(self):
        assert len(self.images_list) == len(self.masks_list)
        return len(self.images_list)
