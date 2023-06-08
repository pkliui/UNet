#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from skimage.io import imread
from pathlib import Path
import torch.nn as nn
import numpy as np

from UNet.classes.base import BaseDataset
from typing import Optional, Callable, List, Tuple

from UNet.utils.resize_data import ResizeData

"""
This class contains a class to read data for UNet-based segmentation
"""


class UNetDataset(BaseDataset):

    def __init__(self,
                 required_image_size: Tuple[int, int],
                 required_mask_size: Tuple[int, int],
                 images_list: List[str],
                 masks_list: List[str],
                 resize_required: bool):
        """
        Initializes class to read images for UNet image segmentation into a dataset. Expects the full paths of images
        and masks to be provided at initialization.

        The file structure follows:
        * root_dir
            * sample1
                * sample1_images_tag
                    * sample1_optional_images_subtag.bmp
                * sample1_masks_tag
                    * sample1_optional_masks_subtag.bmp
            * sample2
                * sample2_images_tag
                    * sample1_optional_images_subtag.bmp
                * sample2_masks_tag
                    * sample2_optional_masks_subtag.bmp

        :param required_image_size: Image size as required by model
        :param required_mask_size: Mask size as required by model
        :param images_list: List of full paths to images
        :param masks_list: List of full paths to masks
        :param resize_required: If True, input images and masks will be resized

        :return sample: A dictionary with keys 'image' and 'mask' containing an image and a mask, respectively
        """

        self.required_image_size = required_image_size
        self.required_mask_size = required_mask_size
        self.images_list = images_list
        self.masks_list = masks_list
        self.resize_required = resize_required

    def __getitem__(self, item):
        """
        Reads images and masks one-by-one by their index in the corresponding lists of paths
        and returns them as a dictionary.

        :param item: Since the paths to images and masks are saved in lists, the item parameter is an integer
        representing the index of the element to be accessed.

        :return sample: A dictionary with keys 'image' and 'mask' containing an image and a mask, respectively
        """
        #
        # read images and masks
        image = imread(self.images_list[item])
        mask = imread(self.masks_list[item])

        # perhaps move to validation whilst resizing
        if len(mask.shape) == 2:
            mask = mask[:,:, np.newaxis]  # add dimension 1 to  mask images

        print("unetdataset image shape ", image.shape)
        print("unetdataset mask shape ", mask.shape)
        #
        #try:
        # if grand-parent folder names are the same
        self.images_parent_folder = os.path.normpath(Path(self.images_list[item]).parents[1]).encode('utf-8')
        self.masks_parent_folder = os.path.normpath(Path(self.masks_list[item]).parents[1]).encode('utf-8')
        #if os.path.normpath(self.images_parent_folder) == os.path.normpath(self.masks_parent_folder):

        print(self.masks_list[item])

        if os.path.abspath(self.images_parent_folder) == os.path.abspath(self.masks_parent_folder):
            print("paths are euql ")
            #
            # save current image and mask into a dictionary
            sample = {'image': image, 'mask': mask}
            #
            # resize
            if self.resize_required is True:
                print(sample)
                print(type(sample["image"]))
                sample = self.resize_sample(sample)

                print(sample)
                print(type(sample["image"]))

                print("unetdataset image shape after transform - resize ", sample["image"].shape)
                print("unetdataset mask shape after transform - resize ", sample["mask"].shape)
            return sample
        else:
            print("paths are NOT  euql ")
            ValueError(f"Parent folder of images {self.images_parent_folder} and"
              f"parent filder of masks {self.masks_parent_folder} are not the same! Skipping this dataset")
        #except ValueError as e:
        #    print(f"Exception ! Parent folder of images {self.images_parent_folder} and"
        #          f"parent filder of masks {self.masks_parent_folder} are not the same! Skipping this dataset")

    def resize_sample(self, sample: dict) -> dict:
        """
        Resize input sample of data
        :param sample: Dictionary with keys 'image' and 'mask' containing an image and a mask, respectively
        :return: Dictionary of resized image and mask
        """
        resizer = nn.Sequential(ResizeData(self.required_image_size, self.required_mask_size))
        return resizer(sample)

    def __len__(self):
        return len(self.images_list)
