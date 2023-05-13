#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from skimage.io import imread
from pathlib import Path

from UNet.classes.base import BaseDataset
from typing import Optional, Callable, List

"""
This class contains a class to read data for UNet-based segmentation
"""


class UNetDataset(BaseDataset):

    def __init__(self,
                 transform: Optional[Callable],
                 images_list: List[str],
                 masks_list: List[str]):
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

        :param transform: Optional transform to be applied on a sample
        :param images_list: List of full paths to images
        :param masks_list: List of full paths to masks
        """

        self.transform = transform
        self.images_list = images_list
        self.masks_list = masks_list

    def __getitem__(self, item):
        """
        Reads images and masks one-by-one by their index in the corresponding lists of paths
        and returns them as a dictionary.

        :param item: Since the paths to images and masks are saved in lists, the item parameter is an integer
        representing the index of the element to be accessed.

        :return sample: Dictionary with keys 'image' and 'mask' and respective
        """
        #
        # read images and masks
        image = imread(self.images_list[item])
        mask = imread(self.masks_list[item])
        #
        try:
            # if grand-parent folder names are the same
            self.images_parent_folder = Path(self.images_list[item]).parents[2]
            self.masks_parent_folder = Path(self.masks_list[item]).parents[2]
            if self.images_parent_folder == self.masks_parent_folder:
                #
                # save current image and mask into a dictionary
                sample = {'image': image, 'mask': mask}
                #
                # transform if needed
                if self.transform:
                    sample = self.transform(sample)
                return sample
            else: ValueError()
        except ValueError as e:
            print(f"Parent folder of images {self.images_parent_folder} and"
                  f"parent filder of masks {self.masks_parent_folder} are not the same! Skipping this dataset")

    def __len__(self):
        return len(self.images_list)
