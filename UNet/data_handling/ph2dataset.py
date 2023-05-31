#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Optional, Callable

from UNet.data_handling.unetdataset import UNetDataset
from UNet.data_handling import utils


class PH2Dataset(UNetDataset):
    """
    Class to make a new PH2 dataset from the PH2 database images
    Copyright: Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira.
    PH² - A dermoscopic image database for research and benchmarking,
    35th International Conference of the IEEE Engineering in Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.

    Uses UNetDataset as base class.

    The file structure is as following:
    * root_dir
        * sample1
            * sample1_Dermoscopic_Image
                * sample1.bmp
            * sample1_lesion
            *    sample1_lesion.bmp
        * sample2
            * sample2_Dermoscopic_Image
                * sample1.bmp
            * sample2_lesion
                * sample2_lesion.bmp
    """

    def __init__(self,
                 root_dir: str,
                 transform: Optional[Callable]):
        """
        Initialize PH2Dataset class

        :param root_dir: root directory that contains folders with samples of data uniquely identifiable by their ID
        :param transform: optional, callable to transform the datasets prior to their use

        :return sample: A dictionary with keys 'image' and 'mask' containing an image and a mask, respectively
        """
        # initialize to keep all paths to images and masks
        images_list = []
        masks_list = []
        #
        # loop through individual folders with data
        for folder in os.listdir(root_dir):
            if not os.path.isdir(os.path.join(root_dir, folder)):
                continue
            #
            # set names for expected image and mask folders
            images_folder = os.path.join(root_dir, folder, f"{folder}_Dermoscopic_Image")
            masks_folder = os.path.join(root_dir, folder, f"{folder}_lesion")
            images_list = utils.get_list_of_data(images_folder, 'bmp', images_list)
            masks_list = utils.get_list_of_data(masks_folder, 'bmp',  masks_list)

        super().__init__(transform, images_list, masks_list)

    def __getitem__(self, item):
        """
        Return a dictionary containing an image and a mask

        :param item: The index or key used to access a specific sample of data from the dataset.
        :return sample: A dictionary with keys 'image' and 'mask' containing an image and a mask, respectively

        """
        sample = super().__getitem__(item)

        return sample
