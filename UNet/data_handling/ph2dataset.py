import os
from skimage.io import imread
import numpy as np

from UNet.classes.base import BaseDataset


"""
This class contains a class for the PH2 Dataset.
"""

class PH2Dataset(BaseDataset):
    """
    Class to read PH2 data for UNet segmentation

    """

    def __init__(self, root_dir=None, images_folder=None, masks_folder=None, transform=None, files_structure = 1):
        """
        Initializes class to read images for UNet image segmentation
        ---
        Parameters
        ---
        root_dir: str, optional
            Directory with images_folder and masks_fodler
            If None, an empty class is created
            Default is None
        images_folder: str
            Folder with images
        masks_folder: str
            Folder with masks
        transform : callable, optional
            Optional transform to be applied on a sample
        files_structure : int = {1, 2}
            Files structure
            1:
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
            2:
            * root dir
                * folder_1
                    * image
                        * image_1.jpg
                    * mask
                        * mask_1.jpg
                    * may be some other folder or folders
                * folder_2
                    * image
                        * image_2.jpg
                    * mask
                        * mask_2.jpg
                    * may be some other folder or folders
                ...


            Default 1
        """
        self.root_dir = root_dir
        self.images_folder = images_folder
        self.masks_folder = masks_folder

        self.transform = transform
        self.files_structure = files_structure

    def __getitem__(self, idx):
        """
        Supports indexing such that dataset[i] can be used to get the i-th sample
        Reads images and masks one by one
        """
        counter = 0
        sample = self.read_data(self.root_dir, self.images_folder, self.masks_folder, idx, counter)
        counter += 1
        #
        # transform if specified
        if self.transform:
            sample = self.transform(sample)

        print("idx ", idx)
        return sample

    def read_data(self, root_dir, images_folder, masks_folder, idx, counter):
        """
        reads raw images from a directory
        ---
        Parameters
        ---
        root_dir: str, optional
            Path used to load raw images
            If None, an empty class is created
            Default is None
        images_folder: str
            Name of the folder under datapath containing images
        masks_folder: str
            Name of the folder under datapath containing masks
        idx: int
            Integer index to be able to use in __getitem__
        """

        #
        # read data for files' structure 1
        if self.files_structure == 1:
            # walk through the root and get images and their masks
            for root, dirs, files in os.walk(root_dir):
                #print("root ", root)
                #
                # exclude files and directories starting with .
                files = [f for f in files if not f[0] == '.']
                dirs[:] = [d for d in dirs if not d[0] == '.']
                #print("dirs ", dirs)
                #print("files ", files)
                #
                if root.endswith(images_folder):
                    image = imread(os.path.join(root, files[idx]))
                    #print("read image ", os.path.join(root, files[idx]))
                if root.endswith(masks_folder):
                    mask = imread(os.path.join(root, files[idx]))
        #
        # read data for files' structure 2
        if self.files_structure == 2:
            #
            images_path = []
            images_list = []
            masks_path = []
            masks_list = []
            # walk through the root and get images and their masks
            for root, dirs, files in os.walk(root_dir):
                #print("root ", root)
                #
                # exclude files and directories starting with .
                files = [f for f in files if not f[0] == '.']
                dirs[:] = [d for d in dirs if not d[0] == '.']
                #print("dirs ", dirs)
                #print("files ", files)
                #
                # append paths to folders with images and images' names
                if root.endswith(images_folder):
                    images_path.append(root)
                    images_list.append(files[0])
                    #print("images_path ", images_path)
                # append paths to folders with masks and masks' names
                if root.endswith(masks_folder):
                    masks_path.append(root)
                    masks_list.append(files[0])
            #
            # now read idx'th image and idx'th mask
            image = imread(os.path.join(images_path[idx], images_list[idx]))
            #print("read image ", os.path.join(images_path[idx], images_list[idx]))
            mask = imread(os.path.join(masks_path[idx], masks_list[idx]))
        # save to a dictionary
        sample = {'image': image, 'mask': mask}
        return sample





    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        image_name = self.images_list[item]
        image = np.asarray(
            Image.open(os.path.join(self.images_path, image_name)))
        image = self.transform_input(image)
        id = int(image_name[:-4])
        if self.has_labels:
            label = self.labels.loc[id, 'label']
            label = self.transform_label(label)
            return {'Input': image, 'Label': label, 'Id': id}
        else:
            return {'Input': image, 'Id': id}

    def transform_input(self, image):
        image = image / 255.0
        image = \
            image - np.mean(image, axis=(0, 1)) / np.std(image, axis=(0, 1))
        return np.transpose(image, [2, 1, 0])

    def transform_label(self, label):
        return self.label_encoder.transform([label])[0]

    def inverse_transform_label(self, one_hot_labels):
        return self.label_encoder.inverse_transform(one_hot_labels)