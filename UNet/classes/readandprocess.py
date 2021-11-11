# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Class to read and process data prior to training, validation and testing
#
# """
#
# import os
# from skimage.io import imread
#
# import numpy as np
# import cv2
#
# # load data using dataloader
# from torch.utils.data import DataLoader
#
# import threading
# from threading import Thread
#
#
# class ReadAndProcess(object):
#
#     def __init__(self, datapath=None, images_folder=None, masks_folder=None, images=None, masks=None, size_X = (572,572), size_Y=(388,388),
#                  train=None, val=None, test=None, train_share=None, val_share=None, test_share=None, batch_size=None):
#         """
#         Initializes ReadAndProcess class to read and process images for UNet image segmentation
#         ---
#         Parameters
#         ---
#         datapath: str, optional
#             Path used to load raw images
#             If None, an empty class is created
#             Default is None
#         images_folder: str
#             Name of the folder under datapath containing images
#         masks_folder: str
#             Name of the folder under datapath containing masks
#         images : list, optional
#             list to initialize ndarrays of input images
#             If None, an empty class is created
#             Default is None
#         masks: list, optional
#             list to initialize ndarrays of masks
#             If None, an empty class is created
#             Default is None
#         size_X : tuple of int, optional
#             a shape expected by UNet to resize images to
#             Default is (578, 578)
#         size_Y : tuple of int, optional
#             a shape expected by UNet to resize masks to
#             Default is (388, 388)
#         train: DataLoader for training data
#             Default is None
#         val: DataLoader for validation data
#             Default is None
#         test: DataLoader for test data
#             Default is None
#         train_share: int
#             Size of train set
#             Default is None
#         val_share: int
#             Size of validation set
#             Default is None
#         test_share: int
#             Size of test set
#             Default is None
#         batch_size: int
#             Batch size
#             Default is None
#         """
#         self.datapath = datapath
#         self.images_folder = images_folder
#         self.masks_folder = masks_folder
#         self.images = images
#         self.masks = masks
#         self.size_X = size_X
#         self.size_Y = size_Y
#         self.train = train
#         self.val = val
#         self.test = test
#         self.train_share = train_share
#         self.val_share = val_share
#         self.test_share = test_share
#         self.batch_size = batch_size
#         #
#         #
#         # if no images and masks provided
#         if images is None and masks is None:
#             if self.datapath is not None and self.images_folder is not None and self.masks_folder is not None:
#                 self.read_data(datapath, images_folder, masks_folder)
#         # if images and masks provided:
#         else:
#             self.images = images
#             self.masks = masks
#
#             #self.resize_data(size_X, size_Y)
#             #self.split_data(train_share, val_share, test_share, batch_size)
#             #
#         # split into train-val-test
#         #if self.train_share is not None and val_share is not None and test_share is not None and self.batch_size is not None:
#         #    self.split_data(train_share, val_share, test_share, batch_size)
#         #else:
#         #    raise ValueError("Training, validation and test set sizes and batch size cannot be None")
#
#     def read_data(self, datapath, images_folder, masks_folder):
#         """
#         reads raw images from a directory
#         ---
#         Parameters
#         ---
#         datapath: str, optional
#             Path used to load raw images
#             If None, an empty class is created
#             Default is None
#         images_folder: str
#             Name of the folder under datapath containing images
#         masks_folder: str
#             Name of the folder under datapath containing masks
#         """
#         images = []
#         masks = []
#         #
#         # read data
#         if os.path.exists(datapath) and os.path.exists(os.path.join(datapath,images_folder)) and os.path.exists(os.path.join(datapath,masks_folder)):
#             #
#             # walk through the root and get images and their masks
#             for root, dirs, files in os.walk(datapath):
#                 if root.endswith(images_folder):
#                     images.append(imread(os.path.join(datapath, files[0])))
#                 if root.endswith(masks_folder):
#                     masks.append(imread(os.path.join(datapath, files[0])))
#             self.images = images
#             self.masks = masks
#         else:
#             raise ValueError("Invalid path to images. Specify a valid path")
#
#
#     def read_data2(self, root_dir, images_folder, masks_folder, idx):
#         """
#         reads raw images from a directory
#         ---
#         Parameters
#         ---
#         root_dir: str, optional
#             Path used to load raw images
#             If None, an empty class is created
#             Default is None
#         images_folder: str
#             Name of the folder under datapath containing images
#         masks_folder: str
#             Name of the folder under datapath containing masks
#         """
#         #images = []
#         #masks = []
#         path2images = []
#         #
#         # read data
#         if os.path.exists(root_dir) and os.path.exists(os.path.join(root_dir,images_folder)) and os.path.exists(os.path.join(root_dir,masks_folder)):
#             #
#             # walk through the root and get images and their masks
#             for root, dirs, files in os.walk(root_dir):
#                 if root.endswith(images_folder):
#                     print(os.path.join(root, files[1]))
#                 #if root.endswith(masks_folder):
#                 #    path2imagesos.path.join(datapath, files[0]))
#             #self.images = images
#             #self.masks = masks
#         else:
#             raise ValueError("Invalid path to images. Specify a valid path")
#
#     def resize_data(self, size_X, size_Y):
#         """
#         resize images as required by UNet architecture
#         and normalize to (0,1)
#         use nearest neighboour interpolation as any other method may result in tampering with the ground truth labels
#         ---
#         Input
#         ---
#         size_X: tuple of int
#          size of images feed into the network
#          Default (572, 572)
#         size_Y: tuple if int
#          size of masks output by the network
#          Default (388, 388)
#         """
#         if size_X is not None and size_Y is not None:
#             # resize to shape expected by UNet, normalize to 0..1 range and convert to float32
#             # no normalization if all pixel values are 0 == max pixel value is 0 to avois division by 0
#             x_resized = [cv2.resize(x, size_X, interpolation=cv2.INTER_NEAREST)/np.max(x) if np.max(x)!=0 else cv2.resize(x, size_X, interpolation=cv2.INTER_NEAREST) for x in self.images]
#             y_resized = [cv2.resize(y, size_Y, interpolation=cv2.INTER_NEAREST)/np.max(y) if np.max(y)!=0 else cv2.resize(y, size_Y, interpolation=cv2.INTER_NEAREST) for y in self.masks]
#             self.images = np.array(x_resized, np.float32)
#             self.masks = np.array(y_resized, np.float32)
#         else:
#             raise ValueError("images' and masks' sizes cannot be None")
#
#     def split_data(self, train_share, val_share, test_share, batch_size):
#         """
#         Splits input images into training, validation and testing sets
#
#         :return:
#         """
#         # generate len(self.images) random indices as if it were np.arange(len(self.images))
#         # False is to generate without replacement (no repetitions)
#         idx = np.random.choice(len(self.images), len(self.images), False)
#         #
#         # split generated indices to train-val-test sets as following: 100 test-50 val-50 test
#         # [100, 150] entries indicate where along axis the ix array is split.
#         tr, val, ts = np.split(idx, [train_share, train_share + val_share])
#         assert (len(tr), len(val), len(ts)) == (train_share, val_share, test_share)
#         #
#         # set the dataloaders
#         # set drop_last to skip the batches with the # elements < batch size
#         if batch_size > 0 and isinstance(batch_size, int) is True:
#             self.train = DataLoader(list(zip(np.rollaxis(self.images[tr], 3, 1), self.masks[tr, np.newaxis])),
#                                  batch_size=batch_size, shuffle=True, drop_last=True)
#             self.val = DataLoader(list(zip(np.rollaxis(self.images[val], 3, 1), self.masks[val, np.newaxis])),
#                                   batch_size=batch_size, shuffle=True, drop_last=True)
#             self.test = DataLoader(list(zip(np.rollaxis(self.images[ts], 3, 1), self.masks[ts, np.newaxis])),
#                                  batch_size=batch_size, shuffle=True, drop_last=True)
#         else:
#             raise ValueError("batch_size should be a positive integer value, but got batch_size ", batch_size)