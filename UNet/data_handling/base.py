from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

"""
This module contains the definition of two base classes
derived from PyTorch Dataset and DataLoader classes.

BaseDataset is here a pure abstract class, all its members
need to defined according to the specific shape of the data.

BaseDataLoader is derived from DataLoader, it can be instantiated as
it is. It handles validation/training val_split_size of the input dataset
in which case it has two attributes: train_loader and val_loader.
"""


class BaseDataset(Dataset):
    def __len__(self):
        """
        Return the length of the dataset.
        """
        return NotImplementedError("__len__ not implemented")

    def __getitem__(self, index):
        """
        Override this function to define the specific data
        loading procedure for one specific item.
        A sample should be define as a dictionary with an
        "Id" key, an "Input" key and optionally a ground label
        identified by a "Label" key.
        """
        return NotImplementedError("GetItem not implemented")

    def transform_input(self):
        """
        Override this function to define any pre-processing you want to apply
        to the inputs in the dataloading step.
        """
        return NotImplementedError("Transform not implemented")

    def transform_label(self):
        """
        Override this function to define any pre-processing you want to apply
        to the labels in the dataloading step.
        """
        return NotImplementedError("Transform not implemented")

class BaseDataLoader:
    def __init__(self, dataset=None,
                 batch_size=None,
                 validation_split=None,
                 shuffle_for_split=True,
                 random_seed_split=42):
        """
        Initializes the dataloader.
        3 configuration are supported:
            *   dataset used for training only (validation_split = 0).
                Only self.train_loader is initialized.
            *   dataset used for validation/testing only (validation_split = 1)
                Only self.val_loader is initialized
            *   dataset to be val_split_sizeted between validation and training
                Both self.train_loader and self.val_loader are initialized
        :param dataset: Dataset to use for training/validation
        :param batch_size: Batch size to use for training/validation
        :param validation_split: Fraction of the dataset to use for validation
        :param shuffle_for_split: Whether to shuffle the dataset before val_split_sizeting
        :param random_seed_split: Random seed to use for shuffling the dataset before val_split_sizeting it.
               Needed to ensure reproducibility of the val_split_size.
        
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.validation_split = validation_split

        # case no validation set
        if self.validation_split == 0:
            print("validation val_split_size = ", self.validation_split)
            # train loader
            self.train_loader = DataLoader(
                self.dataset, self.batch_size, shuffle=True)

        # case validation set only (that is only test set)
        elif self.validation_split == 1:
            print("validation val_split_size = ", self.validation_split)
            # set the test set to validation set
            self.val_loader = DataLoader(
                self.dataset, self.batch_size, shuffle=True)

        # case training/validation set val_split_size
        else:
            print("validation val_split_size = ", self.validation_split)
            indices = np.arange(len(dataset))
            # generate random indicies for val_split_size
            if shuffle_for_split is True:
                np.random.seed(random_seed_split)
                indices = np.random.permutation(indices)
                print(indices)
            #
            # setup random samplers for data loaders
            val_split_size = int(np.floor(validation_split * len(dataset)))

            print("train_split_size ", int(len(dataset) - val_split_size))
            print("val_split_size", val_split_size)
            #
            # check that the val_split_size is not too big or too small
            # specifically, check that the remaining smaller portion for train is not smaller than the batch size
            if validation_split >= 0.5 and (len(dataset) - val_split_size >= self.batch_size):
                print("the remaining train split size is just right ")
            # specifically, check that the val_split_size is not smaller than the batch size
            elif validation_split < 0.5 and (val_split_size >= self.batch_size):
                print("the val_split_size is just right  ")
            else:
                raise ValueError("Decrease the batch size or change the validation val_split_size")

            train_sampler = SubsetRandomSampler(indices[val_split_size:])
            val_sampler = SubsetRandomSampler(indices[:val_split_size])
            #
            # load date with data loaders
            print("preparing train and val loaders ... ")
            self.train_loader = DataLoader(
                self.dataset, self.batch_size, sampler=train_sampler)
            self.val_loader = DataLoader(
                self.dataset, self.batch_size, sampler=val_sampler)
