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
        "image" key, a "mask" key.
        """
        return NotImplementedError("GetItem not implemented")

    def transform_data(self):
        """
        Override this function to define any pre-processing you want to apply
        to the inputs in the dataloading step.
        """
        return NotImplementedError("Transform not implemented")


class BaseDataLoader:
    def __init__(self, dataset=None,
                 batch_size=None,
                 validation_split=0,
                 test_split=0,
                 shuffle_for_split=True,
                 random_seed_split=42):
        """
        Initializes the dataloader to handle training, validation and testing data

        3 configuration are supported:
            *   dataset used for training only (validation_split = 0).
                Only self.train_loader is initialized.
            *   dataset used for validation/testing only (validation_split = 1)
                Only self.val_loader is initialized
            *   dataset to be split between validation and training
                Both self.train_loader and self.val_loader are initialized
        :param dataset: Dataset to use for training/validation
        :param batch_size: Batch size to use for training/validation
        :param validation_split: Fraction of the dataset to use for validation
        :param test_split: Fraction of the dataset to use for testing
        :param shuffle_for_split: Whether to shuffle the dataset before val_split_sizeting
        :param random_seed_split: Random seed to use for shuffling the dataset before val_split_sizeting it.
               Needed to ensure reproducibility of the val_split_size.
        
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

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
            # generate random indicies for val_split_size and test_split_size
            if shuffle_for_split is True:
                # set random seed
                np.random.seed(random_seed_split)
                indices = np.random.permutation(indices)
            #
            # calculate sizes of train, validation, and test splits
            val_split_size = int(np.floor(validation_split * len(dataset)))
            test_split_size = int(np.floor(test_split * len(dataset)))
            train_split_size = len(dataset) - val_split_size - test_split_size
            print("val_split_size ", val_split_size)
            print("test_split_size ", test_split_size)
            print("train_split_size ", train_split_size)
            #
            # check that the val_split_size is not too big or too small
            # specifically, check that the remaining smaller portion for train is not smaller than the batch size
            if validation_split >= 0.5 and (len(dataset) - val_split_size - test_split_size >= self.batch_size):
                print("the remaining train split size is just right ")
            # specifically, check that the val_split_size is not smaller than the batch size
            elif validation_split < 0.5 and (val_split_size >= self.batch_size):
                print("the val_split_size is just right  ")
            else:
                raise ValueError("Decrease the batch size or change the validation val_split_size")
            #
            # check that the test_split_size is not too big or too small
            # specifically, check that the remaining smaller portion for train is not smaller than the batch size
            if test_split >= 0.5 and (len(dataset) - val_split_size - test_split_size >= self.batch_size):
                print("the remaining train split size is just right ")
            # specifically, check that the test_split_size is not smaller than the batch size
            elif (test_split > 0) and (test_split < 0.5) and (test_split_size >= self.batch_size):
                print("the test_split_size is just right  ")
            elif test_split == 0:
                pass
            else:
                raise ValueError("Decrease the batch size or change the test test_split_size")

            # split the indices into train, validation, and test sets
            train_indices = indices[:train_split_size]
            val_indices = indices[train_split_size:train_split_size+val_split_size]
            test_indices = indices[train_split_size+val_split_size:]

            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            #
            # load data with data loaders
            self.train_loader = DataLoader(self.dataset, self.batch_size, sampler=train_sampler)
            self.val_loader = DataLoader(self.dataset, self.batch_size, sampler=val_sampler)
            self.test_loader = DataLoader(self.dataset, self.batch_size, sampler=test_sampler)