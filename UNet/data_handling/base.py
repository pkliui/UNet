from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

"""
This module contains the definition of two base classes
derived from PyTorch Dataset and DataLoader classes.

BaseDataset is here a pure abstract class, all its members
need to defined according to the specific shape of the data.

BaseDataLoader is derived from DataLoader, it can be instantiated as
it is. It handles validation/training split of the input dataset
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
                 batch_size=1,
                 validation_split=0,
                 shuffle_for_split=True,
                 random_seed_split=0):
        """
        Initializes the dataloader.
        3 configuration are supported:
            *   dataset used for training only (validation_split = 0).
                Only self.train_loader is initialized.
            *   dataset used for validation/testing only (validation_split = 1)
                Only self.val_loader is initialized
            *   dataset to be splitted between validation and training
                Both self.train_loader and self.val_loader are initialized
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.validation_split = validation_split

        # case no validation set
        if self.validation_split == 0:
            print("validation split = 0", self.validation_split)
            self.train_loader = DataLoader(
                self.dataset, self.batch_size, shuffle=True)

        # case validation set only (test set)
        elif self.validation_split == 1:
            print("validation split = 1", self.validation_split)
            self.val_loader = DataLoader(
                self.dataset, self.batch_size, shuffle=True)

        # case training/validation set split
        else:
            print("validation split = ", self.validation_split)
            indices = np.arange(len(dataset))
            # generate random indicies for split
            if shuffle_for_split:
                np.random.seed(random_seed_split)
                indices = np.random.permutation(indices)
            #
            # setup random samplers for data loaders
            split = int(np.floor(validation_split * len(dataset)))

            print("int(len(dataset) - split)", int(len(dataset) - split))
            print("split", split)
            if validation_split >= 0.5 and (len(dataset) - split >= self.batch_size):
                pass
            elif validation_split < 0.5 and (split >= self.batch_size):
                pass
            else:
                raise ValueError("Decrease the batch size or change the validation split")

            train_sampler = SubsetRandomSampler(indices[split:])
            val_sampler = SubsetRandomSampler(indices[:split])
            #
            # load date with data loaders
            self.train_loader = DataLoader(
                self.dataset, self.batch_size, sampler=train_sampler)
            self.val_loader = DataLoader(
                self.dataset, self.batch_size, sampler=val_sampler)
