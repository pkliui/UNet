
from torch.utils.data.dataset import Dataset

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
        A sample should be defined as a dictionary with an
        "image" key, a "mask" key.
        """
        return NotImplementedError("GetItem not implemented")

    def read_data(self):
        """
        Override this function to define read_data
        """
        return NotImplementedError("read_data not implemented")
