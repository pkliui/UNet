import torch
from torchvision import transforms
import numpy as np


class AugmentImageAndMask(torch.utils.data.Dataset):
    """
    class to augment images for image segmentation problem
    applies the same transformations to both the input image and its segmented counterpart (mask)
    ---
    input requirements
    ---
    all images are rectangular, i.e. have the same size in both dimensions
    all mask images have lower linear number of pixels (as expected by Unet)
    only even linear number of pixels are accepted
    ---
    return
    ---
    tuple of (augmented original image, its augmented mask)
    i.e. ((augmented image [0], augmented mask [0]),
          (augmented image [1], augmented mask [1]),...
          )

    """

    def __init__(self, tensors, transform=None):
        # check that all tensors have the same size as the first one
        assert (tensor.size(0) == tensors[0].size(0) for tensor in tensors)
        assert (tensor.size(1) == tensors[0].size(1) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        #
        # get images and masks one by one
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        #
        # assert both images are rectangular
        assert x.shape[-1] == x.shape[-2], \
            "Dimensions of input images are assumed to be equal. Shape of a current input image is ({},{})".format(
            x.shape[-2], x.shape[-1])
        assert y.shape[-1] == y.shape[-2], \
            "Dimensions of input masks are assumed to be equal. Shape of a current input mask is ({},{})".format(
            y.shape[-2], y.shape[-1])
        #
        # zero-pad mask images to the shape of input images:
        #
        # set padding's width and height
        # for the -2 and -1 dimensions (width and height of individual images)
        n_pad_12 = int(np.subtract(x.shape[-1], y.shape[-1]) // 2)
        # for -3 dimension (number of channels)
        # if the input images are RGB:
        if len(x) == 3 and x.shape[0] == 3:
            n_pad_3 = 1
        # if the input images are grayscale
        elif x.shape[0] == 1:
            n_pad_3 = 0
        else:
            raise ValueError(
                "Input images must be either RGB or gray. Expected input dimensions are [3,x,x] for RGB or [1,x,x] or [x,x] for grayscale")
        # zero-pad
        y_pad = torch.from_numpy(
            np.pad(y, ((n_pad_3, n_pad_3), (n_pad_12, n_pad_12), (n_pad_12, n_pad_12)), 'constant', constant_values=0))
        #
        # concatenate to apply the same transform to both image(x) and its mask(y)
        xy = torch.cat((x, y_pad), dim=0)
        #
        # transform
        if self.transform:
            xy = self.transform(xy)
            #
            # split back into image(x) and mask(y)
            x, y_pad_aug = torch.chunk(xy, chunks=2, dim=0)
            #
            # unpad trasformed mask back to its original size
            # if RGB input images
            if len(x) == 3 and x.shape[0] == 3:
                # if images and masks have different sizes
                if n_pad_12 != 0:
                    y = y_pad_aug[1, n_pad_12:-n_pad_12, n_pad_12:-n_pad_12]
                # if images and masks have the same sizes
                else:
                    y = y_pad_aug[1, :, :]
            # if grayscale input images
            elif x.shape[0] == 1:
                # if images and masks have different sizes
                if n_pad_12 != 0:
                    y = y_pad_aug[:, n_pad_12:-n_pad_12, n_pad_12:-n_pad_12]
                # if images and masks have the same sizes
                else:
                    y = y_pad_aug[:, :, :]
            # add dimension 1 to augmented and unpadded mask images
            y = y[np.newaxis, :, :]
        # return augmented images and masks
        return x, y



def random_transforms(prob=0.5):
    """
    Defines transforms for image augmentation
    ---
    Parameters
    ---
    prob: float between 0 and 1
        probability of applying random transformations (the same for all transformations)
        Default: 0.5
    ---
    Return
    ---
    transform:
        an instance of Compose class
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=prob),
        transforms.RandomVerticalFlip(p=prob),
    ])

    return transform


def get_augmented_tensors(*augmented_xy_pairs):
    """
    unzips (reorders) the ouput of AugmentImageAndMask class augmented_xy_pairs
    so that the augmented images and augmented masks are in two separate tensors
    ---
    input
    ---
    augmented pairs of (image,mask) tensors
    ---
    return
    ---
    tuple of augmented images, tuple of augmented masks
    each tuple element is a tensor
    the length of each tuple corresponds to the length of the 0th dimension
    of the images' and masks' tensors parsed into AugmentImageAndMask class
    """
    #
    # note:
    # the star in *augmented_xy_pairs is needed to unpack the list so that all elements of it can be passed as different parameters.
    augmented_xy = list(zip(*augmented_xy_pairs))
    return augmented_xy[0], augmented_xy[1]


def reshape_batches(X_batch, Y_batch):
    """
    checks the shape of the input batches
    ensures the shape is (batch size, 3, width, height) for images
    and (batch size, 1, width, height) for masks
    :return: X_batch of shape (batch size, 3, width, height), Y_batch of shape (batch size, 1, width, height)
    """
    X_batch = np.array(X_batch, np.float32)
    Y_batch = np.array(Y_batch, np.float32)

    if len(X_batch.shape) == 4 and X_batch.shape[-1] == 3:
        X_batch_reshaped = np.rollaxis(X_batch, 3, 1)
    elif len(X_batch.shape) == 4 and X_batch.shape[-3] == 3:
        X_batch_reshaped = X_batch
    else:
        raise ValueError("Image's dimensions must be (batch_size, 3, width, height) or (batch_size, width, height, 3)! Current image's shape is {}".format(X_batch.shape))

    if len(Y_batch.shape) == 4 and Y_batch.shape[-1] == 1:
        Y_batch_reshaped = np.rollaxis(Y_batch, 3, 1)
    elif len(Y_batch.shape) == 4 and Y_batch.shape[-3] == 1:
        Y_batch_reshaped = Y_batch
    elif len(Y_batch.shape) == 3:
        Y_batch_reshaped = Y_batch[:, np.newaxis]
    else:
        raise ValueError("Mask's dimensions must be (batch_size, width, height)! Current mask's shape is {}".format(Y_batch.shape))

    return torch.from_numpy(X_batch_reshaped), torch.from_numpy(Y_batch_reshaped)
