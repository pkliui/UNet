"""This module contains functions to manipulate images' sampling """

import torch
import numpy as np


def zero_pad_masks(image: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    Zero-pad masks to have the same shape as images to be able to concat and transform them simultaneously

    :param image: image of shape (3, n_image, height) or (1, n_image, n_image)
        or (n_image, n_image, 3) or (n_image, n_image, 1)
    :param mask: mask of shape (1, n_mask, n_mask) or (n_mask, n_mask, 1)
    Width and height for masks and images, n_mask, n_image, are not necessarily the same.
    Even and odd differences in the number of pixels in width and height are handled accordingly

    :return mask_pad: mask zero padded to the shape of image (3, n_image, height) or (1, n_image, n_image)
    """
    # set padding's width and height
    # for the -2 and -1 dimensions (width and height of individual images)
    n_pad_12 = int(np.subtract(image.shape[-1], mask.shape[-1]) // 2)

    # for -3 dimension (number of channels)
    # if the input images are RGB:
    if len(image) == 3 and image.shape[0] == 3:
        n_pad_channel = 1
    # if the input images are grayscale
    elif image.shape[0] == 1:
        n_pad_channel = 0
    else:
        raise ValueError("Input images must be either RGB or grayscale.")

    # Zero pad for odd number of pixels difference
    if (image.shape[-1] - mask.shape[-1]) % 2 != 0:
        mask_pad = torch.from_numpy(np.pad(mask,
                                           ((n_pad_channel, n_pad_channel),
                                            (n_pad_12, n_pad_12 + 1),
                                            (n_pad_12, n_pad_12 + 1)),
                                           'constant',
                                           constant_values=0))
    # Zero pad for even number of pixels differences
    else:
        mask_pad = torch.from_numpy(np.pad(mask,
                                           ((n_pad_channel, n_pad_channel),
                                            (n_pad_12, n_pad_12),
                                            (n_pad_12, n_pad_12)),
                                           'constant',
                                           constant_values=0))
    return mask_pad


def unpad_transformed_masks(image: torch.Tensor,
                            mask: torch.Tensor,
                            mask_transformed_pad: torch.Tensor) -> torch.Tensor:
    """
    Unpad padded mask back to its original size

    :param image: image of shape (3, n_image, n_image) or (1, n_image, n_image)
        or (n_image, n_image, 3) or (n_image, n_image, 1),
        needed to extract the shape of the original image
    :param mask: mask of shape (n_image, n_image) or (n_image, n_image, 1),
        needed to extract the shape of the original mask
    :param mask_transformed_pad: masks padded to the same shape as image and transformed

    :return: mask_transformed: mask of shape (n_image, n_image) or (n_image, n_image, 1) as the original
        and transformed
    """
    # get padding width from the originals
    n_pad_12 = int(np.subtract(image.shape[-1], mask.shape[-1]) // 2)
    #print("n pad 12 ", n_pad_12)

    # get the zero pad width for odd number of pixels difference
    if (image.shape[-1] - mask.shape[-1]) % 2 != 0:
        n_pad_12_right = n_pad_12 + 1
        n_pad_12_left = n_pad_12
        #print("odd diff ")
    # get the zero pad width for even number of pixels differences
    else:
        #print("even  diff ")
        n_pad_12_right = n_pad_12
        n_pad_12_left = n_pad_12

    # if RGB input images
    if len(image) == 3 and image.shape[0] == 3:
        # if images and masks have different sizes
        if n_pad_12_right != 0:
            mask_transformed = mask_transformed_pad[1, n_pad_12_left:-n_pad_12_right, n_pad_12_left:-n_pad_12_right]
        # if images and masks have the same sizes
        else:
            mask_transformed = mask_transformed_pad[1, :, :]
        p#rint("RGB ")
        #print("mask transformed ", mask_transformed)
    # if grayscale input images
    elif image.shape[0] == 1:
        # if images and masks have different sizes
        if n_pad_12_right != 0:
            mask_transformed = mask_transformed_pad[:, n_pad_12_left:-n_pad_12_right, n_pad_12_left:-n_pad_12_right]
        # if images and masks have the same sizes
        else:
            mask_transformed = mask_transformed_pad[:, :, :]
        #print("gray scale ")
        #print("mask transformed ", mask_transformed)

    # add dimension 1 to augmented and unpadded mask images
    mask_transformed = mask_transformed[np.newaxis, :, :]

    return mask_transformed