import torch
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
import numpy as np
from typing import Tuple, List

from UNet.utils.sampling import zero_pad_masks, unpad_transformed_masks, reshape_batches


class AugmentImageAndMask(torch.utils.data.Dataset):

    """Class to augment a batch of images and masks one-by-one as per provided transform and probability"""

    def __init__(self, images_batch=None, masks_batch=None, transforms_types=None, prob=None):
        """
        :param images_batch: images batch of shape (batch_size, 3, n_image, n_image) or (batch_size, 1, n_image, n_image)
            or (batch_size, n_image, n_image, 3) or (batch_size, n_image, n_image, 1)
        :param masks_batch: masks batch of shape (batch_size, 1, n_mask, n_mask) or (batch_size, n_mask, n_mask, 1)
        :param transforms_types: a list of transforms to apply
            Must be one of str listed in transform_types_allowed variable
        :param prob: Probability of applying random transformations (the same for all transformations)
            Must be a float between 0 and 1

        Parameters defined in _validate_augment_image_and_masks_inputs method:
        :ivar transforms_types_allowed: Types of transforms that are currently implemented
            Example: "RandomHorizontalFlip", "RandomVerticalFlip"
        :type transforms_types_allowed: List[str]

        All params must have values as per validation function
        """
        self.transforms_types_allowed: List[str] = []
        """transform types currently implemented in random_transforms method"""

        self._validate_augment_image_and_masks_inputs(images_batch, masks_batch, transforms_types, prob)
        self.images_batch, self.masks_batch = reshape_batches(images_batch, masks_batch)
        self.transforms_types = transforms_types
        self.prob = prob
        self.transform = self._random_transforms(transforms_types=self.transforms_types,
                                                 prob=self.prob)
        """Obtain transform from the specified type and probability"""

    def __getitem__(self, index):
        """
        Get a pair of transformed image and mask by its index in the batch
        :param index: image's or mask's index within the batch
        :return: a tuple of an image and mask after transformation
        """
        # get an image and a mask
        image = self.images_batch[index]
        mask = self.masks_batch[index]

        # zero-pad mask to the shape of input image
        mask_pad = zero_pad_masks(image, mask)

        # concatenate and apply the same transform
        image_mask_pad = torch.cat((image, mask_pad), dim=0)
        if self.transform:
            image_mask_pad = self.transform(image_mask_pad)

            # split back into image and mask and unpad
            image, mask_pad = torch.chunk(image_mask_pad, chunks=2, dim=0)
            mask = unpad_transformed_masks(image, mask, mask_pad)

        return image, mask

    def _validate_augment_image_and_masks_inputs(self,
                                                 images_batch: torch.Tensor,
                                                 masks_batch: torch.Tensor,
                                                 transforms_types: List[str],
                                                 prob: float):
        """
        Validate the shapes of images and masks to be used for augmentation and the transform types

        :param images_batch: images batch of shape (batch_size, 3, n_image, n_image) or (batch_size, 1, n_image, n_image)
            or (batch_size, n_image, n_image, 3) or (batch_size, n_image, n_image, 1)
        :param masks_batch: masks batch of shape (batch_size, 1, n_mask, n_mask) or (batch_size, n_mask, n_mask, 1)
        :param transforms_types: a list of transforms to apply
            Must be one of "RandomHorizontalFlip", "RandomVerticalFlip" as per validate_random_transforms
        :param prob: Probability of applying random transformations (the same for all transformations)
            Must be a float between 0 and 1
            Default: 0.5

        Images and masks are assumed to have equal number of pixels in width and height, n_image, n_mask,
        but n_image and n_mask are not necessarily the same

        """
        # transform types currently implemented
        self.transforms_types_allowed = ["RandomHorizontalFlip", "RandomVerticalFlip"]

        # check transform types are not empty
        if not transforms_types:
            raise ValueError('Transform_types cannot be empty.')

        print("transforms_types", transforms_types)
        # check transform types values
        for transform_type in transforms_types:
            print("transform_type", transform_type)
            if transform_type not in self.transforms_types_allowed:
                raise ValueError('Invalid transform type specified in transform_types.'
                                 f'Must be one of the following: {self.transforms_types_allowed}')

        # check prob values
        if not (isinstance(prob, float) or isinstance(prob, int)):
            raise ValueError("Probability of transform must be a float between 0 or 1 or respective integer")

        if not (0 <= prob <= 1):
            raise ValueError(f'Probability of transform must be a float between 0 and 1 inclusive')

        # check batch dimensions
        if not (len(images_batch.shape) == 4 and len(masks_batch.shape) == 4):
            raise ValueError("Images and masks tensors must be of length 4."
                             f"Shape of a current input images tensor is ({images_batch.shape})"
                             f"Shape of a current input masks tensor is ({masks_batch.shape})")

        if not (images_batch.shape[-3] == 3 or images_batch.shape[-3] == 1
                or images_batch.shape[-1] == 3 or images_batch.shape[-1] == 1):
            raise ValueError("Images tensor must be of shape (batch_size, 3, n_image, n_image) or"
                             f" (batch_size, 1, n_image, n_image) or (batch_size, n_image, n_image, 3)"
                             f" or (batch_size, n_image, n_image, 1). Shape of a current input tensor is ({images_batch.shape})")

        if not (masks_batch.shape[-3] == 1 or masks_batch.shape[-1] == 1):
            raise ValueError("Masks tensor must be of shape (batch_size, 1, n_mask, n_mask) or "
                             "(batch_size, n_mask, n_mask, 1). "
                             f"Shape of a current input tensor is ({masks_batch.shape})")

        if images_batch.shape[-3] == 3 or images_batch.shape[-3] == 1:
            if not (images_batch.shape[-1] == images_batch.shape[-2]):
                raise ValueError("Dimensions of input images are assumed to be equal. "
                                 f"Shape of a current input tensor is ({images_batch.shape})")
        elif images_batch.shape[-1] == 3 or images_batch.shape[-1] == 1:
            if not (images_batch.shape[-2] == images_batch.shape[-3]):
                raise ValueError("Dimensions of input images are assumed to be equal. "
                                 f"Shape of a current input tensor is ({images_batch.shape})")

        if masks_batch.shape[-3] == 1:
            if not (masks_batch.shape[-1] == masks_batch.shape[-2]):
                raise ValueError("Dimensions of input masks are assumed to be equal. "
                                 f"Shape of a current input tensor is ({masks_batch.shape})")
        elif masks_batch.shape[-1] == 1:
            if not (masks_batch.shape[-2] == masks_batch.shape[-3]):
                raise ValueError("Dimensions of input masks are assumed to be equal. "
                                 f"Shape of a current input tensor is ({masks_batch.shape})")

        if not (images_batch.shape[0] == masks_batch.shape[0]):
            raise ValueError("The number of batches of input images and masks are assumed to be equal. "
                             f"Shape of a current input tensor of images is ({images_batch.shape})"
                             f"Shape of a current input tensor of masks is ({masks_batch.shape})")

    def _random_transforms(self, transforms_types: List[str],
                           prob: float = 0.5) -> nn.Sequential:
        """
        Defines transforms for image augmentation by using Compose class

        :param transforms_types: a list of transforms to apply
            Must be one of transforms in transforms_types_allowed variable validated in
            _validate_augment_image_and_masks_inputs method
        :param prob: Probability of applying random transformations (the same for all transformations)
            Must be a float between 0 and 1
            Default: 0.5
        :return transform (nn.Sequential): an instance of the `Sequential` class that joins the transforms together.

        Currently implemented transforms (validated in _validate_augment_image_and_masks_inputs method):
        RandomHorizontalFlip: Horizontally flip the given image randomly with a given probability.
        RandomVerticalFlip: Vertically flip the given image randomly with a given probability.
        """
        transforms = []

        for transform_type in transforms_types:
            if transform_type == 'RandomHorizontalFlip':
                transforms.append(RandomHorizontalFlip(p=prob))
            elif transform_type == 'RandomVerticalFlip':
                transforms.append(RandomVerticalFlip(p=prob))
            else:
                raise ValueError('Invalid transform type specified in transform_types.'
                                 f'Must be one of the following: {self.transforms_types_allowed}')
        transform = nn.Sequential(*transforms)

        return transform


class AugmentBatch(AugmentImageAndMask):
    """
    Class to augment a batch of images and masks as per provided transform and probability
    """

    def __init__(self, images_batch=None, masks_batch=None, transforms_types=None, prob=None):
        """
        :param images_batch: images batch of shape (batch_size, 3, n_image, n_image) or (batch_size, 1, n_image, n_image)
            or (batch_size, n_image, n_image, 3) or (batch_size, n_image, n_image, 1)
        :param masks_batch: masks batch of shape (batch_size, 1, n_image, n_image) or (batch_size, n_image, n_image, 1)
        :param transforms_types: a list of transforms to apply
            Must be one of str listed in transform_types_allowed variable
        :param prob: Probability of applying random transformations (the same for all transformations)
            Must be a float between 0 and 1

        Parameters defined in _validate_augment_image_and_masks_inputs method:
        :ivar transforms_types_allowed: Types of transforms that are currently implemented
            Example: "RandomHorizontalFlip", "RandomVerticalFlip"
        :type transforms_types_allowed: List[str]

        All params must have values as per validation function
        """
        # initialize parent class to validate
        super().__init__(images_batch, masks_batch, transforms_types, prob)

        self.images_batch = images_batch
        self.masks_batch = masks_batch
        self.transforms_types = transforms_types
        self.prob = prob

        # augment
        self.augment_batches()

    def augment_batches(self):
        """
        Augment batches of images and masks
        :return: a tuple of an images batch and masks batch after transformation
        """

        # reshape batches to correct size

        print("images_batch before reshape  ", self.images_batch)
        print("masks_batch before reshape  ", self.masks_batch)

        images_batch, masks_batch = reshape_batches(self.images_batch, self.masks_batch)

        print("images_batch  after reshape ", type(images_batch))
        print("masks_batch  after reshape ", type(masks_batch))

        # augment images together with masks using specified transform
        augmented_pairs = AugmentImageAndMask(images_batch=images_batch,
                                              masks_batch=masks_batch,
                                              transforms_types=self.transforms_types,
                                              prob=self.prob)

        images_batch, masks_batch = self.get_augmented_tensors(*augmented_pairs)

        images_batch = torch.stack(images_batch, dim=0)
        masks_batch = torch.stack(masks_batch, dim=0)

        return images_batch, masks_batch

    @staticmethod
    def get_augmented_tensors(*pairs):
        """
        Unzip batches of images and masks pairs into batches of images and batches masks
        :param pairs:
        :return:
        """
        if pairs:
            images_batch, masks_batch = list(zip(*pairs))
        else:
            raise ValueError("Input images and masks cannot be empty")

        return images_batch, masks_batch
