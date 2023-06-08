# Pre-processing

Images and masks are pre-processed in a few ways prior to their use in the framework. 

## Resizing

Resizing is done using the ```UNet.utils.resize_data.ResizeData``` class. 

``` python
class ResizeData(nn.Module):
    """
    Class to resize images and masks and normalize pixel values to 1
    """

    def __init__(self,
                 new_image_size: Tuple[int, int],
                 new_mask_size: Tuple[int, int]):
        """
        :param new_image_size: New output image size specified as tuple of 2 integers (no channel size)
            Example: (572, 572)
        :param new_mask_size: New output mask size specified as tuple of 2 integers (no channel size)
            Example: (388, 388)
        """
```

Required image and mask sizes can be parsed as tuples of shape 2, i.e. ```(572, 572)``` for UNet images and data should be provided as a dictionary to the return of Sequential container as follows, for example:

```python
resizer = nn.Sequential(ResizeData((572, 572)), (388, 388))
{'image': RESIZED_IMAGE, 'mask': RESIZED_MASK} = resizer({'image': YOUR_IMAGE_DATA, 'mask': YOUR_MASK_DATA})
```
 

## Augmentation

Images and masks are augmented prior to their use in training using ```UNet.utils.augment.AugmentImageAndMask``` class by providing respective batches, transformation types to be applied and transformation probability. The augmentation is currently done on the batch level, i.e. the class requires batches as input and it happens in the ```BaseTrainer```'s  training loop.

```python
images, masks = AugmentBatch(images_batch=batch['image'],
                                             masks_batch=batch['mask'],
                                             transforms_types=self.transforms_types,
                                             prob=self.prob)
```

Transform types must be compliant with one of the currently implemented transforms ```["RandomHorizontalFlip" "RandomVerticalFlip"]```. See more information in the docstrings of the class. 

