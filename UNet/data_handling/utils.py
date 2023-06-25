import os
import fnmatch
from typing import Tuple, List
import glob
import cv2
import numpy as np


"""
Modules to treat raw data before training
"""


def list_files_in_dir(path_to_dir: str,
                      extension: str):
    """
    Returns a list of files names in a specific directory with a specific extension

    :param path_to_dir: path to directory to survey
    :param extension: file extension to seach for
    Args:
        path_to_dir:
        extension:
    """
    return [f for f in os.listdir(path_to_dir)
            if os.path.isfile(os.path.join(path_to_dir, f))
            and fnmatch.fnmatch(f, extension) and f[0] != '.']


def get_list_of_data(folder_with_image: str,
                     image_extension: str,
                     list_of_images: List[str]):
    """
    Appends to and returns a list of full paths to image(s) with specified extension and location

    :param folder_with_image: full path to a folder with an image, the folder is expected to have one image only
    :param image_extension: image extension
        example: "bmp"
    :param list_of_images: can be an empty list or a list of strings containing full paths to images

    :return: list_of_images: a list of strings containing full paths to images, or empty if no images found
    """
    # read out data (e.g. images or masks), only one image per folder is allowed
    if os.path.exists(folder_with_image):
        image_name = list_files_in_dir(folder_with_image, f"*.{image_extension}")
        print("image_name ", image_name)
        print("len image name ", len(image_name))
        if len(image_name) == 1:
            list_of_images.append(os.path.join(folder_with_image, str(image_name[0])))
        else:
            raise ValueError(f"Only one image per folder is expected. Skipping data in folder {folder_with_image}")
    else:
        raise ValueError("Nonexistent folder " + folder_with_image)

    return list_of_images


# def set_image_size(input_dir: str,
#                    extension: str,
#                    target_size: Tuple[int, int],
#                    image_type: str):
#     """
#     Resize images to have the size expected by UNet by nearest neighbour interpolation.
#     This is important because using any other interpolation "may result in tampering with the ground truth labels"
#     [https://ai.stackexchange.com/questions/6274/how-can-i-deal-with-images-of-variable-dimensions-when-doing-image-segmentation]
#
#     If the size of the image is not as per target size, resizes and saves the resized version
#
#     :param input_dir: input directory where original images reside
#     :param extension: image extension
#         examples: 'bmp', 'tiff'
#     :param target_size: target size to crop images to
#         examples: (572,572)
#     :param image_type: must be either image or mask, this will be used later to normalize the images upon resizing
#         examples: 'image' or 'mask'
#     """
#     if not os.path.exists(input_dir):
#         raise ValueError(f"Input directory '{input_dir}' does not exist.")
#
#     # Use glob to find all files in input directory with specified extensions
#     files = glob.glob(os.path.join(input_dir, f"*.{extension}"))
#
#     # loop through images in input directory
#     for filename in files:
#         if os.path.exists(filename):
#             image = cv2.imread(filename)
#             #
#             # check if image is already of correct size
#             if image.size == (target_size[0], target_size[1]):
#                 print(f'Skipping {filename}, image is already of correct size')
#             # resize image
#             if image_type == 'image':
#                 image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
#                 image_resized = image_resized / np.max(image_resized)
#                 cv2.imwrite(filename, image_resized)
#             elif image_type == 'mask':
#                 image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST) > 0.5
#                 print(image_resized.shape)
#                 #image_resized = image_resized / np.max(image_resized)
#
#                 #cv2.imwrite(f"{os.path.splitext(filename)[0]}.tiff", image_resized)
#                 #cv2.imwrite(filename, image_resized, [cv2.IMWRITE_PXM_BINARY, 1])
#                 # Apply threshold to convert the image to binary
#                 ret, img_binary = cv2.threshold(image_resized, 128, 1, cv2.THRESH_BINARY)
#                 # Convert the binary image to a 1-channel numpy array of dtype uint8
#                 img_1ch = np.uint8(img_binary.reshape(img_binary.shape[0], img_binary.shape[1], 1))
#
#                 # Save the image
#                 cv2.imwrite(filename, img_1ch)
#             else:
#                 raise ValueError("Argument 'image_type' can be only 'image' or 'mask'.")
#
#             #image_resized =  Image.fromarray(image_resized)
#
#
#             # save resized image to output directory
#             #image_resized.save(filename)
#         else:
#             raise ValueError(
#                     f"No file '{filename}' found in input directory!")
