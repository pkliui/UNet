import unittest
from ddt import ddt
import torch

import shutil, tempfile


from UNet.training.base_trainer import BaseTrainer
from UNet.data_handling.unetdataset import UNetDataset
from UNet.classes.preprocess import Resize
from UNet.data_handling.base import BaseDataLoader
from UNet.models.unet import UNet
from torchvision import transforms


import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from UNet.data_handling.base import BaseDataLoader
from UNet.data_handling.unetdataset import UNetDataset
from UNet.classes.preprocess import Resize

from UNet.models.unet import UNet

from UNet.metrics import bce_loss, iou_pytorch



@ddt
class TestBaseTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    #def setUp(self):
    #    # create an instance of BaseTrainer class
    #    self.basetrainer = BaseTrainer()
    #    # create a temporary directory
    #    self.test_dir = tempfile.mkdtemp()

    #def tearDown(self):
    #    # remove temporary directory after the test
#   #     shutil.rmtree(self.test_dir)

    #def test_arguments(self):
    #    """
    #    test input arguments are existing and are either None or equal to expected default values
    #    """
    #    for var in ["model"]:
    #        self.assertIn(var, self.basetrainer.__dict__)

    def test_trainer_check_batch_size(self):
        # specify the root dir, images' and masks' folders and the image size
        extension = "*.bmp"
        #extension = "*.jpeg"
        root_dir = "/Users/Pavel/Documents/repos/UNet/docs/data/PH2_Dataset_images/"
        images_folder = "images"
        masks_folder = "masks"

        SIZE_X = (572, 572) # size of input images
        SIZE_Y = (388, 388) # size of input segmented images

        batch_size = 1
        validation_split = 0
        shuffle_for_split = True
        random_seed_split = 0
        #
        # specify an image transform function
        transform = transforms.Compose([Resize(SIZE_X, SIZE_Y)])
        #
        # read masks and images
        unet_data = UNetDataset(root_dir, images_folder, masks_folder, extension, transform=transform)
        # make a dataloader from the dataset
        data_loader = BaseDataLoader(dataset=unet_data,
                                     batch_size=batch_size,
                                     validation_split=validation_split,
                                     shuffle_for_split=shuffle_for_split,
                                     random_seed_split=random_seed_split)

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Define the model
        model = UNet()#.to(device)
        save_dir = root_dir

        # Define the loss function and the optimizer
        criterion = bce_loss()
        optimizer = optim.AdamW(model.parameters(), lr=0.4)

        #import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        basetrainer = BaseTrainer(model=model,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  data_loader=data_loader,
                                  epochs=1)

        basetrainer.train()

        #rint("xbatch len", len(basetrainer.xbatch))
