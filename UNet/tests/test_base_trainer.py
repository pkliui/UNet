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

from UNet.metrics import iou_pytorch


import numpy as np
import torch, os
import random
import torch.nn as nn


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



    # fix seed for reproducible results
    def set_seed(self, seed):
        torch.manual_seed(seed)
        #torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def test_trainer_check_batch_size(self):
        #
        # set data loader
        #
        # specify the size of the input and output images
        SIZE_X = (572, 572)
        SIZE_Y = (388, 388)
        BATCH_SIZE = 4
        #
        # read data
        unet_data = UNetDataset(root_dir="/Users/Pavel/Documents/repos/UNet/docs/data/PH2_Dataset_images/",
                                images_folder="images", masks_folder="masks", extension="*.bmp",
                                transform=transforms.Compose([Resize(SIZE_X, SIZE_Y)]))
        # create data loader - only train data
        data_loader = BaseDataLoader(dataset=unet_data,
                                     batch_size=BATCH_SIZE,
                                     validation_split=0,
                                     shuffle_for_split=True,
                                     random_seed_split=0)
        print(data_loader.__dict__)
        if hasattr(data_loader, "val_loader"):
            print("has val loader")
        else:
            print("no val loader")
        #
        # set base trainer
        self.set_seed(42)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # define the model
        model = UNet().to(device)
        # define the quality metric
        metric = iou_pytorch
        # define the loss function and the optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=5e-3)
        #save_dir = root_dir
        #import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)
        # scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)
        basetrainer = BaseTrainer(model=model,
                                  metric=metric,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  data_loader=data_loader,
                                  epochs=1,
                                  lr_sched=scheduler,
                                  device=device)

        basetrainer.train()

        print("xbatch len", len(basetrainer.xbatch))
        self.assertEqual(len(basetrainer.xbatch), BATCH_SIZE)
        self.assertEqual(len(basetrainer.ybatch), BATCH_SIZE)

        #print("data loader", len(val_loader.train_loader.dataset))
        print("data loader", len(data_loader.train_loader.dataset))
