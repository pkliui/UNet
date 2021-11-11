import unittest
from ddt import ddt

import shutil, tempfile


from UNet.training.base_trainer import BaseTrainer
from UNet.classes.unetdataset import UNetDataset
from UNet.classes.preprocess import Resize, SplitDataLoader

from torchvision import transforms


@ddt
class TestBaseTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # create an instance of BaseTrainer class
        self.basetrainer = BaseTrainer()
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_arguments(self):
        """
        test input arguments are existing and are either None or equal to expected default values
        """
        for var in ["model"]:
            self.assertIn(var, self.basetrainer.__dict__)

    def test_trainer_check_batch_size(self):
        images_folder = "Dermoscopic_Image"
        masks_folder = "lesion"
        root_dir = "/UNet/data/PH2_Dataset_images/"
        files_structure = 2
        batch_size = 2

        im_size = (100,100)
        transform = transforms.Compose([
            Resize(im_size, im_size)
        ])
        tr_size = 4
        vl_size = 4
        ts_size = 2
        #
        # read data into a dataset
        unet_data = UNetDataset(root_dir=root_dir, images_folder=images_folder,
                                masks_folder=masks_folder, transform=transform,
                                files_structure=files_structure)
        #
        #
        # make a dataloader from the dataset
        data_loader = SplitDataLoader(dataset=unet_data,
                              batch_size=batch_size,
                              tr=tr_size, vl=vl_size, ts=ts_size)

        basetrainer = BaseTrainer(data_loader=data_loader,
                                     epochs=1)

        basetrainer.train()

        print("xbatch len", len(basetrainer.xbatch))
