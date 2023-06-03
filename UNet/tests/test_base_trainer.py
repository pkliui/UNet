import unittest

from PIL import Image
from ddt import ddt
import shutil
import tempfile
import numpy as np
import torch
import os
import random
import torch.optim as optim

from UNet.training.base_trainer import BaseTrainer
from UNet.data_handling.base import BaseDataLoader
from UNet.data_handling.unetdataset import UNetDataset
from UNet.models.unet import UNet
from UNet.metric.metric import iou_tgs_challenge


def set_seed(seed):
    """
    fix seed for reproducible results
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


@ddt
class TestBaseTrainer(unittest.TestCase):

    def setUp(self):
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_basetrainer_returns(self):
        """
        Check returns of the basetrainer
        """

        # Create dummy image and mask folders and files
        sample_data = []
        num_images = 10

        for i in range(1, num_images + 1):
            sampleID = f"sampleID{i}"
            image_file = f"image{i}.png"
            mask_file = f"mask{i}.png"
            sample_data.append({
                "sampleID": sampleID,
                "image_file": image_file,
                "mask_file": mask_file
            })

            # make sample folder containing images and masks
            sample_folder = os.path.join(self.test_dir, sampleID)
            os.makedirs(sample_folder)

            # make image folder and create a dummy image in it
            image_folder = os.path.join(sample_folder, f"{sampleID}_image")
            os.makedirs(image_folder)
            image_file_path = os.path.join(image_folder, image_file)
            random_array = np.random.randint(low=0, high=256, size=(572, 572, 3), dtype=np.uint8)
            image = Image.fromarray(random_array, 'RGB')
            image.save(image_file_path, "PNG")

            # make mask folder and create a dummy mask in it
            mask_folder = os.path.join(sample_folder, f"{sampleID}_mask")
            os.makedirs(mask_folder)
            mask_file_path = os.path.join(mask_folder, mask_file)
            random_array = np.random.randint(low=0, high=1, size=(388, 388), dtype=np.uint8)
            mask = Image.fromarray(random_array)
            mask.save(mask_file_path, "PNG")

        # Create the UNetDataset with the temporary image and mask files
        images_list = [os.path.join(self.test_dir,
                                    sample["sampleID"],
                                    f"{sample['sampleID']}_image",
                                    sample["image_file"]) for sample in sample_data]
        masks_list = [os.path.join(self.test_dir,
                                   sample["sampleID"],
                                   f"{sample['sampleID']}_mask",
                                   sample["mask_file"]) for sample in sample_data]

        # make unetdataset
        unet_data = UNetDataset(images_list=images_list,
                                masks_list=masks_list,
                                transform=None)

        # create data loader - only train data
        data_loader = BaseDataLoader(dataset=unet_data,
                                     batch_size=2,
                                     validation_split=0.4,
                                     shuffle_for_split=True,
                                     random_seed_split=0)
        # set base trainer
        set_seed(42)
        basetrainer = BaseTrainer(model = UNet().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                  metric = iou_tgs_challenge,
                                  loss_function = torch.nn.BCEWithLogitsLoss(),
                                  optimizer = optim.AdamW(UNet().parameters(), lr=5e-1),
                                  data_loader=data_loader,
                                  n_epochs=1,
                                  lr_sched = optim.lr_scheduler.StepLR(optimizer = optim.AdamW(UNet().parameters(), lr=5e-1), step_size=50, gamma=0.1),
                                  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                  save_dir=self.test_dir)

        result = basetrainer.train()

        # Perform assertions to check the correctness of the results
        self.assertIsInstance(result, dict)
        self.assertIn("avg_val_loss", result)
        self.assertIn("avg_score", result)

        # Assert that the average validation loss is a numeric value
        self.assertIsInstance(result["avg_val_loss"], float)

        # Assert that the average score is a numeric value
        self.assertIsInstance(result["avg_score"], float)

        # Assert that the average validation loss is not negative
        self.assertGreaterEqual(result["avg_val_loss"], 0)

        # Assert that the average score is between 0 and 1 (inclusive)
        self.assertGreaterEqual(result["avg_score"], 0)
        self.assertLessEqual(result["avg_score"], 1)

        # Assert that the average validation loss is finite (not NaN or Inf)
        self.assertTrue(np.isfinite(result["avg_val_loss"]))

        # Assert that the average score is finite (not NaN or Inf)
        self.assertTrue(np.isfinite(result["avg_score"]))

