# Training

## BaseTrainer

The training and validation loops are implemented in ```UNet.training.BaseTrainer``` class which can be called as follows:

```python
self.basetrainer = BaseTrainer(model=self.model,
                            loss_function=self.loss_function,
                            optimizer=self.optimizer,
                            data_loader=self.data_loader,
                            metric=self.metric,
                            n_epochs=self.n_epochs,
                            lr_sched=self.lr_sched,
                            device=self.device,
                            early_stop_save_dir=self.output_dir,
                            early_stop_patience=self.early_stop_patience,
                            save_dir=self.output_dir,
                            experiment_name=self.experiment_name)
```

## PH2Trainer

Specifically for PH2 dataset, use ```UNet.training.PH2Trainer``` class 

```python

import os

from UNet.data_handling.base import BaseDataLoader
from UNet.data_handling.ph2dataset import PH2Dataset
from UNet.training.base_trainer import BaseTrainer
from UNet.metric.metric import iou_tgs_challenge
import torch.optim as optim
import torch
from ray import tune

import torch.nn as nn
from UNet.utils.utils import make_directory
from UNet.models.unet import UNet
from UNet.utils.validation import validate_config_ph2data


class PH2Trainer(tune.Trainable):

    def setup(self, config: dict):
        """
        Load model hyperparameters and other inputs from config file,
        load data and create dataloaders

        :param config: A dictionary of hyperparameters to search amongst.

            Required fields are
            "experiment_name", "output_dir", "datapath", "learning_rate",
            "step_size", "gamma", "batch_size", "n_epochs",
            "validation_split", "test_split"

            Example:
            config = {"experiment_name": "ph2_test",
              "output_dir": "/Users/Pavel/ray_results/",
              "datapath": "/Users/Pavel/Documents/repos_data/UNet/PH2_Dataset_images/PH22/",
              "learning_rate": tune.choice([1e-1, 2e-1]),
              "step_size": 1,
              "gamma": 0.1,
              "batch_size": tune.choice([1, 2]),
              "n_epochs": 2,
              "validation_split": 0.25,
              "test_split": 0.25}
        """
        # validate inputs
        validate_config_ph2data(config)

        self.experiment_name = config["experiment_name"]
        """str: name of experiment, will be used as folder name to save results under output_dir"""
        self.output_dir = make_directory(config["output_dir"])
        """str: full path to a directory to save results"""
        self.datapath = config["datapath"]
        """str: full path to a directory with data"""
        self.learning_rate = config["learning_rate"]
        """float: learning rate"""
        self.step_size = config["step_size"]
        """int: step size for learning rate decay, in epochs"""
        self.gamma = config["gamma"]
        """float: learning rate decay rate"""
        self.batch_size = config["batch_size"]
        """int: image batch size to be processed at single epoch"""
        self.n_epochs = config["n_epochs"]
        """int: number of epochs"""
        self.validation_split = config["validation_split"]
        """float: fraction of data to be used for validation, float between 0 and 1"""
        self.test_split = config["test_split"]
        """float: fraction of data to be used for testing, float between 0 and 1"""

        # other model-specific hyperparameters
        self.early_stop_patience = 5
        """int: early stop patience, in epochs"""
        self.size_images = (572, 572)
        """tuple of int: size of input images"""
        self.size_masks = (388, 388)
        """tuple of int: size of input segmented images"""
        self.model = UNet()
        """model to train on the PH2 data, original UNet model for now"""
        self.loss_function = nn.BCEWithLogitsLoss()
        """loss function"""
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.learning_rate)
        """optimizer"""
        self.lr_sched = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                  step_size=self.step_size,
                                                  gamma=self.gamma)
        """learning rate scheduler"""
        self.metric = iou_tgs_challenge
        """metric"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """torch device"""
        #
        # create dataset for training, validation and testing
        self.ph2_dataset = PH2Dataset(root_dir=self.datapath,
                                      required_image_size=self.size_images,
                                      required_mask_size=self.size_masks,
                                      resize_required=True)
        print("self.ph2_dataset   ", self.ph2_dataset)
        # Create dataloader
        if len(self.ph2_dataset) > 0:
            self.data_loader = BaseDataLoader(dataset=self.ph2_dataset,
                                              batch_size=self.batch_size,
                                              validation_split=self.validation_split,
                                              test_split=self.test_split,
                                              random_seed_split=42)
        else:
            raise ValueError(f"Dataset is empty, check the folder {self.ph2_dataset} "
                             f"has images and they are read correctly len({self.ph2_dataset})={len(self.ph2_dataset)}")

        self.basetrainer = BaseTrainer(model=self.model,
                                  loss_function=self.loss_function,
                                  optimizer=self.optimizer,
                                  data_loader=self.data_loader,
                                  metric=self.metric,
                                  n_epochs=self.n_epochs,
                                  lr_sched=self.lr_sched,
                                  device=self.device,
                                  early_stop_save_dir=self.output_dir,
                                  early_stop_patience=self.early_stop_patience,
                                  save_dir=self.output_dir,
                                  experiment_name=self.experiment_name)


    def step(self) -> dict:
        """
        Train the model using Base Trainer

        :return results_dict: dictionary as per BaseTrainer documentation
        """
        results_dict = self.basetrainer.train()
        return results_dict

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


```
