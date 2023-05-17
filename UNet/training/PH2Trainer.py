from ray.air.integrations.wandb import setup_wandb

from UNet.data_handling.base import BaseDataLoader
from UNet.data_handling.ph2dataset import PH2Dataset
from UNet.training.base_trainer import BaseTrainer
from UNet.classes.preprocess import Resize
from UNet.metric.metric import iou_tgs_challenge
import torch.optim as optim
import torch
from ray import tune

import torch.nn as nn
from UNet.utils.utils import make_directory
from UNet.models.unet import UNet


class PH2Trainer(BaseTrainer, tune.Trainable):

    def __init__(self, config: dict):
        """
        config: dict
            A dictionary of hyperparameters to search amongst
            Must contain mandatory following keys:
            config = {
                        "learning_rate": tune.loguniform(1e-3, 1e-1),
                        "step_size ": tune.choice([10,20]),
                        "gamma": tune.choice([0.1, 0.3]),
                        "batch_size": tune.choice([4, 8, 16]),
                        "n_epochs": tune.choice([10, 30, 60]),
                        "output": tune.choice([10, 30, 60])
                    }
        """

        # hyperparameters to vary from config
        self.learning_rate = config["learning_rate"]
        self.step_size = config["step_size"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]

        # fixed hyperparameters from config
        self.output_dir = make_directory(config["output_dir"])
        self.datapath = config["datapath"]
        self.validation_split = config["validation_split"]
        self.test_split = config["test_split"]

        # other model-specific hyperparameters
        self.early_stop_patience = 5 # early stop patience
        self.size_images = (572, 572)  # size of input images
        self.size_masks = (388, 388)  # size of input segmented images
        self.model = UNet() # model

        self.loss_function = nn.BCEWithLogitsLoss()  # loss function
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.lr_sched = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                             step_size=self.step_size,
                                             gamma=self.gamma)
        self.metric = iou_tgs_challenge
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create a new unet dataset
        ph2_dataset = PH2Dataset(
            root_dir=self.datapath,
            transform=nn.Sequential(Resize(self.size_images, self.size_masks)))

        # Create the corresponding dataloader for training and validation
        data_loader = BaseDataLoader(dataset=ph2_dataset,
                                     batch_size=self.batch_size,
                                     validation_split=self.validation_split,
                                     test_split=self.test_split,
                                     random_seed_split=42
                                     )

        super(BaseTrainer).__init__(
            model=self.model,
            loss_function=self.loss_function,
            optimizer=self.optimizer,
            data_loader=data_loader,
            metric=self.metric,
            n_epochs=self.n_epochs,
            lr_sched=self.lr_sched,
            device=self.device,
            early_stop_save_dir=self.output_dir,
            early_stop_patience=self.early_stop_patience,
            save_dir=self.output_dir
        )
