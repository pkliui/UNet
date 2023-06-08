# Examples

This document contains a sample task for the segmentation pipeline. 

The prerequisite is to create and activate conda environment as specified [here](setup.md)


## Sample segmentation task: Moles segmentation using PH2 Dataset

### Download and prepare dataset
1. The dataset is described in more detail in the section on datasets [here](dataset.md). Download it from the website `PH2 dermoscopic image database <https://www.fc.up.pt/addi/ph2%20database.html>` to your local storage or elsewhere. 

For segmentation purposes we are interested in *PH2 Dataset images* folder containing the image data. Make sure it has the following file structure:

```
PH2Dataset
├── PH_Dataset_images
│    ├── IMD002
│    │    ├── IMD002_Dermoscopic_Image
│    │    │    └── IMD002.bmp
│    │    └── IMD002_lesion
│    │         └── IMD002_lesion.bmp
│    ├── IMD003
│    │    ├── IMD003_Dermoscopic_Image
│    │    │    └── IMD003.bmp
│    │    └── IMD003_lesion
│    │         └── IMD003_lesion.bmp
│    ├──...
├── PH2_dataset.txt
└── PH2_dataset.xlsx
```

2. In current version, all the pre-processing happens on the batch level. The dataset can be used as it is.

### Create a config file for training

* Next, you should make a config file and put it somewhere on your machine. 

* In this file, you should specify input parameters for your training. All these arguments are necessary to initialize various classes for data loading and training.

Here is a config file to be used with Ray Tune for hyperparameter search

```python
from ray import tune


"""This is config file for PH2 data training to test how it works on local machine"""


def config():
    return  {"experiment_name": "ph2_test",
              "output_dir": "/Users/Pavel/ray_results/",
              "datapath": "/Users/Pavel/Documents/repos_data/UNet/PH2_Dataset_images/PH22/",
              "learning_rate": tune.choice([1e-1, 2e-1]),
              "step_size": 1,
              "gamma": 0.1,
              "batch_size": 2,
              "n_epochs": 1,
              "validation_split": 0.25,
              "test_split": 0.25,
              "metric": "avg_score",
              "metric_mode": "max",
              "checkpoint_frequency": 1,
              "checkpoint_num_to_keep": 1,
              "num_samples_tune": 10,
              "grace_period": 5}

```



### Setup training

We use Ray Tune framework for hyperparameter search and ASHA scheduler for early stopping. Training for PH2 data can be done by running the following command from the terminal:

```python
python path/to/repo/UNet/examples/example_ph2data/ph2_runner.py -config_file_path path/to/ph2hyperpar_config.py

```

We use ```Runner``` class that setups the training and specifically ```ph2_runner.py``` to train on PH2 data.

```python
import argparse
import ray
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler

from UNet.utils.utils import load_module
from UNet.training.PH2Trainer import PH2Trainer
from UNet.tuner_results_analysis import tuner_results_analysis

from pathlib import Path
from typing import Optional

import os

from UNet.utils.validation import validate_runner_inputs


class Runner(object):
    """
    This Runner class setups the high-level logic to start training:
        choose a model configuration class or a respective Trainer class by name,
        initialize Ray Tuner for hyperparameter search if needed,
        configure Ray Tuner and start training

    """

    def __init__(self,
                 config_file_path: str,
                 model_trainer_name: str,
                 model_config_name: Optional[str] = None):
        """
        :param config_file_path: Path to a config file for this particular experiment, given model and data,
            specifying paths to data, model hyperparameters, etc.
        :param model_trainer_name: Name of a trainer class specific to this particular experiment, given model and data
        :param model_config_name: optional. Name of a model configuration class in case a pre-trained model
                  available for this specific type of data. If it is not None,
                  then it overrides any variables set in config_file_path and model_trainer_name.
                  Default: None
        """

        validate_runner_inputs(config_file_path,
                               model_trainer_name,
                               model_config_name)

        self.config_file_path = config_file_path
        self.model_trainer_name = model_trainer_name
        self.model_config_name = model_config_name

        #
        if self.model_config_name is None:
            if self.model_trainer_name == 'PH2Trainer':
                self.trainer = PH2Trainer
        """Get the trainer class if not model configuration is available. 
        If model provided, the model configuration should be loaded instead! (TBI)"""
        #
        #
        self.config = self.get_config()
        """Get config dictionary"""


    def get_config(self) -> dict:
        """
        Get config dictionary
        :return:
        """
        module = load_module(self.config_file_path)
        print(f"Using config file: {self.config_file_path}")
        config = module.config()
        print(f"Using config  : {config}")
        return config

    def setup_tuner(self):
        #

        # initialize ray
        # ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
        ray.init()
        #
        # set up ASHA for early stopping
        sched = ASHAScheduler(time_attr='training_iteration',
                              max_t=self.config["n_epochs"],
                              brackets=1)  # one bracket as per Ray docs and discussion with the ASHA authors
        # set up tuner
        tuner = tune.Tuner(PH2Trainer,
                           run_config=air.RunConfig(checkpoint_config=air.CheckpointConfig(
                               checkpoint_frequency=self.config["checkpoint_frequency"],
                               num_to_keep=self.config["checkpoint_num_to_keep"],
                               checkpoint_score_attribute=self.config["metric"],
                               checkpoint_score_order=self.config["metric_mode"]),
                                                    local_dir=self.config["output_dir"],
                                                    name=self.config["experiment_name"]),
                           tune_config=tune.TuneConfig(scheduler=sched,
                                                       num_samples=self.config["num_samples_tune"],
                                                       metric=self.config["metric"],
                                                       mode=self.config["metric_mode"]
                                                       ),
                           param_space=self.config)

        return tuner


def run(config_file_path: str,
        model_trainer_name: str,
        model_config_name: Optional[str]):
    runner = Runner(config_file_path=config_file_path,
                    model_trainer_name=model_trainer_name,
                    model_config_name=model_config_name)
    # train
    result_grid = runner.setup_tuner().fit()
    return result_grid


def get_args() -> argparse.Namespace:
    """Get command-line arguments"""
    project_name = 'Arguments for Runner class'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('-config_file_path',
                        metavar='CONFIG_FILE_PATH',
                        type=str,
                        help='Path to the Ray Tunes python config file specific to PH2 dataset')
    parser.add_argument("--model_trainer_name",
                        metavar='MODEL_TRAINER_NAME',
                        type=str,
                        help="Name of a trainer class specific to this particular experiment.")
    parser.add_argument("--model_config_name",
                        metavar='MODEL_CONFIG_NAME',
                        type=str,
                        help="Name of a model configuration class in case a pre-trained model available.")
    return parser.parse_args()
```
 