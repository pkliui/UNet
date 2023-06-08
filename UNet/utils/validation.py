"""This module contains functions for validation fo input config files"""

import os
import torch
from typing import List


def validate_runner_inputs(config_file_path,
                           model_trainer_name,
                           model_config_name):
    """
    Validate inputs to Runner class

    Currently Runner class supports only PH2 data, hence the validation values in here

    :param config_file_path: Path to a config file for this particular experiment, given model and data,
        specifying paths to data, model hyperparameters, etc.
    :param model_trainer_name: Name of a trainer class specific to this particular experiment, given model and data
    :param model_config_name: optional. Name of a model configuration class in case a pre-trained model
          available for this specific type of data. If it is not None,
          then it overrides any variables set in config_file_path and model_trainer_name
    """

    if not os.path.exists(config_file_path):
        raise ValueError(f"Path to config file {config_file_path} does not exist.")

    model_trainer_names = ["PH2Trainer"]
    if model_trainer_name not in model_trainer_names:
        raise ValueError(f'Please enter valid model trainer name. Currently supported values are {model_trainer_names}')

    # no model ready yet
    # model_config_name


def validate_config_ph2data(config: dict):
    """
    Validates config dictionary for PH2 dataset

    PH2 data copyright: Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira.
    PH² - A dermoscopic image database for research and benchmarking,
    35th International Conference of the IEEE Engineering in Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.

    :param config: dictionary with input parameters needed to train PH2 data, required fields are
        "experiment_name": name of experiment, will be used as folder name to save results under output_dir
        "output_dir": full path to a directory to save results
        "datapath": full path to a directory with data
        "learning_rate": learning rate, must be between 0 and 1
        "step_size": step size for learning rate decay, in epochs
        "gamma": multiplicative factor of learning rate decay
        "batch_size": image batch size to be processed at single epoch
        "n_epochs": number of epochs
        "validation_split": fraction of data to be used for validation, float between 0 and 1
        "test_split": fraction of data to be used for testing, float between 0 and 1
        "metric": name of metric used to monitor training, must be "avg_val_loss" or "avg_score"
        "metric_mode": min or max, depending on the metric
        "checkpoint_frequency":  Number of iterations between checkpoints. If 0 this will disable checkpointing.
        "checkpoint_num_to_keep": The number of checkpoints to keep on disk for this run. If a checkpoint is persisted
            to disk after there are already this many checkpoints, then an existing checkpoint will be deleted.
            If this is None then checkpoints will not be deleted. Must be >= 1.
        "num_samples_tune":  Number of times to sample from the hyperparameter space.
            If grid_search is provided as an argument, the grid will be repeated num_samples of times.
            If this is -1, (virtually) infinite samples are generated until a stopping condition is met.


    :raises ValueError if input data types are not as expected
    """

    required_keys = ["experiment_name", "output_dir", "datapath", "learning_rate", "step_size", "gamma", "batch_size",
                     "n_epochs", "validation_split", "test_split", "metric", "metric_mode", "checkpoint_frequency",
                     "checkpoint_num_to_keep", "num_samples_tune"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    # Perform additional validation for specific keys
    if not isinstance(config["experiment_name"], str):
        raise ValueError("experiment_name must be a string")

    if not os.path.isdir(config["output_dir"]):
        raise ValueError(f"Output directory '{config['output_dir']}' does not exist")
    #
    if not os.path.isdir(config["datapath"]):
        raise ValueError(f"Datapath directory '{config['datapath']}' does not exist")
    #
    if (not isinstance(config["learning_rate"], int)) and not (0 < config["learning_rate"] <= 1):
        raise ValueError("learning_rate must be a float between 0 and 1")

    if (not isinstance(config["step_size"], int)) and config["step_size"] <= 0:
        raise ValueError("step_size must be a positive integer")

    if not isinstance(config["gamma"], float):
        raise ValueError("gamma must be a float")

    if (not isinstance(config["batch_size"], int)) and config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")

    if (not isinstance(config["n_epochs"], int)) and config["n_epochs"] <= 0:
        raise ValueError("n_epochs must be a positive integer")

    if (not isinstance(config["validation_split"], float)) and (0 < config["validation_split"] < 1):
        raise ValueError("validation_split must be a float between 0 and 1")

    if (not isinstance(config["test_split"], float)) and not (0 < config["test_split"] < 1):
        raise ValueError("test_split must be a float between 0 and 1")

    if config["metric"] not in ["avg_val_loss", "avg_score"]:
        raise ValueError('metric_mode must be either "avg_val_loss", "avg_score"')

    if config["metric_mode"] not in ["min", "max"]:
        raise ValueError("metric_mode must be either 'min' or 'max'")

    if (not isinstance(config["checkpoint_frequency"], int)) and config["checkpoint_frequency"] < 0:
        raise ValueError("checkpoint_frequency must be a non-negative integer")

    if (not isinstance(config["checkpoint_num_to_keep"], int)) and config["checkpoint_num_to_keep"] < 1:
        raise ValueError("checkpoint_num_to_keep must be a positive integer")

    if (not isinstance(config["num_samples_tune"], int)) and config["num_samples_tune"] < 1:
        raise ValueError("num_samples_tune must be an integer >= 1")


