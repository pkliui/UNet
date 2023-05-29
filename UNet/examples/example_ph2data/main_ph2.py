import argparse
import ray
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler

from UNet.utils.utils import load_module
from UNet.training.PH2Trainer2 import PH2Trainer2
from UNet.tuner_results_analysis import tuner_results_analysis

"""
This is an example file to train UNet on PH2 data using Ray Tune hyperparameter search framework

Usage:
Call from command line within the respective conda environment via
    python path/to/main_ph2.py --config_path /path/to/ph2_config_file.py
"""


def get_args() -> argparse.Namespace:
    """Get command-line arguments"""
    project_name = 'PH2 Dataset Segmentation with UNet'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('--config_path',
                        metavar='CONFIG_PATH',
                        type=str,
                        help='Path to the Ray Tunes python config file specific to PH2 dataset')
    return parser.parse_args()


if __name__ == "__main__":
    #
    # get input arguments
    args = get_args()
    #
    # get path to a config file
    config_path = args.config_path
    print(f"Using config file: {config_path}")

    # get config
    module = load_module(config_path)
    config = module.config()
    print(f"Using config  : {config}")

    # initialize ray
    # ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
    ray.init(address=args.ray_address)
    #
    # set up ASHA for early stopping
    sched = ASHAScheduler(time_attr='training_iteration',
                          max_t=config["n_epochs"],
                          grace_period=1,
                          brackets=1)  # one bracket as per Ray docs and discussion with the ASHA authors
    # set up tuner
    tuner = tune.Tuner(PH2Trainer2,
                       run_config=air.RunConfig(
                           checkpoint_config=air.CheckpointConfig(checkpoint_frequency=config["checkpoint_frequency"],
                                                                  num_to_keep=config["checkpoint_num_to_keep"],
                                                                  checkpoint_score_attribute=config["metric"],
                                                                  checkpoint_score_order=config["metric_mode"]),
                           local_dir=config["output_dir"],
                           name=config["experiment_name"]),
                       tune_config=tune.TuneConfig(scheduler=sched,
                                                   num_samples=2,
                                                   metric=config["metric"],
                                                   mode=config["metric_mode"]
                                                   ),
                       param_space=config)
    # train
    result_grid = tuner.fit()
    #
    # display results
    tuner_results_analysis(result_grid=result_grid,
                           trainable=PH2Trainer2,
                           metric=config["metric"],
                           mode=config["metric_mode"])
