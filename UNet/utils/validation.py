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
        "learning_rate": learning rate
        "step_size": step size for learning rate decay, in epochs
        "gamma": learning rate decay rate
        "batch_size": image batch size to be processed at single epoch
        "n_epochs": number of epochs
        "validation_split": fraction of data to be used for validation, float between 0 and 1
        "test_split": fraction of data to be used for testing, float between 0 and 1
    :raises ValueError if input data types are not as expected
    """

    required_keys = ["experiment_name", "output_dir", "datapath", "learning_rate",
                     "step_size", "gamma", "batch_size", "n_epochs",
                     "validation_split", "test_split"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    # Perform additional validation for specific keys
    if not isinstance(config["experiment_name"], str):
        raise ValueError("experiment_name must be a string")

    if not isinstance(config["output_dir"], str):
        raise ValueError("output_dir must be a string")

    if not isinstance(config["datapath"], str):
        raise ValueError("datapath must be a string")

    if not isinstance(config["learning_rate"], float):
        raise ValueError("learning_rate must be a float")

    if not isinstance(config["step_size"], int):
        raise ValueError("step_size must be an integer")

    if not isinstance(config["gamma"], float):
        raise ValueError("gamma must be a float")

    if not isinstance(config["batch_size"], int):
        raise ValueError("batch_size must be an integer")

    if not isinstance(config["n_epochs"], int):
        raise ValueError("n_epochs must be an integer")

    if not isinstance(config["validation_split"], float):
        raise ValueError("validation_split must be a float")

    if not isinstance(config["test_split"], float):
        raise ValueError("test_split must be a float")

