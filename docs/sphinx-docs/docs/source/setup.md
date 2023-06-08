# Setup


## Operating system
The code was run and tested on MacOS Big Sur 11.2.3


## Clone repository

* Clone the repo by
```
git clone https://github.com/pkliui/UNet/
cd UNet
```

* The code code in python was edited using [PyCharm 2023.1.1 Community Edition](<https://www.jetbrains.com/pycharm/download/#section=mac>).


## Set up conda

[Conda](https://docs.conda.io/en/latest/) is an open source package management system. All python packages in SkinSeg are managed with conda.


### Windows/MacOS Users

* Check if you have conda already installed by running in your shell

```
conda --version
```

* If you an error "command not found", install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [conda](https://docs.conda.io/en/latest/). The first one is more light-weight version. The SkinSeg code was made under conda 4.10.3.


## Create a conda environment

* Make sure you are inside the project directory, then run

```
conda env create --file environment.yml
conda activate unet-env
```

* Build tools should have been automatically set up with Conda distribution



