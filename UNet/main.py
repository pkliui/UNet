#import packages
import shutil
import sys
import glob, os, fnmatch
import subprocess
import threading

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from UNet.data_handling.base import BaseDataLoader
from UNet.models.unet import UNet
from UNet.training.base_trainer import BaseTrainer
from UNet.data_handling.unetdataset import UNetDataset
from UNet.classes.preprocess import Resize
from UNet.metric.metric import iou_tgs_challenge

from UNet.evaluation.evaluation import evaluate_model
from UNet.metric.metric import dice_coefficient


# use cuda if available
import torch

from UNet.utils.augment import AugmentImageAndMask, random_transforms, get_augmented_tensors


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# set parameters
DATAPATH = os.path.abspath("/Users/Pavel/Documents/repos/UNet/docs/data/PH2_Dataset_images/")
OUTPUT_DIR = "/Users/Pavel/Documents/repos/UNet/docs/output"

SIZE_X = (572, 572) # size of input images
SIZE_Y = (388, 388) # size of input segmented images

LEARNING_RATE = 1e-1 # learning rate
BATCH_SIZE = 1 # batch size
VALIDATION_SPLIT = 0.25 # validation split
TEST_SPLIT = 0.25 # test split

MAX_EPOCHS = 1 # number of epochs

SCHEDULER_STEP = 50 # scheduler step
SCHEDULER_GAMMA = 0.1 # scheduler gamma
EARLY_STOP_PATIENCE = 2 # early stopping patience

# make output directory to keep the model
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
else:
    os.makedirs(OUTPUT_DIR)


unet_dataset = UNetDataset(
    root_dir=DATAPATH,
    images_folder="images",
    masks_folder="masks",
    extension="*.bmp",
    transform=transforms.Compose([Resize(SIZE_X, SIZE_Y)]))

# Create the corresponding dataloader for training and validation
data_loader = BaseDataLoader(dataset=unet_dataset,
                             batch_size=BATCH_SIZE,
                             validation_split=VALIDATION_SPLIT,
                             test_split=TEST_SPLIT
                             )

# Define the model
unet_model = UNet()
save_dir = os.getcwd()+'/runs/exp1'

# Define the loss function and the optimizer
bce_loss = nn.BCEWithLogitsLoss()
unet_optimizer = optim.AdamW(unet_model.parameters(), lr=LEARNING_RATE)

# scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer=unet_optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

# Initialize the trainer
basetrainer = BaseTrainer(model = unet_model,
                          loss_function = bce_loss,
                          metric = iou_tgs_challenge,
                          optimizer = unet_optimizer,
                          data_loader = data_loader,
                          n_epochs = MAX_EPOCHS,
                          lr_sched=scheduler,
                          device = device,
                          early_stop_save_dir=DATAPATH,
                          early_stop_patience=EARLY_STOP_PATIENCE,
                          save_dir=DATAPATH)

# call tensorboard
log_dir = os.path.join(DATAPATH, 'summaries')

def start_tensorboard(logdir):
    subprocess.call(['tensorboard', '--logdir', logdir])

tb_thread = threading.Thread(target=start_tensorboard, args=(log_dir,))
tb_thread.start()

# Start training
train_loss_values, val_loss_values, avg_score_values = basetrainer.train()

# Evaluate the model on the test set
score, accuracy, precision, recall, F1_score, conf_matrix = evaluate_model(unet_model, data_loader.test_loader, device, dice_coefficient)


