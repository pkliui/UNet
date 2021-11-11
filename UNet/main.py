import torch
import torch.optim as optim
from train import train
from unet import UNet
from metrics  import bce_loss, iou_pytorch

#
# set parameters
#
SIZE_X = (572, 572) # size of input images
SIZE_Y = (388, 388) # size of input segmented images
#
BATCH_SIZE = 8 # batch size
#
TRAIN_SHARE = 100 # size of train set
VAL_SHARE = 50# size of val set
TEST_SHARE = 50# size of test  set
#
MAX_EPOCHS = 150 # number of epochs
LEARNING_RATE = 5e-3 # learning rate
#
SCHEDULER_STEP = 50 # scheduler step
SCHEDULER_GAMMA = 0.1



# use cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = UNet().to(device)
# optimizer
opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# loss
loss_fn = bce_loss()
# metric
metric = iou_pytorch()
# scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer=opt, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

# train
train_loss_values, val_loss_values, avg_score_values = train(model, opt, loss_fn, MAX_EPOCHS, BATCH_SIZE, data_tr, data_val, metric, lr_sched=scheduler)

