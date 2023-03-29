import torch
from IPython.core.display import clear_output

from UNet.utils import augment
from UNet.models.unet import UNet
from UNet.metrics import bce_loss
from matplotlib import pyplot as plt


"""
This module contains a base implementation for a trainer.
It defines a training_step method, a validation_step method
and a train method which consists of the main training loop.
"""

class BaseTrainer:
    def __init__(self,
                 model=None,
                 criterion=None,
                 metric=None,
                 optimizer=None,
                 data_loader=None,
                 epochs=None,
                 lr_sched=None,
                 device=None,
                 save_dir=None,
                 save_model_frequency=None):
        """
        Initializes BaseTrainer class
        ---
        Parameters
        ---
        model:
            the model to train
        criterion:
            the loss function to optimize
        optimizer:
            the optimizer to use for training
        data_loader:
            SplitDataLoader that provides training and validation loaders
        epochs: int
            Number of epochs for training
        lr_sched:
            Learning rate scheduler
            Default: None
        save_dir: str
            Directory to save tensorboard logs and models
        save_model_frequency: int
            Frequency (in epochs) to save the model
        """
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.train_loader = data_loader.train_loader
            #self.n_train_batch = len(data_loader.train_loader)
            #self.n_val_batch = len(data_loader.val_loader)
        #
        # load validation data if any
        if hasattr(data_loader, "val_loader"):
            self.val_loader = data_loader.val_loader
            self.validate = True
        #
        self.epochs = epochs
        #
        self.xbatch = None
        self.ybatch = None
        # batches of training images and masks
        #
        self.lr_sched = lr_sched
        self.device = device
        #self.save_dir = save_dir
        #self.writer = SummaryWriter(os.path.join(save_dir, "summaries"))
        #self.save_model_frequency = save_model_frequency

    def train(self):
        """
        Training pipeline
        """
        #
        #initialization of the loss and score values
        train_loss_values = []
        val_loss_values = []
        avg_score_values = []
        #
        # #iterate through epochs
        for epoch in range(self.epochs):
             #
             # epochs counter
            ii = 0
            print('* Epoch %d/%d' % (epoch + 1, self.epochs))
             #
            # initialization of average loss values at the current epoch
            avg_loss = 0
            val_avg_loss = 0
            #
        #     ###############
        #     # train model
        #     ###############
            self.model.train()
        #     #
            for _, sample in enumerate(self.train_loader):
                X_batch = sample[0]
                Y_batch = sample[1]

        #         # augment images
                print("X batch", X_batch.shape)
                print("Y batch", Y_batch.shape)
                print(Y_batch[0].unique())
                #plt.imshow(Y_batch[0])
                #plt.show()
        #         #
        #         # reshape batches to have the size of
                  # (batch size, 3, width, height) for X and (batch size, 1, width, height) for Y
                X_batch, Y_batch = augment.reshape_batches(X_batch, Y_batch)
        #
        #         # augment images
                print("X batch res", X_batch.shape)
                print("Y batch res", Y_batch.shape)
        #
                augmented_xy_pairs = augment.AugmentImageAndMask(tensors=(X_batch, Y_batch),
                                                          transform=augment.random_transforms(prob=0.5))
        #         #
        #         # note:  the star * in *augmented_xy_pairs is needed to unpack the tuple of (x,y) tuples
        #         # into individual (x,y) tuples
                augmented_x, augmented_y = augment.get_augmented_tensors(*augmented_xy_pairs)
                X_batch = torch.stack(augmented_x, dim=0)
                Y_batch = torch.stack(augmented_y, dim=0)
                # send to device
                if self.device is not None:
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                self.xbatch = X_batch
                self.ybatch = Y_batch
                print("augmented_x ", X_batch.shape)
                print("augmented_y ", Y_batch.shape)

                 #print("batch ", ii, " out of ", len(data_tr) )
                 # epochs counter
                 #ii += 1
                 # print("X_batch.shape from data_tr",X_batch.shape)
                 # print("Y_batch.shape from data_tr",Y_batch.shape)
                 #

                 # data to device
                #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #X_batch = X_batch.to(device)
                 # print("X_batch.shape to device",X_batch.shape)
                #Y_batch = Y_batch.to(device)
                 # print("Y_batch.shape to device",Y_batch.shape)
             #
                # set parameter gradients to zero
                self.optimizer.zero_grad()
                 #
                 # forward propagation
                 #
                 # get logits
                #with torch.autograd.set_grad_enabled(True):
                Y_pred = self.model(X_batch)

                # print("Y_pred.shape",Y_pred.shape)
                # compute train loss
                #print("loss computing ...")
                #print("y batch ", Y_batch[0].unique())
                #print("y pred ", Y_pred[0].detach().unique())
                #
                #target = torch.ones([2, 3], dtype=torch.float32)  # 64 classes, batch size = 10
                #print("target", target)
                #output = torch.full([2, 3], -1.5)  # A prediction (logit)
                #print("output", output)
                #pos_weight = torch.ones([3])  # All weights are equal to 1
                #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                #criterion(output, target)  # -log(sigmoid(1.5))
                #tensor(0.2014)

                loss = self.criterion(Y_pred, Y_batch)  # forward-pass - BCEWithLogitsLoss (pred,prob)
                 #
                 # backward-pass
                 #
                    #loss.retain_grad()
                loss.backward()
                 # update weights
                self.optimizer.step()
                 #
                 # calculate loss to show the user
                avg_loss += loss
            avg_loss = avg_loss / len(self.train_loader)
            # append train loss
            if self.device is not None:
                train_loss_values.append(avg_loss.detach().cpu().numpy())
            train_loss_values.append(avg_loss)
            #
            print('loss: %f' % avg_loss)
            # validate model
            #
            with torch.no_grad():
                # set dropout and batch normalization layers to evaluation mode before running the inference
                self.model.eval()
                score = 0
                avg_score = 0
            #
                for _, sample_val in enumerate(self.val_loader):
                    #X_val = sample_val[0].to(device)
                    #Y_val= sample_val[1].to(device)
                    X_val = sample_val[0]
                    Y_val= sample_val[1]
                    #augment images
                    print("X val", X_val.shape)
                    print("Y val", Y_val.shape)
                    print(Y_val[0].unique())
                    #plt.imshow(Y_batch[0])
                    #plt.show()
                    #
                    # reshape batches to have the size of
                    # (batch size, 3, width, height) for X and (batch size, 1, width, height) for Y
                    X_val, Y_val = augment.reshape_batches(X_val, Y_val)
                    #
                    print("X val res", X_val.shape)
                    print("Y val res", Y_val.shape)
                    if self.device is not None:
                        X_val = X_val.to(self.device)
                        Y_val = Y_val.to(self.device)
                    # get logits for val set
                    Y_hat = self.model(X_val)
            #       #
            #       # only for plotting purposes: apply sigmoid and round to the nearest integer (0,1)
            #       # to obtain binary image
                #   #Y_hat_2plot = torch.round(torch.sigmoid(Y_hat.detach().to('cpu')))
                    if self.device is not None:
                        Y_hat = Y_hat.detach().to('cpu')
                        Y_val = Y_val.detach().to('cpu')
                    Y_hat_2plot = torch.round(torch.sigmoid(Y_hat))
            #         #
            #         # compute val loss and append it
                    val_loss = self.criterion(Y_hat, Y_val)
                    val_avg_loss += val_loss
            #         #
            #         # compute score for the current batch
            #         # score += metric(Y_hat_2plot.to(device), Y_val.to(device)).mean().item()
            #         # temporarily replace by metric().mean without .item() because this leads to float has no attribute detach error
            #         # see https://github.com/horovod/horovod/issues/852
                    score += self.metric(Y_hat_2plot, Y_val).mean()
            #        score += self.metric(Y_hat_2plot.to(device), Y_val.to(device)).mean()
            #     #
            #     # compute and append average val loss at current epoch
                val_avg_loss = val_avg_loss / len(self.val_loader)
                val_loss_values.append(val_avg_loss)
            #     #val_loss_values.append(val_avg_loss.detach().cpu().numpy())
            #     #
            #     # compute and append average score at current epoch
                avg_score = score / len(self.val_loader)
                avg_score_values.append(avg_score)

                print('val_avg_loss: %f' % val_avg_loss)
                #print('avg_score: %f' % avg_score)
             #     #avg_score_values.append(avg_score.detach().cpu().numpy())
            # #
            # #clear_output(wait=True)
            # # make a scheduler step if required
            # #if self.lr_sched != None:
            # #    self.lr_sched.step()

