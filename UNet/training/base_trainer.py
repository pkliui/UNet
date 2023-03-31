import matplotlib.pyplot as plt
import torch
from UNet.utils import augment


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
                 device=None):
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
        device:
            Device to use for training
        ---
        Return
        ---
        None
        """
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        #
        # load training data - this is a mandatory step for training
        # unless we load validation data only
        if hasattr(data_loader, "train_loader"):
            self.train_loader = data_loader.train_loader
        #
        # load validation data if any
        if hasattr(data_loader, "val_loader"):
            self.val_loader = data_loader.val_loader
            self.validate = True
        else:
            self.validate = False
        #
        self.epochs = epochs
        self.xbatch = None
        self.ybatch = None
        # batches of training images and masks
        #
        self.lr_sched = lr_sched
        self.device = device

    def train(self):
        """
        Training pipeline
        """
        #
        #initialization of the loss and score values
        print("#initialization of the loss and score values")
        train_loss_values = []
        val_loss_values = []
        score_values = []
        #
        # #iterate through epochs
        print("# #iterate through epochs")
        for epoch in range(self.epochs):
            print('* Epoch %d/%d' % (epoch + 1, self.epochs))
            #
            # make a train step
            print("# make a train step")
            avg_train_loss = self.training_step()
            print("avg_train_loss ", avg_train_loss)
            if self.device is not None:
                # if we use GPU we need to detach the loss value from the computational graph
                # and move it to the CPU memory to be able to plot it later
                avg_train_loss = avg_train_loss.detach().cpu().numpy()
            train_loss_values.append(avg_train_loss)
            #
            # make a validation step
            if self.validate:
                avg_val_loss, avg_score = self.validation_step()
                if self.device is not None:
                    avg_val_loss = avg_val_loss.detach().cpu().numpy()
                    avg_score = avg_score.detach().cpu().numpy()
                val_loss_values.append(avg_val_loss)
                score_values.append(avg_score)
            #else:
            #    pass
            #
            # make a scheduler step if required
            if self.lr_sched is not None:
                self.lr_sched.step()
        return train_loss_values, val_loss_values, score_values

    def training_step(self):
        """
        Training step for one batch.
        ---
        Args:
        ---
        batch: batch of data
        ---
        Returns:
        ---
        loss: training loss value, sum for all batches
        """
        # initialization of loss values at the current epoch
        train_loss_all_batches = 0
        print("training step ... ")
        self.model.train()

        for _, batch in enumerate(self.train_loader):
            # get images and masks
            print("X_batch size  ", batch[0].size())
            #
            # reshape batches to the size (batch size, 3, width, height) for X and (batch size, 1, width, height) for Y
            X_batch, Y_batch = augment.reshape_batches(batch[0], batch[1])
            print("X_batch size after reshape ", X_batch.size())
            #
            # augment images together with masks
            augmented_xy_pairs = augment.AugmentImageAndMask(tensors=(X_batch, Y_batch),
                                                             transform=augment.random_transforms(prob=0.5))
            augmented_x, augmented_y = augment.get_augmented_tensors(*augmented_xy_pairs)
            self.xbatch = torch.stack(augmented_x, dim=0)
            self.ybatch = torch.stack(augmented_y, dim=0)
            #
            if self.device is not None:
                self.xbatch = self.xbatch.to(self.device)
                self.ybatch = self.ybatch.to(self.device)

            # set parameter gradients to zero
            self.optimizer.zero_grad()
            #
            # forward propagation
            Y_pred = self.model(X_batch)
            loss = self.criterion(Y_pred, Y_batch)
            #
            # backward propagation
            loss.backward()
            # update weights
            self.optimizer.step()
            #
            # add loss for the current batch to the total loss
            train_loss_all_batches += loss
        # calculate an average loss across all the batches to show the user
        avg_train_loss = train_loss_all_batches / len(self.train_loader)
        print("avg train loss", avg_train_loss)

        return avg_train_loss

    def validation_step(self):
        """
        Validation step for one batch.
        """
        # set dropout and batch normalization layers to evaluation mode before running the inference

        with torch.no_grad():
            self.model.eval()
            # initialization of loss values at the current epoch
            val_loss_all_batches = 0
            score = 0
            #
            for _, sample_val in enumerate(self.val_loader):
                # reshape batches to have the size of
                # (batch size, 3, width, height) for X and (batch size, 1, width, height) for Y
                X_val, Y_val = augment.reshape_batches(sample_val[0], sample_val[1])
                #print("X val res", X_val.shape)
                #print("Y val res", Y_val.shape)
                #if self.device is not None:
                #    X_val = X_val.to(self.device)
                    #Y_val = Y_val.to(self.device)
                # get logits for val set
                #Y_hat = self.model(X_val)
                Y_hat =  self.model(X_val.to(self.device)).detach().to('cpu')

                #plt.imshow(Y_hat[0][0, :, :])
                #plt.title("Y_hat")
                #plt.colorbar()
                #plt.show()


                #if self.device is not None:
                #    Y_hat = Y_hat.detach().to('cpu')
                    #Y_val = Y_val.detach().to('cpu')

                Y_hat_2plot = torch.round(torch.sigmoid(Y_hat))

                #plt.imshow(Y_hat_2plot[0][0, :, :])
                #plt.title("Y_hat_2plot")
                #plt.colorbar()
                #plt.show()
                #
                # compute val loss and append it
                loss = self.criterion(Y_hat, Y_val)
                # add loss for the current batch to the total loss
                val_loss_all_batches += loss
                #
                # compute score for the current batch
                # score += metric(Y_hat_2plot.to(device), Y_val.to(device)).mean().item()
                # temporarily replace by metric().mean without .item() because this leads to float has no attribute detach error
                # see https://github.com/horovod/horovod/issues/852

                #if self.device is not None:
                #score += self.metric(Y_hat, Y_val).mean()
                print("Y_hat_2plot ", Y_hat_2plot)
                print("Y_val ", Y_val)
                print("Y_hat_2plot shape ", Y_hat_2plot.shape)
                print("Y_val shape ", Y_val.shape)
                print("self.device ", self.device)
                print("Y_val shape ", Y_val.shape)
                score += self.metric(Y_hat_2plot.to(self.device), Y_val.to(self.device)).mean()
                print("score ", score)

            #score += self.metric(Y_hat_2plot.to(self.device), Y_val.to(self.device)).mean()
            #else:
                #    score += self.metric(Y_hat_2plot, Y_val).mean()
                print("score", score)
            #
            # calculate an average loss across all the batches to show the user
            avg_val_loss = val_loss_all_batches / len(self.val_loader)
            #avg_val_loss = avg_val_loss.detach().cpu().numpy()
            print("avg val loss", avg_val_loss)
            #
            # compute and append average score at current epoch
            avg_score = score / len(self.val_loader)
            #avg_score = avg_score.detach().cpu().numpy()
            #if self.device is not None:
            #    avg_score = avg_score.detach().cpu().numpy()

            print('avg_score: %f' % avg_score)

            return avg_val_loss, avg_score