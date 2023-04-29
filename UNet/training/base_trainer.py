import matplotlib.pyplot as plt
import torch
import os
from UNet.utils import augment
from torch.utils.tensorboard import SummaryWriter


"""
This module contains a base implementation for a trainer.
It defines a training_step method, a validation_step method
and a train method which consists of the main training loop.
"""

class BaseTrainer:
    def __init__(self,
                 model=None,
                 loss_function=None,
                 metric=None,
                 optimizer=None,
                 data_loader=None,
                 n_epochs=None,
                 lr_sched=None,
                 device=None,
                 early_stop_save_dir=None,
                 early_stop_patience=None,
                 save_dir=None):
        """
        Initializes BaseTrainer class
        ---
        Parameters
        ---
        model:
            the model to train
        loss_function:
            the loss function to optimize
        metric:
            the metric to use for validation
        optimizer:
            the optimizer to use for training
        lr_sched:
            Learning rate scheduler
            Default: None
        data_loader:
            SplitDataLoader that provides training and validation loaders
        n_epochs: int
            Number of epochs for training
        device:
            Device to use for training
            Default: None
        early_stop_save_dir:
            Path to save the model when early stopping is triggered
            Default: None
        early_stop_patience:
            Early stopping patience
            Default: 5
        save_dir:
            Path to save the model in case early stopping is not triggered
            Default: None
        ---
        Return
        ---
        None
        """
        self.model = model
        self.loss_function = loss_function
        self.metric = metric
        self.optimizer = optimizer
        self.lr_sched = lr_sched
        self.data_loader = data_loader
        self.n_epochs = n_epochs
        self.device = device
        self.early_stop_save_dir = early_stop_save_dir
        self.early_stop_patience = early_stop_patience
        if save_dir is not None and os.path.exists(save_dir):
            self.save_dir = save_dir
        else:
            raise ValueError("save_dir must be specified and exist")
        #
        # other attributes derived from the input parameters
        if hasattr(data_loader, "train_loader"):
            self.train_loader = data_loader.train_loader
        else:
            self.train_loader = None
        """ Training data loader with the length equal to the number of batches """
        if hasattr(data_loader, "val_loader"):
            self.val_loader = data_loader.val_loader
        else:
            self.val_loader = None
        """ Validation data loader with the length equal to the number of batches """
        #
        self.validate = self.val_loader is not None
        """ Flag to indicate whether validation is performed """
        #
        self.n_train_batch = len(self.train_loader) if self.train_loader is not None else 0
        """ Number of batches in the training set"""
        self.n_val_batch = len(self.val_loader) if self.validate else 0
        """ Number of batches in the validation set"""
        #
        self.image_batch = None
        """ Batch of training images """
        self.mask_batch = None
        """ Batch of training masks """
        #
        self.patience = 0
        """ Patience counter for early stopping """
        self.best_val_loss = float('inf')
        """ Best validation loss for early stopping """
        self.writer = SummaryWriter(os.path.join(save_dir, 'summaries'))

    def train(self):
        """
        Training pipeline
        """
        # initialize lists to store training and validation loss values for each epoch
        train_loss_values = []
        val_loss_values = []
        score_values = []
        #
        # #iterate through epochs
        for epoch in range(1, self.n_epochs+1):
            print('* Epoch %d/%d' % (epoch , self.n_epochs))
            #
            # initialization of loss values at the current epoch
            accum_loss = 0
            #
            for batch_index, batch in enumerate(self.train_loader):
                # augment the batch
                images_batch, masks_batch = augment.augment_batch(batch['image'], batch['mask'], self.device, prob=0.5)
                # train the model on the current batch and accumulate loss over the batches
                current_loss = self.training_step(images_batch, masks_batch)
                accum_loss += current_loss
            #
            # calculate an average train loss across all the batches
            average_loss = accum_loss / len(self.train_loader)
            if self.device is not None:
                average_loss = average_loss.detach().cpu().numpy()
                #average_loss = average_loss.cpu().numpy()
            train_loss_values.append(average_loss)
            #
            # add loss to writer for tensorboard visualization
            self.writer.add_scalar('Loss/train', average_loss, epoch)
            self.writer.flush()
#           #
            #
            # make a validation step
            if self.validate:
                avg_val_loss, avg_score = self.validation_step()
                if self.device is not None:
                    avg_val_loss = avg_val_loss.detach().cpu().numpy()
                    avg_score = avg_score.detach().cpu().numpy()
                val_loss_values.append(avg_val_loss)
                score_values.append(avg_score)
                #
                # add loss to writer for tensorboard visualization
                self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
                self.writer.flush()
                # add score to writer for tensorboard visualization
                self.writer.add_scalar('Score/val', avg_score, epoch)
                self.writer.flush()
                #
                # Early stopping
                if self.early_stop_save_dir is not None:
                    # if the validation loss is better than the best one, save the model
                    if avg_val_loss < self.best_val_loss:
                        # set the best validation loss to the current one
                        self.best_val_loss = avg_val_loss
                        # reset the patience counter
                        self.patience = 0
                        # save the model
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.early_stop_save_dir,
                                                'model_state_dict.pth'))
                    # if the validation loss is not better than the best one, increase the patience counter
                    else:
                        self.patience += 1
                    # if the patience counter is equal to the early stopping patience, stop the training
                    if self.patience >= self.early_stop_patience:
                        print(f"Validation loss did not improve for {self.early_stop_patience} epochs. "
                              f"Training stopped early.")
                        break
    #
            # make a scheduler step if required
            if self.lr_sched is not None:
                self.lr_sched.step()
        # load the best model if early stopping is triggered
        #
        #if self.early_stop_save_dir is not None:
        #    self.model.load_state_dict(torch.load(os.path.join(self.early_stop_save_dir,
        #                                                       'model_state_dict.pth')))
        return train_loss_values, val_loss_values, score_values

    def training_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        Training step for one batch
        :param inputs: batch of inputs, shape: (batch_size, channels, height, width)
        :param labels: batch of labels, shape: (batch_size, channels, height, width)
        :return: loss value, shape: (1)
        """
        #
        # set dropout and batch normalization layers to training mode
        self.model.train()
        # zero gradients as PyTorch accumulates them
        self.optimizer.zero_grad()
        #
        # forward propagation
        predictions = self.model(inputs)
        #
        # calculate loss and update weights
        loss = self.loss_function(predictions, labels)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self):
        """
        Validation step for one epoch
        """
        # set dropout and batch normalization layers to evaluation mode before running the inference
        self.model.eval()
        with torch.no_grad():
            #
            # initialization of loss values at the current epoch
            accum_loss = 0
            score = 0
            for batch in self.val_loader:
                # augment the batch
                inputs, labels = batch['image'], batch['mask']
                inputs, labels = augment.reshape_batches(inputs, labels)
                #
                # forward propagation
                predictions = self.model(inputs)
                predictions_2plot = torch.round(torch.sigmoid(predictions))
                loss = self.loss_function(predictions, labels)
                accum_loss += loss
                #
                # calculate score
                # temporarily replace by metric().mean without .item()
                # because this leads to float has no attribute detach error
                # see https://github.com/horovod/horovod/issues/852
                score += self.metric(predictions_2plot.to(self.device), predictions.to(self.device)).mean()
            #
            # calculate an average loss and score across all the batches to show the user
            average_loss = accum_loss / len(self.val_loader)
            average_score = score / len(self.val_loader)
            return average_loss, average_score
