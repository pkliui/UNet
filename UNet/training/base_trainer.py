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
                 optimizer=None,
                 data_loader=None,
                 epochs=None,
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
        save_dir: str
            Directory to save tensorboard logs and models
        save_model_frequency: int
            Frequency (in epochs) to save the model
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if data_loader is not None:
            self.train_loader = data_loader.train_loader
            #self.n_train_batch = len(data_loader.train_loader)
            #self.n_val_batch = len(data_loader.val_loader)
        #
        # if data loader has val_loader attribute, validate
        if hasattr(data_loader, "val_loader"):
            self.val_loader = data_loader.val_loader
            self.validate = True
        #
        self.epochs = epochs
        self.xbatch = None
        self.ybatch = None
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
        # for epoch in range(self.epochs):
        #     #
        #     # epochs counter
        #     ii = 0
        #     print('* Epoch %d/%d' % (epoch + 1, self.epochs))
        #     #
        #     # initialization of average loss values at the current epoch
        #     avg_loss = 0
        #     val_avg_loss = 0
        #     #
        #     ###############
        #     # train model
        #     ###############
        #     #self.model.train()
        #     #
        #     for _, sample in enumerate(self.train_loader):
        #         X_batch = sample["image"]
        #         Y_batch = sample["mask"]
        #         #
        #         # augment images
        #         print("X batch", X_batch.shape)
        #         print("Y batch", Y_batch.shape)
        #         #
        #         # reshape batches to have
        #         # (batch size, 3, width, height) for X and (batch size, 1, width, height) for Y
        #         X_batch, Y_batch = augment.reshape_batches(X_batch, Y_batch)
        #
        #         # augment images
        #         print("X batch res", X_batch.shape)
        #         print("Y batch res", Y_batch.shape)
        #
        #         augmented_xy_pairs = augment.AugmentImageAndMask(tensors=(X_batch, Y_batch),
        #                                                  transform=augment.random_transforms(prob=0.5))
        #         #
        #         # note:  the star * in *augmented_xy_pairs is needed to unpack the tuple of (x,y) tuples
        #         # into individual (x,y) tuples
        #         augmented_x, augmented_y = augment.get_augmented_tensors(*augmented_xy_pairs)
        #         X_batch = torch.stack(augmented_x, dim=0)
        #         Y_batch = torch.stack(augmented_y, dim=0)
        #
        #         self.xbatch = X_batch
        #         self.ybatch = Y_batch
        #
        # #return  X_batch
        # #self.ybatch = Y_batch

