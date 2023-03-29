import torch

from UNet.utils.augment import AugmentImageAndMask, get_augmented_tensors, random_transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from IPython.display import clear_output
from matplotlib import pyplot as plt

def train(model,
          opt,
          loss_fn,
          epochs,
          batch_size,
          data_tr,
          data_val,
          metric,
          lr_sched=None):
    """
    - trains the model
    - computes training loss, validation loss and validation score
    - plots the progress
    ---
    input
    ---
    model:
        pytorch  model to train
    opt:
        optimizer
    loss_fn :
        loss function
    epochs: int
        number of epochs
    data_tr: dataloader
        training data
    data_val: dataloader
        validation data
    metric:
        segmentation metric
    lr_sched:
        scheduler
        default: None
    ---
    return
    ---
    training loss, validation loss and validation score
    """
    # loss and score declaration
    train_loss_values = []
    val_loss_values = []
    avg_score_values = []
    #
    # iterate through epochs
    for epoch in range(epochs):
        #
        # epochs counter
        ii = 0
        print('* Epoch %d/%d' % (epoch + 1, epochs))
        #
        # average loss declaration
        avg_loss = 0
        val_avg_loss = 0
        #
        ##############
        # train model
        ##############
        #
        model.train()
        #

        for X_batch, Y_batch in data_tr:
            # augment images
            augmented_xy_pairs = AugmentImageAndMask(tensors=(X_batch, Y_batch), transform=random_transforms(prob=0.5))
            #
            # note:  the star * in *augmented_xy_pairs is needed to unpack the tuple of (x,y) tuples
            # into individual (x,y) tuples
            augmented_x, augmented_y = get_augmented_tensors(*augmented_xy_pairs)
            X_batch = torch.stack(augmented_x, dim=0)
            Y_batch = torch.stack(augmented_y, dim=0)
            # print("augmented_x ", X_batch.shape)
            # print("augmented_y ", Y_batch.shape)
            # print("batch ", ii, " out of ", len(data_tr) )
            # epochs counter
            ii += 1
            # print("X_batch.shape from data_tr",X_batch.shape)
            # print("Y_batch.shape from data_tr",Y_batch.shape)
            #
            # data to device
            X_batch = X_batch.to(device)
            # print("X_batch.shape to device",X_batch.shape)
            Y_batch = Y_batch.to(device)
            # print("Y_batch.shape to device",Y_batch.shape)
            #
            # set parameter gradients to zero
            opt.zero_grad()
            #
            # forward propagation
            #
            # get logits
            Y_pred = model(X_batch)
            # print("Y_pred.shape",Y_pred.shape)
            #
            # compute train loss
            loss = loss_fn(Y_pred, Y_batch)  # forward-pass - BCEWithLogitsLoss (pred,prob)
            #
            # backward-pass
            #
            loss.backward()
            # update weights
            opt.step()
            #
            # calculate loss to show the user
            avg_loss += loss
        avg_loss = avg_loss / len(data_tr)
        #
        print('loss: %f' % avg_loss)
        # append train loss
        train_loss_values.append(avg_loss.detach().cpu().numpy())

        #
        # validate model
        #
        with torch.no_grad():
            #
            # set dropout and batch normalization layers to evaluation mode before running inference
            model.eval()
            score = 0
            avg_score = 0
            #
            for X_val, Y_val in data_val:
                # get logits for val set
                Y_hat = model(X_val.to(device)).detach().to('cpu')
                #
                # only for plotting purposes: apply sigmoid and round to the nearest integer (0,1)
                # to obtain binary image
                Y_hat_2plot = torch.round(torch.sigmoid(Y_hat))
                # Y_hat = torch.round(torch.sigmoid(Y_hat))

                # print("Y hat shape", Y_hat.shape)
                #
                # compute val loss and append it
                val_loss = loss_fn(Y_hat, Y_val)
                val_avg_loss += val_loss
                #
                # compute score for the current batch
                # score += metric(Y_hat_2plot.to(device), Y_val.to(device)).mean().item()
                # temporarily replace by metric().mean without .item() because this leads to float has no attribute detach error
                # see https://github.com/horovod/horovod/issues/852
                score += metric(Y_hat_2plot.to(device), Y_val.to(device)).mean()
            #
            # compute and append average val loss at current epoch
            val_avg_loss = val_avg_loss / len(data_val)
            val_loss_values.append(val_avg_loss.detach().cpu().numpy())
            #
            # compute and append average score at current epoch
            avg_score = score / len(data_val)
            avg_score_values.append(avg_score.detach().cpu().numpy())

        clear_output(wait=True)

        # plotting
        num_images_to_plot = 5 * (batch_size > 5) + batch_size * (batch_size <= 5)
        rcParams['figure.figsize'] = (2 * num_images_to_plot, 2 * 4)
        #
        for k in range(num_images_to_plot):
            # subplot (height, width, absolute image position)
            plt.subplot(4, num_images_to_plot, k + 1)
            plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
            plt.title('Input image')
            plt.axis('off')

            plt.subplot(4, num_images_to_plot, k + num_images_to_plot + 1)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')

            plt.subplot(4, num_images_to_plot, k + num_images_to_plot * 2 + 1)
            plt.imshow(Y_hat_2plot[k, 0], cmap='gray')
            plt.title('Binary output')
            plt.axis('off')

            plt.subplot(4, num_images_to_plot, k + num_images_to_plot * 3 + 1)
            plt.imshow(Y_val[k, 0], cmap='gray')
            plt.title('Ground truth')
            plt.axis('off')

            plt.tight_layout()

        plt.suptitle('%d / %d - train loss: %f' % (epoch + 1, epochs, avg_loss))
        plt.suptitle('%d / %d - val. loss: %f' % (epoch + 1, epochs, val_avg_loss))
        plt.show()

        # CHANGES HERE
        # make a scheduler step if required
        if lr_sched != None:
            lr_sched.step()
        # CHANGES END

    plt.plot(train_loss_values)
    plt.plot(val_loss_values)
    plt.plot(avg_score_values)
    plt.legend(["train_loss", "val_loss", "val_score"], loc="lower right")
    plt.show

    return train_loss_values, val_loss_values, avg_score_values