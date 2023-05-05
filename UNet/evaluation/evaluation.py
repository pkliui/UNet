"""
This module contains functions for evaluation of the model.
"""
import warnings

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from UNet.utils.augment import reshape_batches


def evaluate_model(model, data_loader, device, metric):
    """
    Evaluate the model performance on the given test dataset.
    :param model: Model to evaluate
    :param data_loader: Data loader to use for evaluation
    :param device: Device to use for evaluation
    :param metric: Metric to use for evaluation
    :return:
    """
    model.eval()
    with torch.no_grad():
        num_batches = len(data_loader)
        score = 0
        conf_matrix = np.zeros((2, 2), dtype=np.int32)
        conf_matrices = []

        # Iterate through the dataset
        for i, batch in enumerate(data_loader):
            images, masks = batch["image"], batch["mask"]
            images = images.to(device)
            masks = masks.to(device)

            # reshape batches to the size (batch size, 3, width, height) for X and (batch size, 1, width, height) for Y
            images, masks = reshape_batches(images, masks)

            # Forward pass
            predictions = model(images.float())

            #
            # compute sigmoid and round to the nearest integer (0,1)
            # to be able to compare with the binary ground truth images
            predictions = torch.round(torch.sigmoid(predictions))

            # Compute the metric
            score += metric(predictions, masks)

            # Convert masks and masks to numpy arrays
            predictions_np = predictions.cpu().detach().numpy()
            masks_np = masks.cpu().detach().numpy()

            # Flatten predicted and true labels for each image in batch
            #predictions_np_flat = predictions_np.reshape(num_batches, predictions_np.shape[-2]*predictions_np.shape[-1])
            #masks_np_flat = masks_np.reshape(num_batches, masks_np.shape[-2]*masks_np.shape[-1])

            predictions_np_flat = predictions_np.flatten()
            masks_np_flat = masks_np.flatten()

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(masks_np_flat, predictions_np_flat)
            print("conf matrix ", conf_matrix)
            conf_matrices.append(conf_matrix)

        avg_conf_matrix = np.mean(conf_matrices, axis=0)
        TN, FP, FN, TP = avg_conf_matrix.ravel()

        print("TP ", TP)
        print("FP ", FP)
        print("FN ", FN)
        print("TN ", TN)

        # Compute accuracy, precision, recall and F1 score
        if TP + FP + FN + TN != 0:
            accuracy = (TP + TN) / (TP + FP + FN + TN)
        else:
            accuracy = 0
            print("TP + FP + FN + TN = 0, setting accuracy to 0")
        print("accuracy ", accuracy)
        if TP + FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
            print("TP + FP = 0, setting precision to 0")
        print("precision ", precision)
        if TP + FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
            print("TP + FN = 0, setting recall to 0")
        print("recall ", recall)
        if precision + recall != 0:
            F1_score = 2 * (precision * recall) / (precision + recall)
        else:
            F1_score = 0
            print("precision + recall = 0, setting F1_score to 0")
        print("F1 ", F1_score)

        # Compute the average dice score
        score /= num_batches
        print("Score: ", score)

    return score, accuracy, precision, recall, F1_score, avg_conf_matrix
