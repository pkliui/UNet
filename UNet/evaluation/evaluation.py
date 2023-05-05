"""
This module contains functions for evaluation of the model.
"""

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

            # Compute the confusion matrix
            conf_matrix += confusion_matrix(masks_np.flatten(), predictions_np.flatten())

        # Compute TP, FP, TN, FN
        TP = np.diag(conf_matrix)
        FP = np.sum(conf_matrix, axis=0) - TP
        FN = np.sum(conf_matrix, axis=1) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        # Compute accuracy, precision, recall and F1 score
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 2 * (precision * recall) / (precision + recall)

        # Compute the average dice score
        score /= num_batches

        print("Confusion matrix: ")
        print(conf_matrix)
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", F1_score)
        print("Score: ", score)

    return score, accuracy, precision, recall, F1_score, conf_matrix
