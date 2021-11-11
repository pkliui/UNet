import torch
import torch.nn as nn

def iou_pytorch(prediction: torch.Tensor, label: torch.Tensor):
    """
    IOU metrics taken from
    https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    #
    # Explanation:
    # We would like to compute IOU of two binary masks (predictions and labels) to assess the prediction quality
    # Convention(*): only predictions whose IOU > 0.5 are  "successful", the rest are ignored whilst computing the score
    #
    # The labels are already binary (zeros and ones)
    # The predictions are thresholded outputs of our model
    # We set a threshold for predictions such that the outputs' pixel values  > 0.5 are set to 1 and <=0.5 are set to 0
    #
    # Because of our convention(*), we must map IOU values such that IOU_new =  2 * IOU - 1.
    # This way, IOU = 0.5 will result in IOU_new = 0 (our convention(*))
    #
    # The IOU values are floats and can be anything between 0 and 1 (e.g. 0.5509)
    # So we round them to the nearest first decimal by multiplying by 10 (that's why we have 20 instead of 2)
    # applying ceil() and the n dividing by 10
    #
    # To avoid negative IOU_new (if IOU<0.5), we clamp to 0..10
    #
    ---
    Input
    ---
    prediction: torch.Tensor
        binary model output (thresholded at 0.5 such that pixels >0.5 are 1 and the rest are 0)
    label: torch.Tensor
        binary ground truth image
    ---
    Return
    ---
    iou_thresholded: float
        thresholded IOU metric at 0.5 and ceil-rounded to the nearest highest 1 decimal
    """
    # squeeze output (remove dimension 1)
    # use byte() to convert loats to bytes as required by AND (&) function
    prediction = prediction.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    label = label.squeeze(1).byte()
    #
    #compute intersection and union
    intersection = (prediction & label).float().sum((1, 2))  # zero if label=0 or prediction=0
    union = (prediction | label).float().sum((1, 2))  # zero if label and prediction are 0
    #
    # compute IOU
    EPS = 1e-8  # add a small number to avoid division by 0 or 0/0
    iou = (intersection + EPS) / (union + EPS)

    iou_thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return iou_thresholded



def bce_loss():
    """
    Binary closs entropy with logits
    :return:
    """
    return nn.BCEWithLogitsLoss