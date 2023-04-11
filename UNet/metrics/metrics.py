import torch

def iou_tgs_challenge(outputs: torch.Tensor, labels: torch.Tensor):
    """
    IOU metrics taken from
    https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    :param outputs: A tensor of shape [batch_size, 1, height, width] Output from network
    :param labels: A tensor of shape [batch_size, 1, height, width] Ground truth
    :return: A tensor of shape [batch_size] with IOU score for each image
    """
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float()
    print("intersection.shape", intersection.shape)
    # sum
    intersection = intersection.sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  #


def iou(outputs: torch.Tensor, labels: torch.Tensor):
    """
    IOU metrics without mapping, modified from
    https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    """
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou

# BCE loss
bce_loss = torch.nn.BCEWithLogitsLoss()


def dice_coefficient(iou: torch.Tensor):
    """
    Computes the dice coefficient
    :param iou: Intersection over union
    :return: Dice coefficient
    """
    return 2 * iou / (iou + 1)

