import torch
import torch.nn as nn
from utils import iou

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        exists_box = (target[..., 4:5] > 0).float()
        # Box Predictions and Targets
        box_predictions = predictions[..., 0:4] 
        box_targets = exists_box * target[..., 0:4]

        # Numerical Stability for sqrt
        box_predictions[..., 2:4] = torch.sqrt(box_predictions[..., 2:4].clamp(min=1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4].clamp(min=1e-6))

        # Box Loss
        box_loss = self.lambda_coord * torch.mean(
            (box_predictions[..., :2] - box_targets[..., :2]) ** 2 + 
            (box_predictions[..., 2:4] - box_targets[..., 2:4]) ** 2
        )

        # Object Loss
        box_predictions_conf = predictions[..., 4:5]
        object_loss = torch.mean(
            (exists_box * box_predictions_conf - exists_box * target[..., 4:5]) ** 2
        )

        # No Object Loss
        no_object_loss = torch.mean(
            (1 - exists_box) * (predictions[..., 4:5] - target[..., 4:5]) ** 2
        )

        # Class Loss
        class_loss = torch.mean(
            (exists_box * predictions[..., 5:5 + self.C] - exists_box * target[..., 5:5 + self.C]) ** 2
        )

        # Total Loss
        loss = box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss
        return loss
    
