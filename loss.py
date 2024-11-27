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
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        # Calculate IoUs
        iou_b1 = iou(predictions[..., 0:4], target[..., 0:4])
        iou_b2 = iou(predictions[..., 5:9], target[..., 0:4])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)  # Best box is 1 or 0
        exists_box = target[..., 20].unsqueeze(3)

        # Box Predictions and Targets
        box_predictions = torch.where(
            bestbox.unsqueeze(-1) == 1,
            predictions[..., 5:9],
            predictions[..., 0:4],
        )
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
        box_predictions_conf = torch.where(
            bestbox.unsqueeze(-1) == 1,
            predictions[..., 9:10],
            predictions[..., 4:5],
        )
        object_loss = torch.mean(
            (exists_box * box_predictions_conf - exists_box * target[..., 4:5]) ** 2
        )

        # No Object Loss
        no_object_loss = torch.mean(
            (1 - exists_box) * (predictions[..., [4, 9]] - target[..., 4:5].unsqueeze(-1)) ** 2
        )

        # Class Loss
        class_loss = torch.mean(
            (exists_box * predictions[..., 10:10 + self.C] - exists_box * target[..., 10:10 + self.C]) ** 2
        )

        # Total Loss
        loss = box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss
        return loss