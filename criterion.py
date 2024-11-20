import torch 
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, S, B, C): 
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S 
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # Resize the format
        batch = predictions.size(0)
        predictions = predictions.view(batch, self.S, self.S, self.C + self.B * 5)

        # Mask: To find the grid which have obj and no obj. If value > 0 => exists obj
        obj_mask = target[..., 4] > 0
        noobj_mask = target[..., 4] == 0

        # Localization loss
        pred_box = predictions[obj_mask].view(-1, 5)
        target_box = target[obj_mask].view(-1, 5)

        loss_xy = torch.sum((pred_box[..., :2] - target_box[..., :2]) ** 2)
        loss_wh = torch.sum((torch.sqrt(pred_box[..., 2:4]) - torch.sqrt(target_box[..., 2:4])) ** 2)
        loc_loss = self.lambda_coord * (loss_xy + loss_wh)
        
        # Confidence loss for objects
        pred_conf = predictions[obj_mask][..., 4]
        target_conf = target[obj_mask][..., 4]
        conf_loss_obj = torch.sum((pred_conf - target_conf) ** 2)

        # Confidence loss for no objects
        pred_conf_noobj = predictions[noobj_mask][..., 4]
        target_conf_noobj = target[noobj_mask][..., 4]
        conf_loss_noobj = torch.sum((pred_conf_noobj - target_conf_noobj) ** 2)

        # Classification loss
        pred_class = predictions[obj_mask][..., 5:]
        target_class = target[obj_mask][..., 5:]
        class_loss = torch.sum((pred_class - target_class) ** 2)

        # Total loss
        total_loss = loc_loss + conf_loss_obj + self.lambda_noobj * conf_loss_noobj + class_loss

        return total_loss