import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from torch.optim import Adam


def iou(boxes_preds, boxes_lables):
    # convert x,y,w,h to x1,y1,x2,y2
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_lables[..., 0:1] - boxes_lables[..., 2:3] / 2
    box2_y1 = boxes_lables[..., 1:2] - boxes_lables[..., 3:4] / 2
    box2_x2 = boxes_lables[..., 0:1] + boxes_lables[..., 2:3] / 2
    box2_y2 = boxes_lables[..., 1:2] + boxes_lables[..., 3:4] / 2
    # find the intersection area
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    # calculate the area of intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    # calculate the area of both boxes
    box1_area = abs(box1_x2 - box1_x1) * abs(box1_y2 - box1_y1)
    box2_area = abs(box2_x2 - box2_x1) * abs(box2_y2 - box2_y1)
    # calculate IOU
    IOU = intersection / (box1_area + box2_area - intersection + 1e-6)
    return IOU


def non_max_suppression(bboxes, iou_threshold, confident):
    # Ensure bboxes is a list
    assert type(bboxes) == list
    # Filter boxes with confidence greater than the threshold
    bboxes_sort = [
        box for box in bboxes if len(box) > 4 and box[4] > confident
    ]  # Ensure each box has at least 5 elements (x, y, w, h, confidence)

    bboxes_sort = sorted(
        bboxes_sort, key=lambda x: x[4], reverse=True
    )  # Sort by confidence
    bboxes_after_nms = []

    while bboxes_sort:
        chosen_box = bboxes_sort.pop(0)

        # Remove boxes with high IOU overlap
        bboxes_sort = [
            box
            for box in bboxes_sort
            if box[5] != chosen_box[5]
            or iou(torch.tensor(chosen_box[0:4]), torch.tensor(box[0:4]))
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def get_bboxes(loader, model, iou_threshold, confidence, device="cuda"):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()  # set model to evaluation mode
    train_idx = 0

    for _, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]  # format of x is [batch_size, channels, height,width]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx], iou_threshold=iou_threshold, confident=confidence
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append(
                    [train_idx] + nms_box
                )  # size of nms_box is [train_idx, x,y,w,h,confidence,class_pred]

            for box in true_bboxes[idx]:
                if box[4] > 0:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()  # set model back to training mode
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    predictions = predictions.to("cuda")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 30)

    # Bbox predictions
    bboxes_1 = predictions[..., 0:4]  # x, y, w, h
    bboxes_2 = predictions[..., 5:9]  # x, y, w, h

    # Confidence scores for both boxes
    scores = torch.cat(
        (predictions[..., 4:5], predictions[..., 9:10]), dim=-1
    )  # Shape: [batch_size, S, S, 2]

    # Select the best box based on confidence
    best_box = scores.argmax(-1).unsqueeze(-1)  # Shape: [batch_size, S, S, 1]
    best_boxes = bboxes_1 * (1 - best_box) + best_box * bboxes_2  # Choose best box

    # Generate cell indices for x and y adjustment
    cell_indices = (
        torch.arange(S, device=predictions.device)
        .repeat(batch_size, S, 1)
        .unsqueeze(-1)
    )

    # Convert to (x, y, w, h) in YOLO format
    x = (1 / S) * (best_boxes[..., 0:1] + cell_indices)  # Adjust x
    y = (1 / S) * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))  # Adjust y
    w_h = (1 / S) * best_boxes[..., 2:4]  # Adjust width and height

    # Concatenate to form bounding box predictions
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)

    # Get class predictions
    predicted_class = (
        predictions[..., 10:30].argmax(-1).unsqueeze(-1)
    )  # Shape: [batch_size, S, S, 1]

    # Get the best confidence score
    best_confidence = torch.max(predictions[..., 4], predictions[..., 9]).unsqueeze(
        -1
    )  # Shape: [batch_size, S, S, 1]

    # Concatenate to final format: [batch_size, S, S, 6]
    converted_preds = torch.cat(
        (converted_bboxes, best_confidence, predicted_class), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, 6)
    converted_pred[..., 5] = converted_pred[..., 5].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []  # store all bboxes for each example
        for bbox_idx in range(S * S):
            bboxes.append(
                [x.item() for x in converted_pred[ex_idx, bbox_idx, :]]
            )  # **note bboxe's size is [49,6]
        all_bboxes.append(bboxes)  # size is [batch_size,49,6]

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def mean_average_precision(pred_boxes, true_boxes, iou_threshold, num_class):
    average_precisions = []
    epsilon = 1e-6
    for c in range(num_class):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if (
                detection[6] == c
            ):  # size of detection is [train_idx,x,y,w,h,confidence,class_pred]
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[6] == c:
                ground_truths.append(true_box)

        amount_bboxes = {
            key: torch.zeros(val)
            for key, val in Counter([gt[0] for gt in ground_truths]).items()
        }  # create tensor with zeros for each bbox in each image

        detections.sort(
            key=lambda x: x[5], reverse=True
        )  # sort the detections by confidence in descending order
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]  # get all bboxes in the same image

            # num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou_score = iou(torch.tensor(detection[1:5]), torch.tensor(gt[1:5]))

                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = idx

                if best_iou > iou_threshold:
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][
                            best_gt_idx
                        ] = 1  # mark as already detected
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)  # calculate the cumulative sum of TP
        FP_cumsum = torch.cumsum(FP, dim=0)
        precisions = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat(
            (torch.tensor([1]), precisions)
        )  # add 1 to the beginning of precisions
        recalls = torch.div(TP_cumsum, (total_true_bboxes + epsilon))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(
            torch.trapz(precisions, recalls)
        )  # calculate the area under the curve
    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    im = np.array(image)
    height, width, _ = im.shape
    _, ax = plt.subplots(1)
    ax.imshow(im)
    for box in boxes:
        x, y, w, h, _, class_pred = box
        x = x * width
        y = y * height
        w = w * width
        h = h * height
        color = (0, 255, 0)
        rect = patches.Rectangle(
            (x - w / 2, y - h / 2), w, h, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        plt.text(x - w / 2, y - h / 2, s=class_pred, color="white")
    plt.show()


def test_iou():
    box1 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    box2 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    assert torch.isclose(iou(box1, box2), torch.tensor([1.0]))


def test_non_max_suppression():
    bboxes = [
        [0.5, 0.5, 1.0, 1.0, 0.9, 0],
        [0.5, 0.5, 1.0, 1.0, 0.8, 0],
        [0.5, 0.5, 1.0, 1.0, 0.7, 1],
    ]
    result = non_max_suppression(bboxes, iou_threshold=0.5, confident=0.5)
    assert len(result) == 2


def test_convert_cellboxes():
    predictions = torch.randn((2, 7, 7, 30))
    result = convert_cellboxes(predictions)
    assert result.shape == (2, 7, 7, 6)


def test_cellboxes_to_boxes():
    predictions = torch.randn((2, 7, 7, 30))
    result = cellboxes_to_boxes(predictions)
    assert len(result) == 2
    assert len(result[0]) == 49


def test_mean_average_precision():
    pred_boxes = [[0, 0.5, 0.5, 1.0, 1.0, 0.9, 0], [0, 0.5, 0.5, 1.0, 1.0, 0.8, 0]]
    true_boxes = [[0, 0.5, 0.5, 1.0, 1.0, 1.0, 0]]
    result = mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, num_class=1
    )
    assert result >= 0 and result <= 1


def test_save_load_checkpoint():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    state = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(state, filename="test_checkpoint.pth.tar")
    load_checkpoint(torch.load("test_checkpoint.pth.tar"), model, optimizer)


if __name__ == "__main__":
    test_iou()
    test_non_max_suppression()
    test_convert_cellboxes()
    test_cellboxes_to_boxes()
    test_mean_average_precision()
    test_save_load_checkpoint()
    print("All tests passed!")
