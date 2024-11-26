import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def iou( boxes_preds,boxes_lables):
    #convert x,y,w,h to x1,y1,x2,y2
    box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3]/2
    box1_y1 = boxes_preds[...,1:2]- boxes_preds[...,3:4]/2
    box1_x2 = boxes_preds[...,0:1] - boxes_preds[...,2:3]/2
    box1_y2 = boxes_preds[...,1:2] - boxes_preds[...,3:4]/2
    box2_x1 = boxes_lables[...,0:1] - boxes_lables[...,2:3]/2
    box2_y1 = boxes_lables[...,1:2] - boxes_lables[...,3:4]/2
    box2_x2 = boxes_lables[...,0:1] + boxes_lables[...,2:3]/2
    box2_y2 = boxes_lables[...,1:2] + boxes_lables[...,3:4]/2
    #find the intersection area
    x1= torch.max(box1_x1,box2_x1)
    y1= torch.max[box1_y1,box2_y1]
    x2=torch.min(box1_x2,box2_x2)
    y2=torch.min(box1_y2,box2_y2)
    #calculate the area of intersection
    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    #calculate the area of both boxes
    box1_area = abs(box1_x2 - box1_x1) * abs(box1_y2 - box1_y1)
    box2_area = abs(box2_x2 - box2_x1) * abs(box2_y2 - box2_y1)
    #calculate IOU
    IOU = intersection /(box1_area+box2_area - intersection + 1e-6)
    return IOU

def non_max_suppression(bboxes,iou_threshold,confident):
    #ensure bboxes is list
    #format of bboxes is [x,y,w,h,confident,class_pred]
    assert type(bboxes) ==list
    bboxes_sort = [box for box in bboxes if box[4]>confident] #remove all boxes with confidence less than confident
    bboxes_sort = sorted(bboxes_sort,key=lambda x:x[4],reverse=True) #sort the boxes by confidence in descending order 
    bboxes_after_nms = []
    #chosen box is the box with the highest confidence
    while bboxes_sort:
        chonsen_box = bboxes_sort.pop(0)
        
        bboxes_sort = [
            box for box in bboxes_sort #remove all boxes with iou greater than iou_threshold and the same class
            if box[5] != chonsen_box[5] or iou(
                torch.tensor(chonsen_box[0:4]),torch.tensor(box[0:4])) < iou_threshold
        ]
        bboxes_after_nms.append(chonsen_box)
        
    return bboxes_after_nms

def get_bboxes(loader,model,iou_threshold,confidence,device="cuda"):
    all_pred_boxes = []
    all_true_boxes = []
    
    model.eval() #set model to evaluation mode
    train_idx = 0
    
    for batch_idx,(x,labels) in enumerate(loader):
        x=x.to(device)
        labels=labels.to(device)
        
        with torch.no_grad():
            predictions = model(x)
            
        batch_size = x.shape[0] #format of x is [batch_size, channels, height,width]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)
        
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                confident=confidence
            )
            
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx]+nms_box)
                
            for box in true_bboxes[idx]:
                if box[4]>0:
                    all_true_boxes.append([train_idx]+box)
            
            train_idx += 1
            
    model.train() #set model back to training mode
    return all_pred_boxes,all_true_boxes

def convert_cellboxes(predictions,S=7):
    predictions =predictions.to("cuda")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size,7,7,30)
    bboxes_1 = predictions[...,0:4] # x,y,w,h
    bboxes_2 = predictions[...,5:9] # x,y,w,h
    scores = torch.cat((predictions[...,4:5].unsqueeze(0),predictions[...,9:10].unsqueeze(0)),dim=0) #size [2,batch_size,7,7]
    best_box = scores.argmax(0).unsqueeze(-1) #chosen box with highest confidence
    best_boxes = bboxes_1 * (1 - best_box) + best_box * bboxes_2
    


def cellboxes_to_boxes(out,S=7):
    
        