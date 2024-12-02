import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from modelYOLOV1 import Yolov1
from utils import non_max_suppression, plot_image, cellboxes_to_boxes

# Load model YOLOv1
model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
checkpoint = torch.load('overfit.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load image
image_path = r'C:\Users\khiem\Downloads\YOLOGIT\xe-dap-co-3.jpg'  # Thay bằng đường dẫn tới ảnh của bạn
image = Image.open(image_path).convert("RGB")

# Transform image
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)  # Thêm batch dimension

# Make prediction
with torch.no_grad():
    predictions = model(image_tensor)

# Convert predictions to bounding boxes
bboxes = cellboxes_to_boxes(predictions)
bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

# Plot image with bounding boxes
plot_image(image, bboxes)