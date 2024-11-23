import torch
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as transforms


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_dir, images_dir, split_txt, class_mapping, split_size=7, num_boxes=2, transform=None):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.split_txt = split_txt
        self.class_mapping = class_mapping
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.transform = transform

        # Load image ids from split file (train.txt or test.txt)
        with open(split_txt, "r") as file:
            self.image_ids = file.read().splitlines()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        annotation_path = os.path.join(self.annotations_dir, f"{image_id}.xml")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Parse XML annotation
        boxes, labels = self.parse_annotation(annotation_path)

        # Convert bounding boxes to YOLO format and normalize image
        if self.transform:
            image, target = self.apply_transform(image, boxes, labels)

        return image, target

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.class_mapping:
                continue

            class_id = self.class_mapping[class_name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)

        return boxes, labels

    def convert_to_yolo_tensor(self, boxes, labels):
        target = torch.zeros((self.split_size, self.split_size, self.num_boxes * 5 + len(self.class_mapping)))

        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            class_id = label

            # Normalize bounding box coordinates
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Normalize to [0, 1] based on image size
            x_center /= 448  # Image size 448x448
            y_center /= 448
            width /= 448
            height /= 448

            # Map the box to the grid cell
            grid_x = int(x_center * self.split_size)
            grid_y = int(y_center * self.split_size)

            # Ensure grid_x and grid_y are within bounds
            grid_x = min(grid_x, self.split_size - 1)
            grid_y = min(grid_y, self.split_size - 1)

            # YOLO format: [x_center, y_center, width, height, confidence, class_one_hot]
            target[grid_y, grid_x, :5] = torch.tensor([x_center, y_center, width, height, 1.0])
            target[grid_y, grid_x, 5 + class_id] = 1  # One-hot encoding the class label

        return target


    def apply_transform(self, image, boxes, labels):
        # Resize image to (448, 448) and normalize to [0, 1]
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        image = transform(image)

        # Convert bounding boxes to YOLO format and normalize
        target = self.convert_to_yolo_tensor(boxes, labels)

        return image, target
