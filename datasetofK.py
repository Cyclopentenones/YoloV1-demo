import torch
import os
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, class_to_idx, S=7, B=2, C=20, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.class_to_idx = class_to_idx
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, img_file.replace('.jpg', '.xml'))
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"No such file: '{label_path}'")

        boxes = []
        tree = ET.parse(label_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Chuyển đổi tọa độ từ xmin, ymin, xmax, ymax sang x, y, width, height
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            # Ánh xạ class_label từ chuỗi ký tự thành số nguyên
            class_label = self.class_to_idx[class_label]

            boxes.append([class_label, x, y, width, height])

        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)

            # Đảm bảo rằng i và j nằm trong phạm vi hợp lệ
            if i >= self.S:
                i = self.S - 1
            if j >= self.S:
                j = self.S - 1

            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

# Hàm hiển thị ảnh
def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Ánh xạ tên lớp thành số nguyên
class_to_idx = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}

# Ví dụ sử dụng
dataset = VOCDataset(
    img_dir='C:/Users/khiem/Downloads/NCKH/Task PASCAL VOC2012/VOCtrainval_11-May-2012 (1)/VOCdevkit/VOC2012/JPEGImages', 
    label_dir='C:/Users/khiem/Downloads/NCKH/Task PASCAL VOC2012/VOCtrainval_11-May-2012 (1)/VOCdevkit/VOC2012/Annotations',
    class_to_idx=class_to_idx
)