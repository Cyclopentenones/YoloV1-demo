import os
import torch
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Danh sách các lớp VOC2012
classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class YOLODataset(Dataset):
    def __init__(
        self, img_files, label_files, img_dir, label_dir, S, B, C, transform=True
    ):
        self.img_files = img_files
        self.label_files = label_files
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Đọc đường dẫn hình ảnh và nhãn
        img_path = os.path.join(self.img_dir, self.img_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])

        # Kiểm tra tệp tồn tại
        assert os.path.exists(img_path), f"Image file not found: {img_path}"
        assert os.path.exists(label_path), f"Label file not found: {label_path}"

        # Đọc và chuyển đổi hình ảnh
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Đọc tệp nhãn
        boxes = []
        with open(label_path, "r") as f:
            for label in f.readlines():
                class_label, x, y, width, height, confidence = [
                    float(val) if "." in val else int(val)
                    for val in label.strip().split()
                ]
                boxes.append([class_label, x, y, width, height, confidence])
        boxes = torch.tensor(boxes)

        # Áp dụng transform nếu có
        if self.transform:
            image, boxes = self.apply_transforms(image, boxes)

        # Tạo ma trận nhãn
        label_matrix = self.create_label_matrix(boxes)

        return image, label_matrix

    def apply_transforms(self, image, boxes):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
            ]
        )
        image = transform(image)
        return image, boxes

    def create_label_matrix(self, boxes):
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height, _ = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 4] == 0:  # Thay đổi từ 20 thành 4
                label_matrix[i, j, 4] = 1  # Thay đổi từ 20 thành 4
                label_matrix[i, j, 0:4] = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 5 + class_label] = 1  # Thay đổi từ class_label thành 5 + class_label

        return label_matrix

    def draw_boxes(self, image_path, label_path):
        # Đọc hình ảnh
        image = cv2.imread(image_path)

        # Kiểm tra tệp tồn tại
        assert os.path.exists(label_path), f"Label file not found: {label_path}"

        with open(label_path, "r") as f:
            for line in f:
                cls_id, x_center, y_center, bbox_width, bbox_height, confidence = map(
                    float, line.strip().split()
                )
                cls_id = int(cls_id)
                h, w, _ = image.shape
                xmin = int((x_center - bbox_width / 2) * w)
                ymin = int((y_center - bbox_height / 2) * h)
                xmax = int((x_center + bbox_width / 2) * w)
                ymax = int((y_center + bbox_height / 2) * h)

                # Vẽ bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Thêm nhãn class
                label = f"{classes[cls_id]}: {confidence:.2f}"
                cv2.putText(
                    image,
                    label,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

