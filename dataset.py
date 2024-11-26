import os
import torch
import cv2

import torchvision.transforms as transforms
from torch.utils.data import Dataset

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
        self, img_files, label_files, img_dir, label_dir, S, B, C, transform=None
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
        img_path = os.path.join(self.img_dir, self.img_files[index])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(self.label_dir, self.label_files[index])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.strip().split()
                ]
                boxes.append([class_label, x, y, width, height])

        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.apply_transforms(image, boxes, label)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

    def apply_transforms(self, image, boxes, labels):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
            ]
        )

        return transform(image), self.convert_to_yolo_tensor(boxes, labels)

    def convert_to_yolo_tensor(self, boxes, labels):
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

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
            grid_x = int(x_center * self.S)
            grid_y = int(y_center * self.S)

            # Ensure grid_x and grid_y are within bounds
            grid_x = min(grid_x, self.S - 1)
            grid_y = min(grid_y, self.S - 1)

            # YOLO format: [x_center, y_center, width, height, confidence, class_one_hot]
            target[grid_y, grid_x, 20:25] = torch.tensor(
                [1.0, x_center, y_center, width, height]
            )
            target[grid_y, grid_x, class_id] = 1  # One-hot encoding the class label

        return target

    def draw_boxes(self, image_path, label_path, class_names=classes):
        image = cv2.imread(image_path)
        with open(label_path, "r") as f:
            for line in f:
                cls_id, x_center, y_center, bbox_width, bbox_height = map(
                    float, line.strip().split()
                )
                cls_id = int(cls_id)
                h, w, _ = image.shape
                xmin = int((x_center - bbox_width / 2) * w)
                ymin = int((y_center - bbox_height / 2) * h)
                xmax = int((x_center + bbox_width / 2) * w)
                ymax = int((y_center + bbox_height / 2) * h)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Add class label
                label = f"{class_names[cls_id]}"
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


def test():
    img_files = ["2008_000008.jpg", "2008_000003.jpg"]
    label_files = ["2008_000008.txt", "2008_000003.txt"]
    img_dir = "E:\\NCKH\\babykiller\\YoloV1-demo\\data\\train"
    label_dir = "E:\\NCKH\\babykiller\\YoloV1-demo\\data\\train_labels"
    S = 7
    B = 2
    C = 20
    transform = None

    dataset = YOLODataset(
        img_files=img_files,
        label_files=label_files,
        img_dir=img_dir,
        label_dir=label_dir,
        S=S,
        B=B,
        C=C,
        transform=transform,
    )

    img, target = dataset[0]
    print(img.shape, target.shape)
    dataset.draw_boxes(
        os.path.join(img_dir, img_files[0]), os.path.join(label_dir, label_files[0])
    )


if __name__ == "__main__":
    test()
