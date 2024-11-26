import os
import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset


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
            image, boxes = self.transform(image, boxes)

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


def draw_boxes(image_path, label_path):
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
    draw_boxes(
        os.path.join(img_dir, img_files[0]), os.path.join(label_dir, label_files[0])
    )


if __name__ == "__main__":
    test()
