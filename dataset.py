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
                values = [
                    float(val) if "." in val else int(val)
                    for val in label.strip().split()
                ]
                class_label, x, y, width, height = values[:5]
                boxes.append([class_label, x, y, width, height])
        boxes = torch.tensor(boxes)

        # Áp dụng transform nếu có
        if self.transform:
            image, boxes = self.apply_transforms(image, boxes)

        # Chuyển đổi hình ảnh thành tensor nếu không áp dụng transform
        if not self.transform:
            image = transforms.ToTensor()(image)

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
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 4] == 0:  # Thay đổi từ 20 thành 4
                label_matrix[i, j, 4] = 1  # Thay đổi từ 20 thành 4
                label_matrix[i, j, 0:4] = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 5 + class_label] = (
                    1  # Thay đổi từ class_label thành 5 + class_label
                )

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


import os
import torch
import cv2
import pytest
from dataset import YOLODataset


@pytest.fixture
def sample_data(tmpdir):
    img_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")

    img_file = img_dir.join("sample.jpg")
    label_file = label_dir.join("sample.txt")

    # Create a dummy image
    image = torch.randint(0, 255, (448, 448, 3), dtype=torch.uint8).numpy()
    cv2.imwrite(str(img_file), image)

    # Create a dummy label
    with open(label_file, "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    return [str(img_file)], [str(label_file)], str(img_dir), str(label_dir)


def test_yolo_dataset_length(sample_data):
    img_files, label_files, img_dir, label_dir = sample_data
    dataset = YOLODataset(img_files, label_files, img_dir, label_dir, S=7, B=2, C=20)
    assert len(dataset) == 1


def test_yolo_dataset_getitem(sample_data):
    img_files, label_files, img_dir, label_dir = sample_data
    dataset = YOLODataset(img_files, label_files, img_dir, label_dir, S=7, B=2, C=20)
    image, label_matrix = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 448, 448)
    assert isinstance(label_matrix, torch.Tensor)
    assert label_matrix.shape == (7, 7, 30)


def test_yolo_dataset_transform(sample_data):
    img_files, label_files, img_dir, label_dir = sample_data
    dataset = YOLODataset(
        img_files, label_files, img_dir, label_dir, S=7, B=2, C=20, transform=True
    )
    image, label_matrix = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 448, 448)
    assert isinstance(label_matrix, torch.Tensor)
    assert label_matrix.shape == (7, 7, 30)


def test_yolo_dataset_no_transform(sample_data):
    img_files, label_files, img_dir, label_dir = sample_data
    dataset = YOLODataset(
        img_files, label_files, img_dir, label_dir, S=7, B=2, C=20, transform=False
    )
    image, label_matrix = dataset[0]

    # Ensure the image has the correct tensor shape (3, 448, 448)
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 448, 448)  # Correct format for PyTorch
    assert isinstance(label_matrix, torch.Tensor)
    assert label_matrix.shape == (7, 7, 30)


if __name__ == "__main__":
    pytest.main([__file__])
