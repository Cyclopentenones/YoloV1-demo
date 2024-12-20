import torch
import torch.nn as nn
from utils import iou, non_max_suppression

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, S=7, B=2, C=20, **kwargs):
        super(Yolov1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        x = self.fcs(torch.flatten(x, start_dim=1))
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        x[..., 0:self.B * 5:5] = torch.sigmoid(x[..., 0:self.B * 5:5])  
        x[..., 1:self.B * 5:5] = torch.sigmoid(x[..., 1:self.B * 5:5])  
        x[..., 4:self.B * 5:5] = torch.sigmoid(x[..., 4:self.B * 5:5])  

        return x


    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,
                        x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),  # Đảm bảo kích thước đúng
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)),
        )

    def predict(self, x, _nms, _conf):
        predictions = self.forward(x)
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)

        all_boxes = []
        for i in range(predictions.size(0)):  # Batch size
            boxes = []
            for j in range(self.S):  # Grid cells
                for k in range(self.S):  # Grid cells (y-axis)
                    for b in range(self.B):  # Bounding boxes
                        # Lấy giá trị bounding box
                        bbox = predictions[i, j, k, b * 5 : (b + 1) * 5]
                        x_center, y_center, w, h, confidence = (
                            bbox[0].item(),
                            bbox[1].item(),
                            bbox[2].item(),
                            bbox[3].item(),
                            bbox[4].item(),
                        )

                        # Lấy class probabilities
                        class_probs = torch.sigmoid(
                            predictions[i, j, k, self.B * 5 :]  # C classes
                        )
                        class_scores = (confidence * class_probs).tolist()

                        # Tạo box dưới dạng list
                        box = [x_center, y_center, w, h, confidence] + class_scores
                        boxes.append(box)

            # Áp dụng Non-Max Suppression
            boxes = non_max_suppression(boxes, _nms, _conf)
            all_boxes.append(boxes)

        return all_boxes



import unittest
from modelYOLOV1 import Yolov1

class TestYolov1(unittest.TestCase):
    def setUp(self):
        self.model = Yolov1()
        self.input_tensor = torch.randn(
             (1, 3, 448, 448)
        )  # Batch size of 1, 3 channels, 448x448 image

    def test_forward_pass(self):
         output = self.model(self.input_tensor)
         self.assertEqual(output.shape, (1, 7 * 7 * (20 + 2 * 5)))

    def test_predict(self):
         _nms = 0.5
         _conf = 0.4
         predictions = self.model.predict(self.input_tensor, _nms, _conf)
         self.assertIsInstance(predictions, list)
         self.assertIsInstance(predictions[0], list)
         if len(predictions[0]) > 0:
             self.assertIsInstance(predictions[0][0], dict)
             self.assertIn("x_center", predictions[0][0])
             self.assertIn("y_center", predictions[0][0])
             self.assertIn("w", predictions[0][0])
             self.assertIn("h", predictions[0][0])
             self.assertIn("confidence", predictions[0][0])
             self.assertIn("class_scores", predictions[0][0])
         else:
             print("No boxes detected, check the confidence threshold.")


if __name__ == "__main__":
     unittest.main()
