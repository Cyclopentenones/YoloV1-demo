import pandas as pd
import xml.etree.ElementTree as ET
import os
import torch
import torch.utils.data
from PIL import Image
from config import class_mapping
import yaml


class Dataset(torch.utils.data.Dataset):

    def __init__(self, annotations_dir, images_dir, yaml_dir, class_mapping):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.class_mapping = class_mapping
        self.yaml_dir = yaml_dir

    def custom_annotations(self):
        annotations = []
        for xml_file in os.listdir(self.annotations_dir):
            if xml_file.endswith(".xml"):
                annotation_path = os.path.join(self.annotations_dir, xml_file)
                tree = ET.parse(annotation_path)
                root = tree.getroot()

                filename_element = root.find("filename")
                if filename_element is not None:
                    image_id = filename_element.text
                else:
                    continue
                for obj in root.findall("object"):
                    name_element = obj.find("name")
                    if name_element is not None:
                        class_name = name_element.text
                        class_id = self.class_mapping[class_name]
                    else:
                        continue
                    bbox = obj.find("bndbox")
                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)

                    annotations.append(
                        {
                            "image_id": image_id,
                            "class_id": class_id,
                            "bbox": [xmin, ymin, xmax, ymax],
                        }
                    )
        return annotations

    def convert_to_yolo_format(self, annotations):
        yolo_annotations = []
        for annotation in annotations:
            image_path = os.path.join(self.images_dir, annotation["image_id"])
            image = Image.open(image_path)
            width, height = image.size

            class_id = annotation["class_id"]
            xmin, ymin, xmax, ymax = annotation["bbox"]

            center_x = (xmin + xmax) / 2.0 / width
            center_y = (ymin + ymax) / 2.0 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            yolo_annotations.append(
                f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
            )

        return yolo_annotations

    def save_yolo_annotations(self, yolo_annotations):
        yaml_files = [
            f
            for f in os.listdir(self.yaml_dir)
            if f.endswith(".yaml") or f.endswith(".yml")
        ]
        for yaml_file in yaml_files:
            if "train" in yaml_file or "val" in yaml_file:
                yaml_path = os.path.join(self.yaml_dir, yaml_file)
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                for annotation in yolo_annotations:
                    image_id = annotation["image_id"]
                    yolo_annotation = annotation["annotation"]

                    for entry in data:
                        if entry["id_img"] == image_id:
                            if "annotations" not in entry:
                                entry["annotations"] = []
                            entry["annotations"].append(yolo_annotation)

                with open(yaml_path, "w") as f:
                    yaml.safe_dump(data, f)


if __name__ == "__main__":
    # Example usage
    annotations_dir = "path/to/your/annotations_dir"
    images_dir = "path/to/your/images_dir"
    yaml_dir = "path/to/your/yaml_dir"
    class_mapping = {"class1": 0, "class2": 1}  # Example class mapping
    yolo_annotations = [
        {"image_id": 1, "annotation": "annotation1"},
        {"image_id": 2, "annotation": "annotation2"},
    ]

    dataset = Dataset(annotations_dir, images_dir, yaml_dir, class_mapping)
    dataset.save_yolo_annotations(yolo_annotations)
