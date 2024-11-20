import pandas as pd 
import xml.etree.ElementTree as ET
import os
import torch
from PIL import Image
from config import class_mapping
import yaml
class Dataset(torch.utils.data.Dataset):

    def __init__(self, annotations_dir, images_dir, yaml_dir ,class_mapping):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.class_mapping = class_mapping
        self.yaml_dir = yaml_dir

    def custom_annotations(self):
        annotations = []
        for xml_file in os.listdir(self.annotations_dir):
            if xml_file.endswith('.xml'):
                annotation_path = os.path.join(self.annotations_dir, xml_file)
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                
                image_id = root.find('filename').text
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_id = self.class_mapping[class_name]
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    annotations.append({'image_id': image_id, 'class_id': class_id, 'bbox': [xmin, ymin, xmax, ymax]})
        return annotations

    def convert_to_yolo_format(self, annotations):
        yolo_annotations = []
        for annotation in annotations:
            image_path = os.path.join(self.images_dir, annotation['image_id'])
            image = Image.open(image_path)
            width, height = image.size
            
            class_id = annotation['class_id']
            xmin, ymin, xmax, ymax = annotation['bbox']
            
            center_x = (xmin + xmax) / 2.0 / width
            center_y = (ymin + ymax) / 2.0 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            yolo_annotations.append(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}")
        
        return yolo_annotations

    def save_yolo_annotations(self, yolo_annotations):
        yaml_files = [f for f in os.listdir(self.yaml_dir) if f.endswith('.yaml') or f.endswith('.yml')]
        for yaml_file in yaml_files:
            if 'train' in yaml_file or 'val' in yaml_file:
                yaml_path = os.path.join(self.yaml_dir, yaml_file)
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                for annotation in yolo_annotations:
                    image_id = annotation['image_id']
                    yolo_annotation = annotation['annotation']
                    
                    for entry in data:
                        if entry['id_img'] == image_id:
                            if 'annotations' not in entry:
                                entry['annotations'] = []
                            entry['annotations'].append(yolo_annotation)
                
                with open(yaml_path, 'w') as f:
                    yaml.safe_dump(data, f)

