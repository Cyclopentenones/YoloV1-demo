import xml.etree.ElementTree as ET
import os
import torch
from PIL import Image
import yaml
import torch.utils.data
class_mapping = {
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
                    xmin = bbox.find('xmin').text
                    ymin = bbox.find('ymin').text
                    xmax = bbox.find('xmax').text
                    ymax = bbox.find('ymax').text
                
                    annotations.append({'image_id': image_id, 'class_id': class_id, 'bbox': [xmin, ymin, xmax, ymax]})
        return annotations

    def convert_to_yolo_format(self, annotations):
        yolo_annotations = []
        for annotation in annotations:
            image_path = os.path.join(self.images_dir, annotation['image_id'])
            image = Image.open(image_path)
            width, height = image.size
            
            class_id = annotation['class_id']
            xmin, ymin, xmax, ymax = map(float, annotation['bbox'])

            center_x = (xmin + xmax) / 2.0 / width
            center_y = (ymin + ymax) / 2.0 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            yolo_annotations.append(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}")
        
        return yolo_annotations

    def save_yolo_annotations(self, yolo_annotations):
        if not os.path.exists(self.yaml_dir):
            raise FileNotFoundError(f"The directory {self.yaml_dir} does not exist.")
        
        yaml_files = [f for f in os.listdir(self.yaml_dir) if f.endswith('.yaml') or f.endswith('.yml')]
        for yaml_file in yaml_files:
            if 'train' in yaml_file or 'val' in yaml_file:
                yaml_path = os.path.join(self.yaml_dir, yaml_file)
                with open(yaml_path, 'r') as f:
                    data = f.read().splitlines()
                
                print(f"Before updating {yaml_file}:")
                print(data)
                
                for annotation in yolo_annotations:
                    image_id = annotation.split()[0]
                    yolo_annotation = annotation
                    
                    for i, line in enumerate(data):
                        if f"id_img: {image_id}" in line:
                            if 'annotations:' not in data[i + 1]:
                                data.insert(i + 1, '  annotations:')
                            data.insert(i + 2, f"  - {yolo_annotation}")
                
                print(f"After updating {yaml_file}:")
                print(data)
                
                with open(yaml_path, 'w') as f:
                    f.write('\n'.join(data))

                # Print the contents of the YAML file after saving
                print(f"Contents of {yaml_file} after saving:")
                with open(yaml_path, 'r') as f:
                    print(f.read())



if __name__ == '__main__':
    train = 'C:\\Users\\ASUS\\Downloads\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\ImageSets\\Main'
    annotations_dir = 'C:\\Users\\ASUS\\Downloads\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\Annotations'
    images_dir = 'C:\\Users\\ASUS\\Downloads\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages'  

    dataset = Dataset(
        annotations_dir = annotations_dir,
        images_dir = images_dir,
        yaml_dir=train,
        class_mapping=class_mapping
    )
    annotations = dataset.custom_annotations()
    yolo_annotations = dataset.convert_to_yolo_format(annotations)
    dataset.save_yolo_annotations(yolo_annotations)