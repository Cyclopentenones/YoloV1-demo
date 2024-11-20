import xml.etree.ElementTree as ET
import os
from PIL import Image
import torch
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
    def __init__(self, annotations_dir, images_dir, train_txt_path, class_mapping):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.class_mapping = class_mapping
        self.train_txt_path = train_txt_path

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

                    # Sử dụng float() thay vì int() vì tọa độ có thể có giá trị thập phân
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

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
            
            # Thêm image_id vào dữ liệu YOLO
            yolo_annotations.append(f"{annotation['image_id']} {class_id} {center_x} {center_y} {bbox_width} {bbox_height}")
        
        return yolo_annotations

    def update_train_txt(self):
        annotations = self.custom_annotations()  # Lấy annotations từ custom_annotations()
        yolo_annotations = self.convert_to_yolo_format(annotations)

        # Mở và ghi vào file train.txt
        with open(self.train_txt_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(f"{annotation}\n")

        print(f"Đã cập nhật {self.train_txt_path} với các bounding box theo định dạng YOLO.")

if __name__ == '__main__':
    # Đường dẫn đến thư mục chứa annotations và images
    annotations_dir = r'C:\Users\khiem\Downloads\NCKH\Task PASCAL VOC2012\VOCtrainval_11-May-2012 (1)\VOCdevkit\VOC2012\Annotations'
    images_dir = r'C:\Users\khiem\Downloads\NCKH\Task PASCAL VOC2012\VOCtrainval_11-May-2012 (1)\VOCdevkit\VOC2012\JPEGImages'
    train_txt_path = r'C:\Users\khiem\Downloads\NCKH\Task PASCAL VOC2012\VOCtrainval_11-May-2012 (1)\VOCdevkit\VOC2012\ImageSets\Main\train.txt'

    dataset = Dataset(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        train_txt_path=train_txt_path,
        class_mapping=class_mapping
    )
    
    dataset.update_train_txt()  # Cập nhật file train.txt với các bounding box theo định dạng YOLO
