import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modelYOLOV1 import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
from tqdm import tqdm

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Adjust based on your GPU memory
WEIGHT_DECAY = 0
EPOCHS = 2
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False #***  # Set to False since it's your first time training
LOAD_MODEL_FILE = "overfit.pth.tar"
  # No model to load initially
IMG_DIR = r"C:\Users\khiem\Downloads\NCKH\Task PASCAL VOC2012\VOCtrainval_11-May-2012 (1)\VOCdevkit\VOC2012\JPEGImages"
LABEL_DIR = r"C:\Users\khiem\Downloads\NCKH\Task PASCAL VOC2012\VOCtrainval_11-May-2012 (1)\VOCdevkit\VOC2012\Annotations"
TRAIN_CSV = r"C:\Users\khiem\Downloads\NCKH\Task PASCAL VOC2012\VOCtrainval_11-May-2012 (1)\VOCdevkit\VOC2012\ImageSets\Main\train.txt"  # Adjust the path as needed
TEST_CSV = r"C:\Users\khiem\Downloads\NCKH\Task PASCAL VOC2012\VOCtrainval_11-May-2012 (1)\VOCdevkit\VOC2012\ImageSets\Main\val.txt"  # Adjust the path as needed

# Transformations for input data
transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    class_mapping = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19
    }

    # Initialize the model
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # Load model if previously trained (but it's your first time)
    if LOAD_MODEL and LOAD_MODEL_FILE:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Load datasets once
    train_dataset = VOCDataset(
        split_txt=TRAIN_CSV,
        transform=transform,
        images_dir=IMG_DIR,
        annotations_dir=LABEL_DIR,
        class_mapping=class_mapping
    )

    test_dataset = VOCDataset(
        split_txt=TEST_CSV,
        transform=transform,
        images_dir=IMG_DIR,
        annotations_dir=LABEL_DIR,
        class_mapping=class_mapping
    )

    # Create DataLoader (but we will not load the data in each epoch)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # Pre-load all data into memory for training
    train_data = list(train_loader)  # Load data into memory once
    test_data = list(test_loader)  # Load data into memory once

    for epoch in range(EPOCHS):
        # Track performance on train set
        pred_boxes, target_boxes = get_bboxes(
            train_data, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        # Training loop, use pre-loaded data
        train_fn(train_data, model, optimizer, loss_fn)

        # Save model after every epoch
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

        

if __name__ == "__main__":
    main()
