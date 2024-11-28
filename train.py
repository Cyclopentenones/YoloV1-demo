import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modelYOLOV1 import Yolov1
from dataset import YOLODataset, classes
from utils import (
    non_max_suppression,
    mean_average_precision,
    iou,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
from tqdm import tqdm
import os
import time

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 4 
PIN_MEMORY = True
LOAD_MODEL = False  
LOAD_MODEL_FILE = "overfit.pt"
IMG_DIR = r"C:\Users\ASUS\Downloads\YoloV1-demo-Kien\YoloV1-demo-Kien\data\train"
LABEL_DIR = r"C:\Users\ASUS\Downloads\YoloV1-demo-Kien\YoloV1-demo-Kien\data\train_labels"
transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True, desc='Training')
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model.predict(x, _nms = 0.5, _conf = 0.4)
        loss = loss_fn.forward(out,y)
        mean_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}

    # Initialize the model
    model = Yolov1(S=7, B=2, C=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # Load model if previously trained (but it's your first time)
    if LOAD_MODEL:
        if os.path.isfile(LOAD_MODEL_FILE):
            print(f"Loading checkpoint from {LOAD_MODEL_FILE}...")
            load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        else:
            print(f"Checkpoint file {LOAD_MODEL_FILE} not found. Starting fresh.")

    # Load image and label files
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    label_files = [f.replace('.jpg', '.txt') for f in img_files]

    # Load datasets
    train_dataset = YOLODataset(
        img_files=img_files,
        label_files=label_files,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=7,
        B=2,
        C=20,
        transform=transform,
    )

    test_dataset = YOLODataset(
        img_files=img_files,
        label_files=label_files,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=7,
        B=2,
        C=20,
        transform=transform,
    )

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

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        start_time = time.time()

        # Track performance on train set
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, confidence=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, num_class=20
        )
        print(f"Train mAP: {mean_avg_prec}")

        # Training loop
        train_fn(train_loader, model, optimizer, loss_fn)

        end_time = time.time()
        print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds")

        # Save model
        if epoch % 5 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)


if __name__ == "__main__":
    main()