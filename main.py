import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import Dataset
from criterion import Loss
from config import train, val, class_mapping, epochs, batch_size, learning_rate

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloaders
    train_dataset = Dataset(annotations_dir='path/to/Annotations', images_dir='path/to/JPEGImages', yaml_dir=train, class_mapping=class_mapping)
    val_dataset = Dataset(annotations_dir='path/to/Annotations', images_dir='path/to/JPEGImages', yaml_dir=val, class_mapping=class_mapping)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    criterion = Loss(S=7, B=2, C=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                predictions = model(images)
                loss = criterion(predictions, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()