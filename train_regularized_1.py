import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
from torch.utils.data import random_split
from helpers import compute_mean_and_std
from transfer_learning_models import ViTBase, MobileNetV3, EfficientNetB0, ResNetBase 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SAVE_DIR = "/content/drive/MyDrive/Arewa_capstone_project/test_save_models"
PATIENCE = 5

def get_transforms():
    mean, std = compute_mean_and_std()
    return {
        'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }
def get_dataloaders(data_dir):
    data_transforms = get_transforms()
    
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Override transforms for val and test sets
    val_dataset.dataset.transform = data_transforms['val_test']
    test_dataset.dataset.transform = data_transforms['val_test']

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  # drop_last=True to avoid batch size 1 issues
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)

    class_names = full_dataset.classes

    return train_loader, val_loader, test_loader, class_names

# =============== TRAIN LOOP =====================

def train_model(data_dir, model_name="vit_b_16", save_dir=SAVE_DIR):
    wandb.init(project="final_training_tomato-disease-detection", name=f"train_{model_name}", config={
        "model": model_name,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 1e-3,
        "patience": PATIENCE
    })
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir)
    num_classes = len(class_names)

    # Build model
    if model_name == "vit_b_16":
        model = ViTBase(n_classes=num_classes).to(DEVICE)
    elif model_name == "resnet18":
        model = ResNetBase(n_classes=num_classes).to(DEVICE)
    elif model_name == "efficientnet_b0":
        model = EfficientNetB0(n_classes=num_classes).to(DEVICE)
    elif model_name == "mobilenet_v3":
        model = MobileNetV3(n_classes=num_classes, dropout=0.3).to(DEVICE)
    else:
        raise ValueError(f"Model {model_name} is not supported. Choose from: 'vit_b_16', 'resnet18', 'efficientnet_b0', 'mobilenet_v3'")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = 100 * train_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * val_correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Early stopping
        model_file = os.path.join(save_dir, f"best_model_{model_name}.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), model_file)
            print("✅ Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    wandb.finish()
    print("Training complete.")
    return model, val_acc


   
