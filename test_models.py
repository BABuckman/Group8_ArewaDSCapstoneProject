
"""
test_models.py
Script to test each saved model on each ward's test dataset and compute performance metrics, confusion matrices, and ROC curves.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from resnet_model import ResNetBase
from vit_model import ViTBase
from efficientnet_model import EfficientNetB0
from mobilenetv3 import MobileNetV3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/content/drive/MyDrive/Arewa_capstone_project/"
BATCH_SIZE = 32

# Test transforms
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_test_loader(data_path):
    dataset = datasets.ImageFolder(data_path, transform=TEST_TRANSFORM)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False), dataset.classes

def load_model(model_name, num_classes, weights_path):
  
    from vit_model import ViTBase
    from resnet_model import ResNetBase
    from efficientnet_model import EfficientNetB0
    from mobilenetv3 import MobileNetV3


    if model_name == "vit_base":
        return ViTBase(n_classes=num_classes).to(DEVICE)
    elif model_name == "resnet18":
        return ResNetBase(n_classes=num_classes).to(DEVICE)
    elif model_name == "efficientnet_b0":
        return EfficientNetB0(n_classes=num_classes).to(DEVICE)
    elif model_name == "mobilenet_v3":
        return MobileNetV3(n_classes=num_classes, dropout=0.3).to(DEVICE)
    else:
        raise ValueError("Unsupported model.")

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def evaluate_model(model, dataloader, class_names, run_name):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except:
        auc_score = None

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {run_name}")
    plt.savefig(os.path.join(SAVE_DIR, f"confmat_test_{run_name}.png"))
    plt.close()

    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {auc(fpr, tpr):.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {run_name}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(SAVE_DIR, f"roc_test_{run_name}.png"))
        plt.close()
    except:
        print(f"ROC curve could not be generated for {run_name}.")

    return acc, f1, auc_score

def run_tests():
    datasets_paths = {
       "Dikumari": "/content/drive/MyDrive/Arewa_capstone_project/Tuta absoluta tomato dataset/DIKUMARI WARD FARM",
        "Kasaisa": "/content/drive/MyDrive/Arewa_capstone_project/Tuta absoluta tomato dataset/KASAISA WARD FARM",
        "Kukareta": "/content/drive/MyDrive/Arewa_capstone_project/Tuta absoluta tomato dataset/KUKARETA WARD FARM",
        "Combined": "/content/drive/MyDrive/Arewa_capstone_project/Tuta absoluta tomato dataset/VillagePlant" 
    }

    model_names = ["vit_base", "resnet18", "mobilenet_v3", "efficientnet_b0"]
    test_results = []

    for model_name in model_names:
        for dataset_label, dataset_path in datasets_paths.items():
            model_file = os.path.join(SAVE_DIR, f"best_model_{model_name}_{dataset_label}.pth")
            if not os.path.exists(model_file):
                print(f"❌ Model not found: {model_file}")
                continue

            test_loader, class_names = load_test_loader(dataset_path)
            model = load_model(model_name, num_classes=len(class_names), weights_path=model_file)
            run_name = f"{model_name}_{dataset_label}_test"
            acc, f1, auc_score = evaluate_model(model, test_loader, class_names, run_name)

            print(f"✅ Tested {model_name} on {dataset_label} → Acc: {acc:.2f}, F1: {f1:.2f}, AUC: {auc_score if auc_score else 'N/A'}")

            test_results.append({
                "Model": model_name,
                "Test Dataset": dataset_label,
                "Accuracy": acc,
                "F1 Score": f1,
                "ROC AUC": auc_score if auc_score else "N/A"
            })

    df = pd.DataFrame(test_results)
    df.to_csv(os.path.join(SAVE_DIR, "test_model_results.csv"), index=False)
    print("📊 Test results saved to test_model_results.csv")

