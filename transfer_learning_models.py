import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet18, ResNet18_Weights

def MobileNetV3(n_classes=2, dropout=0.3, freeze_backbone=True):
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    num_ftrs = model.classifier[3].in_features

    model.classifier[3] = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.BatchNorm1d(num_ftrs * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_ftrs * 2, n_classes)
    )

    return model




def ViTBase(n_classes=2, freeze_backbone=True):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    if freeze_backbone:
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Get features
    num_ftrs = model.heads.head.in_features

    # Replace classification head
    model.heads.head = nn.Sequential(
        nn.LayerNorm(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(num_ftrs * 2, n_classes)
    )

    return model

def EfficientNetB0(n_classes=2, freeze_backbone=True, dropout=0.5):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    num_ftrs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_ftrs * 2, n_classes)
    )

    return model


def ResNetBase(n_classes=2, freeze_backbone=True, dropout=0.3):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, in_features * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_features * 2, n_classes)
    )

    return model
