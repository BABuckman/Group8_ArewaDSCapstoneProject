
import torch
import torchvision.models as models
import torch.nn as nn

def ResNetBase(n_classes=2, freeze_backbone=True, dropout=0.3, model_name='resnet18'):
    """
    Loads a pre-trained ResNet model and modifies the classifier
    for transfer learning with a custom number of output classes.

    Args:
        n_classes (int): Number of target output classes.
        freeze_backbone (bool): Whether to freeze the backbone (feature extractor).
        dropout (float): Dropout rate for regularization.
        model_name (str): ResNet variant to load ('resnet18', 'resnet34', 'resnet50', 'resnet101').

    Returns:
        model (torch.nn.Module): Modified ResNet model ready for fine-tuning.
    """

    # Load the specified ResNet model
    model_name = 'resnet18'
    model = getattr(models, model_name)(pretrained=True)

    # Optionally freeze the backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Get number of input features for the final FC layer
    in_features = model.fc.in_features

    # Replace the final FC layer with a custom head
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, in_features * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_features * 2, n_classes)
    )

    return model
