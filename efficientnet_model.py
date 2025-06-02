
import torch
import torchvision.models as models
import torch.nn as nn

def EfficientNetB0(n_classes=2):
    """
    Loads a pre-trained EfficientNetB0 model and modifies the classifier
    for transfer learning with a custom number of output classes.

    Args:
        n_classes (int): Number of output classes.

    Returns:
        model (torch.nn.Module): Modified EfficientNetB0 model ready for training.
    """

    # Load pre-trained EfficientNetB0 model
    model = models.efficientnet_b0(pretrained=True)

    # Optionally freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the number of input features for the classifier
    num_ftrs = model.classifier[1].in_features

    # Replace the classifier with a custom head
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs * 2, n_classes)
    )

    return model
