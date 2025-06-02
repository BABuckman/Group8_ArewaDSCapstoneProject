
import torch
import torchvision.models as models
import torch.nn as nn

def MobileNetV3(dropout, model_name='mobilenetv3', n_classes=2):
    """
    Loads a pre-trained MobileNetV3-Large model and modifies the classifier
    for transfer learning with a custom number of output classes.
    
    Args:
        model_name (str): Name for reference (not used to select model here).
        n_classes (int): Number of output classes.

    Returns:
        model (torch.nn.Module): Modified MobileNetV3 model ready for training.
    """

    # Load the pre-trained MobileNetV3-Large model
    model = models.mobilenet_v3_large(pretrained=True)

    # Freeze feature extractor layers (optional: comment this block if you want full fine-tuning)
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the number of input features to the final classifier layer
    num_ftrs = model.classifier[3].in_features

    # Redefine the classifier
    model.classifier[3] = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.BatchNorm1d(num_ftrs * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_ftrs * 2, n_classes)
    )

    return model
