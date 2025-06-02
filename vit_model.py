
import torch
import torchvision.models as models
import torch.nn as nn

def ViTBase(n_classes=2):
    """
    Loads a pre-trained ViT-Base (vit_b_16) model and modifies the classifier
    for transfer learning with a custom number of output classes.

    Args:
        n_classes (int): Number of target classes for classification.

    Returns:
        model (torch.nn.Module): Modified ViT-Base model ready for fine-tuning.
    """

    # Load pre-trained ViT-Base model (ViT-B/16 from torchvision)
    model = models.vit_b_16(pretrained=True)

    # Freeze the patch + transformer encoder (optional; enables fast training)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Get number of features from the original head
    num_ftrs = model.heads.head.in_features

    # Replace classification head with a custom classifier
    model.heads.head = nn.Sequential(
        nn.LayerNorm(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(num_ftrs * 2, n_classes)
    )

    return model
