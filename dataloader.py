


import os
import random
import torch
import multiprocessing
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from helpers import compute_mean_and_std, get_data_location



# Function to load data and create DataLoader instances
def load_data(data_dir, batch_size, **kwargs):



    data_loaders = {"train": None, "valid": None, "test": None}

    

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomAffine(scale = (0.9, 1.1), translate = (0.1, 0.1), degrees = 10, fill = 0,interpolation=transforms.InterpolationMode.BILINEAR),
             transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(224, padding_mode="reflect", pad_if_needed=True),

            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),

            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]),
        'val_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ])
    }

    data_dir = get_data_location()
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    val_dataset.dataset.transform = data_transforms['val_test']
    test_dataset.dataset.transform = data_transforms['test']

    num_workers = multiprocessing.cpu_count() 

         # Check if CUDA is available
    if torch.cuda.is_available():
        gpu_kwargs = {
            "persistent_workers": True, 
            "pin_memory": True, 
            "pin_memory_device": 'cuda:0', 
            "prefetch_factor": 4
        }
    else:
        gpu_kwargs = {}
    
    

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,**gpu_kwargs),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,**gpu_kwargs),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,**gpu_kwargs)
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    class_names = full_dataset.classes

    return dataloaders, dataset_sizes, class_names







def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    
    # obtain one batch of training images
    # First obtain an iterator from the train dataloader
    dataiter  = iter(data_loaders["train"]) 
    # Then call the .next() method on the iterator you just
    # obtained
    images, labels  = next(dataiter)  

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names  =  data_loaders['train'].dataset.dataset.classes
   

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])

def ward_dataloaders(root_dir, batch_size=32, val_split=0.2):
    """
    Loads train and validation loaders from WARD's directory with subfolders as class labels.

    Args:
        root_dir (str): Path to the WARD's folder
        batch_size (int): Batch size for DataLoader
        val_split (float): Fraction of data to use for validation

    Returns:
        train_loader, val_loader, class_names
    """

    # Define standard transforms
    mean, std  = compute_mean_std()

    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomAffine(scale = (0.9, 1.1), translate = (0.1, 0.1), degrees = 10, fill = 0,interpolation=transforms.InterpolationMode.BILINEAR),
             transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(224, padding_mode="reflect", pad_if_needed=True),

            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),

            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]),
        'val_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ])
    }

    data_dir = get_data_location(root_dir)
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    val_dataset.dataset.transform = data_transforms['val_test']
    test_dataset.dataset.transform = data_transforms['test']

    num_workers = multiprocessing.cpu_count() 

         # Check if CUDA is available
    if torch.cuda.is_available():
        gpu_kwargs = {
            "persistent_workers": True, 
            "pin_memory": True, 
            "pin_memory_device": 'cuda:0', 
            "prefetch_factor": 4
        }
    else:
        gpu_kwargs = {}
    
    

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,**gpu_kwargs),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,**gpu_kwargs),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,**gpu_kwargs)
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    class_names = full_dataset.classes

    return dataloaders, dataset_sizes, class_names