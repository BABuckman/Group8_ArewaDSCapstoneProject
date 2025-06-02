
from PIL import Image
import os
import torch
import multiprocessing
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from helpers import compute_mean_and_std, get_data_location

class CustomTomatoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        for ward_farm in os.listdir(root):
            ward_path = os.path.join(root, ward_farm)
            if not os.path.isdir(ward_path):
                continue
            for label_type in os.listdir(ward_path):
                class_path = os.path.join(ward_path, label_type)
                if not os.path.isdir(class_path):
                    continue
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        self.samples.append((img_path, label_type.upper()))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(data_dir, batch_size, **kwargs):
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomAffine(scale=(0.9, 1.1), translate=(0.1, 0.1), degrees=10, fill=0,
                                     interpolation=transforms.InterpolationMode.BILINEAR),
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
        ])
    }

    data_dir = get_data_location()
    full_dataset = CustomTomatoDataset(data_dir, transform=data_transforms['train'])

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    val_dataset.dataset.transform = data_transforms['val_test']
    test_dataset.dataset.transform = data_transforms['val_test']

    num_workers = multiprocessing.cpu_count()

    gpu_kwargs = {
        "persistent_workers": True,
        "pin_memory": True,
        "pin_memory_device": 'cuda:0',
        "prefetch_factor": 4
    } if torch.cuda.is_available() else {}

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, **gpu_kwargs),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, **gpu_kwargs),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, **gpu_kwargs)
    }

    class_names = list(set(label for _, label in full_dataset.samples))
    dataset_sizes = {k: len(dataloaders[k].dataset) for k in dataloaders}

    return dataloaders, dataset_sizes, class_names

def visualize_one_batch(data_loaders, max_n: int = 5):
    import matplotlib.pyplot as plt
    from torchvision import transforms

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    # Unnormalize images for display
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
        transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0])
    ])
    images = invTrans(images)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    fig = plt.figure(figsize=(25, 4))
    for idx in range(min(max_n, len(images))):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])

        # Extract ward and label from file path
        img_path, _ = data_loaders["train"].dataset[idx]
        path_parts = img_path.split(os.sep)
        ward_name = path_parts[-3]  # e.g., 'ward3'
        label_str = labels[idx]     # 'INFECTED' or 'HEALTHY'

        color = 'red' if 'INFECTED' in label_str.upper() else 'green'
        ax.set_title(f"{ward_name.upper()} | {label_str}", color='black', fontweight='bold')
        ax.text(0.5, -0.1, f"{label_str.upper()}", size=12,
                ha='center', transform=ax.transAxes, color=color)

    plt.tight_layout()
    plt.show()

# def visualize_one_batch(data_loaders, max_n: int = 5):
#     dataiter = iter(data_loaders["train"])
#     images, labels = next(dataiter)

#     mean, std = compute_mean_and_std()
#     invTrans = transforms.Compose([
#         transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
#         transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0])
#     ])
#     images = invTrans(images)

#     images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

#     fig = plt.figure(figsize=(25, 4))
#     for idx in range(min(max_n, len(images))):
#         ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
#         ax.imshow(images[idx])

#         label = labels[idx]
#         if isinstance(label, str) and ':' in label:
#             ward, state = map(str.strip, label.split(':'))
#             state_color = 'green' if 'HEALTHY' in state.upper() else 'red'
#             ax.set_title(f"{ward}: {state}", color='black', fontweight='bold')
#             ax.text(0.5, -0.1, f"{state}",
#                     size=12,
#                     ha='center',
#                     transform=ax.transAxes,
#                     color=state_color)
#         else:
#             ax.set_title(label, color='black')
