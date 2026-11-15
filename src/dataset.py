import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# 2. Image transformations
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# --------------------------------------------------
# 3. Download CIFAR-10
# --------------------------------------------------

full_train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# --------------------------------------------------
# 4. Train / Validation split
# --------------------------------------------------

train_size = 45000
validation_size = 5000

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)


# --------------------------------------------------
# 5. DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# --------------------------------------------------
# 6. Check dataset
# --------------------------------------------------

print("\nDataset information")
print("-------------------")

print("Training samples   :", len(train_dataset))
print("Validation samples :", len(validation_dataset))
print("Test samples       :", len(test_dataset))

print("\nClasses:")
print(full_train_dataset.classes)


# --------------------------------------------------
# 7. Test one batch
# --------------------------------------------------

images, labels = next(iter(train_loader))

print("\nBatch information")
print("-----------------")

print("Image shape :", images.shape)
print("Label shape :", labels.shape)

print("Image device before GPU:", images.device)


# --------------------------------------------------
# 8. Move batch to GPU
# --------------------------------------------------

images = images.to(device)
labels = labels.to(device)

print("Image device after GPU :", images.device)
print("Label device after GPU :", labels.device)