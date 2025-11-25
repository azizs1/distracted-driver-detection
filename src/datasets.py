# src/datasets.py - defines dataset class for loading our dataset and data transforms
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from collections import Counter
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, folder, transform=None):
        """
        Initialized the dataset by reading the annotations file.
        Args:
            folder (string): Path to the dataset folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder = folder
        self.transform = transform

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        with open(os.path.join(folder, "_classes.txt"), "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.data = []
        with open(os.path.join(folder, "_annotations.txt"), "r") as f:
            for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if len(line.split(None, 1)) < 2:
                        print("Missing Label:", repr(line))
                        continue
                    filename, label = line.split(None, 1)
                    self.data.append([os.path.join(folder, filename), int(label.split(",")[-1])])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves a sample and its label by index."""
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label
    
    def print_class_distribution(self):
        labels = [label for _, label in self.data]
        counts = Counter(labels)
        
        print("Class distribution:")
        for label, count in sorted(counts.items()):
            print(f"    class {self.classes[label]}: {count}")


def get_transforms(img_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], augment=False):
    """
    Returns transforms for train/val/test.
    args:
        img_size (int): Desired image size (img_size x img_size).
        augment (bool): Whether to include data augmentation transforms.
        mean (list): Mean for normalization.
        std (list): Std for normalization.
    """
    base = [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),]
    
    if augment:
        aug = [transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.4),]
        return transforms.Compose(aug + base)
    
    return transforms.Compose(base)

def get_image_means_stds(folder):
    """Computes the mean and std of images in the dataset folder.
    args:
        folder (string): Path to the dataset folder.
    """
    if os.path.exists("data/mean.npy") and os.path.exists("data/std.npy"):
        print("==> Loading saved mean and std..")  
        mean = np.load("data/mean.npy")
        std = np.load("data/std.npy")
        return mean, std
    
    # Code refrences https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2?utm_source=chatgpt.com
    dataset = CustomDataset(folder, transform=transforms.ToTensor())
    full_loader = DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in tqdm(full_loader):
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset)).tolist()
    std.div_(len(dataset)).tolist()
    print(mean, std)
    print('==> Saving mean and std..')
    np.save("data/mean.npy", mean)
    np.save("data/std.npy", std)
    return mean, std


def show_samples(ds, k=6, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Shows first k samples from the dataset.  
    args:
        ds (CustomDataset): Dataset to show samples from.
        k (int): Number of samples to show.
        mean (list): Mean for normalization.
        std (list): Std for normalization.
    """
    # show first k samples
    plt.figure(figsize=(12, 6))

    mean = np.asarray(mean, dtype=np.float32)
    std  = np.asarray(std, dtype=np.float32)
    for i in range(k):
        img, label = ds[i]

        # (C,H,W) => (H,W,C)
        img = img.permute(1, 2, 0).numpy()

        # un-normalize
        # (H,W,C) * (C) + (C) => (H,W,C)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(ds.classes[label])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def load_datasets(batch_size=8, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], num_workers=4):
    """ Loads datasets"""
    train_dataset = CustomDataset("data/train", transform=get_transforms(img_size=image_size, mean=mean, std=std, augment=True))
    val_dataset = CustomDataset("data/valid", transform=get_transforms(img_size=image_size, mean=mean, std=std))
    test_dataset = CustomDataset("data/test", transform=get_transforms(img_size=image_size, mean=mean, std=std))

    # Normilize class imbalance using WeightedRandomSampler.
    labels = [label for _, label in train_dataset.data]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = np.array([weights[label] for label in labels])

    weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Load datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=weighted_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
def main():
    if os.path.exists("data/mean.npy") and os.path.exists("data/std.npy"):
        print("==> Loading saved mean and std..")  
        mean = np.load("data/mean.npy")
        std = np.load("data/std.npy")
    else:
        mean, std = get_image_means_stds("data/train")

    print("\n==> Loading datasets..")
    train = CustomDataset("data/train", transform=get_transforms(mean=mean, std=std, augment=True))
    print("\nTrain datasets length:", len(train))
    train.print_class_distribution()

    val = CustomDataset("data/valid", transform=get_transforms(mean=mean, std=std))
    print("\nValidation datasets length:", len(val))
    val.print_class_distribution()

    test = CustomDataset("data/test", transform=get_transforms(mean=mean, std=std))
    print("\nTest datasets length:", len(test))
    test.print_class_distribution()

    print("\nShowing some training samples:")
    show_samples(train, mean=mean, std=std)

if __name__ == "__main__":
    main()