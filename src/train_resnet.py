# src/train_resnet.py - script to train a resnet on the dataset
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import CustomDataset, get_transforms, get_image_means_stds, load_datasets
from train import train, evaluation, print_results, pick_device
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.models as models
from collections import Counter

# Source https://www.geeksforgeeks.org/deep-learning/how-to-implement-transfer-learning-in-pytorch/
class ModifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Change the final fully connected layer for 10 classes
        # Only train the final fully connected layer
        for name, param in self.resnet.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)


def main():
    # Device setup
    device = pick_device()

    # ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

     # Hyperparameters
    batch_size = 8
    image_size = 224
    learning_rate = 0.001
    num_epochs = 20
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 4
    label_smoothing = 0.05

    # Create datasets
    print("\n==> Loading datasets..")
    train_loader, val_loader, test_loader = load_datasets(batch_size=batch_size, image_size=image_size, mean=mean, std=std, num_workers=num_workers)

    labels = [label for _, label in train_loader.data]
    # model = models.resnet50(pretrained=True)
    model = ModifiedResNet(len(set(labels))).to(device)
    summary(model, input_size=(3, image_size, image_size))

    criterion = nn.CrossEntropyLoss() # label_smoothing=0.05
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1) Possibly add a learning rate scheduler later

    print("Loader len:", len(train_loader))
    x, y = next(iter(train_loader))
    print("First batch shapes:", x.shape, y.shape)

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train(model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs, device=device)
    evaluation(model, val_loader, verbose=True, device=device)

    # Save the model
    SIMPLECNN_MODEL_PATH = 'data/customcnn_model_resnet.pth'
    torch.save(model.state_dict(), SIMPLECNN_MODEL_PATH)
    print(f'\n==> Saved trained model to {SIMPLECNN_MODEL_PATH}')

    # Print results
    print_results(train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == "__main__":
    main()