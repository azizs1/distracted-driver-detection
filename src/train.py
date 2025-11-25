# src/train.py - script to train a model on the dataset
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import CustomDataset, get_transforms, get_image_means_stds
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data.sampler import WeightedRandomSampler

class customCNN(nn.Module):
    '''Custom CNN neural network. See above description for what each layer does.'''

    def __init__(self):
        '''Defines each layer for customCNN.'''
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.adapt = nn.AdaptiveAvgPool2d((64, 64))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=128 * 56 * 56, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        '''Defines graph of connections for each layer in customCNN.'''
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(model, loader, validation, criterion, optimizer, epochs=2, device="cpu"):
    '''Train a model from training data.

    Args:
    - model: Neural network to train
    - epochs: Number of epochs to train the model
    - loader: Dataloader to train the model with
    '''
    print('Start Training')
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        total = 0
        correct = 0
        tqdm_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        #scheduler.step()
        for i, data in enumerate(tqdm_bar):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Move inputs and labels to the correct device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            tqdm_bar.set_postfix(loss=running_loss/(i+1), acc=correct/total)
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    val_loss, val_acc = evaluation(model, validation, criterion=criterion, verbose=False, device=device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print('\nFinished Training')
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluation(model, loader, criterion=None, verbose=False, device="cpu"):
    '''Evaluate a model and output its accuracy on a test dataset.

    Args:
    - model: Neural network to evaluate
    - loader: Dataloader containing test dataset
    '''
    # Evaluate accuracy on validation / test set
    correct = 0
    total = 0
    running_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(loader):
            images, labels = data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # calculate outputs by running images through the network
            outputs = model(images)

            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if verbose:
        print(f'\nAccuracy of the network on the {len(loader)} validation images: {100 * correct // total} %')

    val_acc = correct / total
    val_loss = running_loss / len(loader)
    return val_loss, val_acc

def main():
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if possible
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    # Load dataset mean and std
    if os.path.exists("data/mean.npy") and os.path.exists("data/std.npy"):
        print("==> Loading saved mean and std..")  
        mean = np.load("data/mean.npy")
        std = np.load("data/std.npy")
    else:
        mean, std = get_image_means_stds("data/train")

    # Hyperparameters
    batch_size = 8
    image_size = 224
    learning_rate = 0.001
    num_epochs = 20
    momentum = 0.9
    weight_decay = 1e-4

    # Create datasets
    print("\n==> Loading datasets..")
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=weighted_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Make the model and loss functions
    print("\n==> Initializing model, loss function, and optimizer..")
    simple_model = customCNN().to(device)
    summary(simple_model, input_size=(3, image_size, image_size))

    criterion = nn.CrossEntropyLoss() # label_smoothing=0.05
    optimizer = optim.SGD(simple_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1) Possibly add a learning rate scheduler later

    print("Loader len:", len(train_loader))
    x, y = next(iter(train_loader))
    print("First batch shapes:", x.shape, y.shape)
    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train(simple_model, train_loader, criterion, optimizer, epochs=num_epochs, device=device)
    evaluation(simple_model, val_loader, verbose=True, device=device)

    # Save the model
    SIMPLECNN_MODEL_PATH = 'data/customcnn_model.pth'
    torch.save(simple_model.state_dict(), SIMPLECNN_MODEL_PATH)
    print(f'\n==> Saved trained model to {SIMPLECNN_MODEL_PATH}')

    # plot training and validation accuracy/loss curves
    epochs_range = range(num_epochs)

    plt.figure(figsize=(12,5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label="Train")
    plt.plot(epochs_range, val_accuracies, label="Validation")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()