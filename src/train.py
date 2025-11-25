# src/train.py - script to train a model on the dataset
import os
import numpy as np
import torch
from datasets import CustomDataset, get_transforms, get_image_means_stds
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
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
        self.adapt = nn.AdaptiveAvgPool2d((14, 14))
        # self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=128 * 14 * 14, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        '''Defines graph of connections for each layer in customCNN.'''
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(model, loader, criterion, optimizer, epochs=2, device="cpu"):
  '''Train a model from training data.

  Args:
    - model: Neural network to train
    - epochs: Number of epochs to train the model
    - loader: Dataloader to train the model with
  '''
  print('Start Training')

  for epoch in range(epochs):  # loop over the dataset multiple times

      running_loss = 0.0
      total = 0
      correct = 0
      tqdm_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
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

  print('\nFinished Training')

def evaluation(model, loader, device="cpu"):
  '''Evaluate a model and output its accuracy on a test dataset.

  Args:
    - model: Neural network to evaluate
    - loader: Dataloader containing test dataset
  '''
  # Evaluate accuracy on validation / test set
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in tqdm(loader):
          images, labels = data
          images = images.to(device, non_blocking=True)
          labels = labels.to(device, non_blocking=True)
          # calculate outputs by running images through the network
          outputs = model(images)
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print(f'\nAccuracy of the network on the {len(loader)} validation images: {100 * correct // total} %')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if possible
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
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
    num_epochs = 10
    momentum = 0.9
    weight_decay = 1e-4

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=weighted_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("\n==> Initializing model, loss function, and optimizer..")
    simple_model = customCNN().to(device)
    summary(simple_model, input_size=(3, image_size, image_size))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(simple_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    print("Loader len:", len(train_loader))
    x, y = next(iter(train_loader))
    print("First batch shapes:", x.shape, y.shape)
    train(simple_model, train_loader, criterion, optimizer, epochs=num_epochs, device=device)
    evaluation(simple_model, val_loader, device=device)

    SIMPLECNN_MODEL_PATH = 'data/customcnn_model.pth'
    torch.save(simple_model.state_dict(), SIMPLECNN_MODEL_PATH)

if __name__ == "__main__":
    main()