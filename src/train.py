# src/train.py - script to train a model on the dataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from datasets import get_image_means_stds, load_datasets
from torch import nn
from torch import optim
from tqdm import tqdm
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class customCNN(nn.Module):
    '''Custom CNN neural network. See above description for what each layer does.'''

    def __init__(self, num_classes=6):
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
        self.fc1 = nn.Linear(in_features=128 * 14 * 14, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        '''Defines graph of connections for each layer in customCNN.'''
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Code from Homework 4 adapted for training and evaluation
def train(model, loader, validation, criterion, optimizer, epochs=2, device="cpu", save_path="data/customcnn_model.pth"):
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
    best_val_acc = 0.0
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        total = 0
        correct = 0
        model.train()
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
        # Save training loss over epochs
        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Test what validation loss is at end of epoch
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _  = evaluation(model, validation, criterion=criterion, verbose=False, device=device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        end_time = time.time()
        epoch_time = end_time - start_time
        tqdm_bar.write(f"Epoch {epoch+1} | took {epoch_time:.2f} seconds | Val Acc: {val_acc:.4f}| Val Loss: {val_loss:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")

        # Save the best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print('\nFinished Training')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print(f'Model saved to: {save_path}')
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
    all_preds = []
    all_labels = []
    prev_mode = model.training
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(loader, disable=not verbose):
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
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
    if prev_mode:
        model.train()
    # Concatenate collected predictions
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Metrics
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average="macro")
    val_recall = recall_score(all_labels, all_preds, average="macro")
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    val_loss = running_loss / len(loader)
    
    # Print out results
    if verbose:
        print(f"\nAccuracy:  {val_acc:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall:    {val_recall:.4f}")
        print(f"F1-score:  {val_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=loader.dataset.classes))

    return val_loss, val_acc, val_precision, val_recall, val_f1, all_preds, all_labels

def print_results(train_losses, train_accuracies, val_losses, val_accuracies):
    '''Print training and validation results after training.
    Args:
    - train_losses(list): List of training losses over epochs
    - train_accuracies(list): List of training accuracies over epochs
    - val_losses(list): List of validation losses over epochs
    - val_accuracies(list): List of validation accuracies over epochs
    '''
    # plot training and validation accuracy/loss curves
    epochs_range = range(len(train_losses))

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

def pick_device():
    '''Pick device to run model on (GPU if available, else CPU)'''
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if possible
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    return device

def main():
    # Device setup
    device = pick_device()

    # Load dataset mean and std
    mean, std = get_image_means_stds("data/train")

    # Hyperparameters
    batch_size = 16 # Tested for best epoch timing
    image_size = 224
    learning_rate = 0.001
    num_epochs = 20
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 4
    label_smoothing = 0.05 # help with noisy labels

    # Create datasets
    print("\n==> Loading datasets..")
    train_loader, val_loader, _ = load_datasets(batch_size=batch_size, image_size=image_size, mean=mean, std=std, num_workers=num_workers)

    # Make the model and loss functions
    print("\n==> Initializing model, loss function, and optimizer..")
    custom_model = customCNN(len(set(train_loader.dataset.classes))).to(device)
    summary(custom_model, input_size=(3, image_size, image_size))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(custom_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    print("Loader len:", len(train_loader))
    x, y = next(iter(train_loader))
    print("First batch shapes:", x.shape, y.shape)

    # Train the model
    CUSTOMCNN_MODEL_PATH = 'data/customcnn_model.pth'
    train_losses, train_accuracies, val_losses, val_accuracies = train(custom_model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs, device=device, save_path=CUSTOMCNN_MODEL_PATH)
    evaluation(custom_model, val_loader, verbose=True, device=device)

    # Print results
    print_results(train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == "__main__":
    main()