# src/train_resnet.py - script to train a resnet on the dataset
import numpy as np
import torch
from datasets import load_datasets
from train import train, evaluation, print_results, pick_device
from torch import nn
from torch import optim
from torchsummary import summary
from torchvision.models import ResNet50_Weights

# Source https://www.geeksforgeeks.org/deep-learning/how-to-implement-transfer-learning-in-pytorch/
class ModifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision', 'resnet50', weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Change the final fully connected layer for 10 classes
        # Only train the final fully connected layer
        for name, param in self.resnet.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
            else:
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
    batch_size = 16
    image_size = 224
    learning_rate_fc = 1e-4
    learning_rate_layer4 = 1e-5
    num_epochs = 20
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 4
    label_smoothing = 0.05

    # Create datasets
    print("\n==> Loading datasets..")
    train_loader, val_loader, test_loader = load_datasets(batch_size=batch_size, image_size=image_size, mean=mean, std=std, num_workers=num_workers)

    # model = models.resnet50(pretrained=True)
    print(f" Number of classes: {len(set(train_loader.dataset.classes))}")
    model = ModifiedResNet(len(set(train_loader.dataset.classes))).to(device)
    summary(model, input_size=(3, image_size, image_size))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) # label_smoothing=0.05
    optimizer = optim.Adam([{"params": model.resnet.fc.parameters(), "lr": learning_rate_fc}, {"params": model.resnet.layer4.parameters(), "lr": learning_rate_layer4},], weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1) Possibly add a learning rate scheduler later

    print("Loader len:", len(train_loader))
    x, y = next(iter(train_loader))
    print("First batch shapes:", x.shape, y.shape)

    # Train the model
    MODIFIEDRESNET_MODEL_PATH = 'data/modifiedresnet_model.pth'
    train_losses, train_accuracies, val_losses, val_accuracies = train(model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs, device=device, save_path=MODIFIEDRESNET_MODEL_PATH)
    evaluation(model, val_loader, verbose=True, device=device)

    # Print results
    print_results(train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == "__main__":
    main()