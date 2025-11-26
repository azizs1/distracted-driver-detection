# src/results.py - script to evaluate trained models and print results compared to the TEST dataset
import os
import numpy as np
import torch
from train import customCNN, evaluation, pick_device
from train_resnet import ModifiedResNet
from datasets import load_datasets, get_image_means_stds

def main():
    # Device setup
    device = pick_device()

    # Stats
    mean_res = np.array([0.485, 0.456, 0.406])
    std_res = np.array([0.229, 0.224, 0.225])
    mean, std = get_image_means_stds("data/train")

    # Hyperparameters
    batch_size = 16
    image_size = 224
    num_workers = 4

    # Load datasets
    _, _, test_loader = load_datasets(batch_size=batch_size, image_size=image_size, mean=mean, std=std, num_workers=num_workers)
    _, _, test_loader_res = load_datasets(batch_size=batch_size, image_size=image_size, mean=mean_res, std=std_res, num_workers=num_workers)


    SIMPLECNN_MODEL_PATH = 'data/customcnn_model_89.pth'
    MODIFIEDRESNET_MODEL_PATH = 'data/modifiedresnet_model.pth'

    # Load models
    simple_model = customCNN().to(device)
    simple_model.load_state_dict(torch.load(SIMPLECNN_MODEL_PATH))

    resnet_model = ModifiedResNet(len(set(test_loader_res.dataset.classes))).to(device)
    resnet_model.load_state_dict(torch.load(MODIFIEDRESNET_MODEL_PATH))
    
    evaluation(simple_model, test_loader, verbose=True, device=device)
    evaluation(resnet_model, test_loader_res, verbose=True, device=device)


if __name__ == "__main__":
    main()  