# src/results.py - script to evaluate trained models and print results compared to the TEST dataset
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
    mean_custom, std_custom = get_image_means_stds("data/train")

    # Hyperparameters
    batch_size = 16
    image_size = 224
    num_workers = 4

    # Load datasets
    _, _, test_loader = load_datasets(batch_size=batch_size, image_size=image_size, mean=mean_custom, std=std_custom, num_workers=num_workers)
    _, _, test_loader_res = load_datasets(batch_size=batch_size, image_size=image_size, mean=mean_res, std=std_res, num_workers=num_workers)


    CUSTOMCNN_MODEL_PATH = 'data/customcnn_model_927_nocoloraug_guassian.pth'
    MODIFIEDRESNET_MODEL_PATH = 'data/modifiedresnet_model_96_adam.pth'

    # Load models
    custom_model = customCNN(len(set(test_loader.dataset.classes))).to(device)
    custom_model.load_state_dict(torch.load(CUSTOMCNN_MODEL_PATH))

    resnet_model = ModifiedResNet(len(set(test_loader_res.dataset.classes))).to(device)
    resnet_model.load_state_dict(torch.load(MODIFIEDRESNET_MODEL_PATH))
    
    evaluation(custom_model, test_loader, verbose=True, device=device)
    evaluation(resnet_model, test_loader_res, verbose=True, device=device)


if __name__ == "__main__":
    main()  