# src/results.py - script to evaluate trained models and print results compared to the TEST dataset
import numpy as np
import torch
from train import customCNN, evaluation, pick_device
from train_resnet import ModifiedResNet
from datasets import load_datasets, get_image_means_stds

def load_model(arch, model_path, num_classes, device):
    print(f"==> Loading model {arch} from {model_path}...")
    if arch == "custom":
        model = customCNN(num_classes)
    elif arch == "resnet":
        model = ModifiedResNet(num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

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
    num_classes_custom = len(set(test_loader.dataset.classes))
    num_classes_resnet = len(set(test_loader_res.dataset.classes))

    custom_model = load_model("custom", CUSTOMCNN_MODEL_PATH, num_classes_custom, device)
    resnet_model = load_model("resnet", MODIFIEDRESNET_MODEL_PATH, num_classes_resnet, device)
    
    evaluation(custom_model, test_loader, verbose=True, device=device)
    evaluation(resnet_model, test_loader_res, verbose=True, device=device)


if __name__ == "__main__":
    main()  