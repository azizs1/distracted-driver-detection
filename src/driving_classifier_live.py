import cv2
import torch
import argparse
import numpy as np
from PIL import Image

from datasets import get_transforms, get_image_means_stds, load_datasets
from train import pick_device
from results import load_model

def preprocess_frame(frame, transform, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    tensor = transform(frame).unsqueeze(0)
    return tensor.to(device, non_blocking=True)
# def preprocess_frame(frame, transform, device):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#     frame = Image.fromarray(frame)
#     tensor = transform(frame).unsqueeze(0)
#     return tensor.to(device)


@torch.no_grad()
def predict(model, frame, transform, class_names, device):
    x = preprocess_frame(frame, transform, device)
    out = model(x)
    probs = torch.softmax(out, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return class_names[idx.item()], conf.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--arch", type=str, choices=["custom", "resnet"], required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=str)
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--out", type=str, default="output.mp4", help="Output video path")
    args = parser.parse_args()
    
    device = pick_device()

    if args.arch == "custom":
        mean, std = get_image_means_stds("data/train")
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    transform = get_transforms(mean=mean, std=std, augment=False)

    _, _, test_loader = load_datasets(batch_size=1, image_size=224, mean=mean, std=std, num_workers=0,)
    CLASS_NAMES = test_loader.dataset.classes

    num_classes = len(CLASS_NAMES)
    model = load_model(args.arch, args.model, num_classes, device)

    # img = cv2.imread("data/train/gA_1_s1_ir_face_mp4-1_jpg.rf.054a08cc8325b4f67b3c254e05aae9d2.jpg")
    # img = cv2.imread("data/train/gA_1_s1_ir_face_mp4-54_jpg.rf.21afb5679bc2769946967554c0dff808.jpg")
    # label, conf = predict(model, img, transform, CLASS_NAMES, device)
    # print("Prediction:", label, "confidence:", conf)
    # from train import evaluation

    # val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluation(
    #     model, test_loader, verbose=True, device=device
    # )
    # print("Sanity check on test set:",
    #     "Acc", val_acc, "Prec", val_prec, "Rec", val_rec, "F1", val_f1)

    if args.live:
        print("Starting live video feed...")
        cap = cv2.VideoCapture(args.camera)
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    else:
        print(f"Opening video file {args.video}...")
        cap = cv2.VideoCapture(args.video)

    writer = None

    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

        print(f"Saving output video to {args.out}")

    print("Quit by pressing 'q'.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict(model, frame, transform, CLASS_NAMES, device)

        if label == "SafeDriving":
            color = (0, 255, 0)
        elif label in ["SleepyDriving", "Yawn"]:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA,)
        if args.save:
            writer.write(frame)
        cv2.imshow("Driving Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
