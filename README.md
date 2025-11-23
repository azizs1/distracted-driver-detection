# Detecting Distracted Driving Using Computer Vision Techniques
**Authors:** Aziz Shaik and Rayden Dodd  
**Course:** ECE 5554: Computer Vision (Fall 2025)

This project explores the use of computer vision techniques to detect distracted driving behaviors in real time. By leveraging deep learning models trained on labeled image data, we aim to classify driver states such as safe, distracted, sleepy, or engaging in risky behaviors.

## How to Use
### Initial Setup
1. Add the needed data into the data folder. There should be 3 folders for test, train, and validate with images in each, along with annotation files.
2. Run `uv sync` to download any dependencies within a virtual environment.
>NOTE: When running Python files, you must use `uv run python sample.py`.

## Acknowledgments
- We use the [Driver Inattention Detection Dataset](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset) from Kaggle, created by Zeyad Mashhour, for training our model.
