# Detecting Distracted Driving Using Computer Vision Techniques
**Authors:** Aziz Shaik and Rayden Dodd  
**Course:** ECE 5554: Computer Vision (Fall 2025)

This project explores the use of computer vision techniques to detect distracted driving behaviors in real time. By leveraging deep learning models trained on labeled image data, we aim to classify driver states such as safe, distracted, sleepy, or engaging in risky behaviors.

## How to Use
### Initial Setup
1. Add the needed data into the data folder. There should be 3 folders for test, train, and validate with images in each, along with annotation files.
2. Run `uv sync` to download any dependencies within a virtual environment.
>NOTE: When running Python files, you must use `uv run python sample.py`.

### Deep Learning

### Optical Flow
The files for this section are `optical_flow_analysis.py` and `optical_flow_live.py`, with the former being the only one that needs to be run. Ensure that this file is run from the base directory.

#### Running Live
From the base directory, run with `uv run python ./src/optical_flow_analysis.py --live`. This should run without problem if using Ubuntu, but WSL will present some issues. With WSL:
1. Ensure that WSL2 is being used with `wsl --version`.
2. Running PS as admin, installed `usbipd`: `winget install --id Microsoft.usbipd-win`
3. Using `usbipd list`, find the BUSID for the desired input camera and bind this to WSL with `usbipd bind --busid <BUSID>`
4. Attach to WSL: `usbipd attach --busid <BUSID> --wsl <DIST>`. Use `wsl --list` to see active distribution.
5. With the camera now attached to WSL, confirm that the web camera is accessible in WSL with `ls /dev/video*`. You should have results appear (i.e. /dev/video0, /dev/video1, /dev/video2, /dev/video3).

>NOTE: If the camera is disconnected, you do not need to bind again, but you will need to attach again.
 
#### Running with an MP4
Simply run `uv run python ./src/optical_flow_analysis.py --frames_dir <FRAMES_DIR>`. This assumes that there is a directory that already exists with the MP4 split up into frames. The `src/generate_frames_from_mp4.py` file can help with this.

## Acknowledgments
- We use the [Driver Inattention Detection Dataset](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset) from Kaggle, created by Zeyad Mashhour, for training our model.
