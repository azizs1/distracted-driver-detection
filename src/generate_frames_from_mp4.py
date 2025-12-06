import os
import cv2
import shutil
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mp4', required=True)
    args = p.parse_args()

    dst_dir = f"data/vid_frames_{Path(args.mp4).stem}"

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    os.makedirs(dst_dir)
    capture = cv2.VideoCapture(args.mp4)

    frame_count = 0
    while True:
        success, frame = capture.read()

        if not success:
            break

        cv2.imwrite(os.path.join(dst_dir, f"frame_{frame_count}.png"), frame)
        frame_count += 1
    
    print(f"Created frames in {dst_dir}")

if __name__ == '__main__':
    main()