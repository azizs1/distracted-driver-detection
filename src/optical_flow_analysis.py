import numpy as np
import cv2
import csv
import os
import sys
import json
import argparse
import mediapipe as mp

# This face detection was inspired by https://www.datacamp.com/tutorial/face-detection-python-opencv
# This uses Haar cascades since these are much faster. take a look at mediapipe face detection if we need better accuracy later
def detect_faces(frames_dir):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces_dict = {} # key is frame name, val is (x, y, w, h)

    for img_file in os.listdir(frames_dir):
        img = cv2.imread(os.path.join(frames_dir, img_file), cv2.IMREAD_GRAYSCALE)

        faces = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(250, 250))

        if len(faces) == 0:
            # just skip this loop if we didn't find any faces (usually happens when face is turned)
            faces_dict[img_file] = None
        else:
            # sometimes it says there are multiple faces. pick the largest one
            faces_dict[img_file] = max(faces, key=lambda f: f[2]*f[3]).tolist()

            # the following code just saves images with boxes on faces for debug purposes
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 4)

            out_path = os.path.join(frames_dir, "facebox_" + img_file)
            cv2.imwrite(out_path, img_color)


    # dump dict into JSON file for debug
    out_json = os.path.join(frames_dir, "1faces_debug.json")
    with open(out_json, "w") as f:
        json.dump(faces_dict, f, indent=4)

    return faces_dict

# Based on https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
def detect_faces_mediapipe(frames_dir):
    faces_dict = {} # key is frame name, val is (x, y, w, h)

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        for img_file in os.listdir(frames_dir):
            img = cv2.imread(os.path.join(frames_dir, img_file), cv2.IMREAD_GRAYSCALE)

            h, w = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.process(img_rgb)

            if not results.detections:
                faces_dict[img_file] = None
            else:
                all_boxes = []
                for detection in results.detections:
                    rbb = detection.location_data.relative_bounding_box
                    all_boxes.append((int(rbb.xmin*w), int(rbb.ymin*h), int(rbb.width*w), int(rbb.height*h)))
                
                biggest_box = max(all_boxes, key=lambda f: f[2]*f[3])
                faces_dict[img_file] = biggest_box

                # # draw green rectangle to debug
                # (x, y, bw, bh) = biggest_box
                # cv2.rectangle(img_rgb, (x, y), (x+bw, y+bh), (0, 255, 0), 4)

                # out_path = os.path.join(frames_dir, "facebox_" + img_file)
                # cv2.imwrite(out_path, img_rgb)

    # dump dict into JSON file to cache
    with open(os.path.join(frames_dir, "faces_dict.json"), "w") as f:
        json.dump(faces_dict, f, indent=4)

    return faces_dict

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--live', action='store_true', default=False)
    p.add_argument('--frames_dir')
    args = p.parse_args()

    if (args.live):
        # work on this component once we add live streaming
        sys.exit()
    else:
        
        if not args.frames_dir:
            print("gotta have frames_dir if not live")
            sys.exit()

        # detect_faces(args.frames_dir)
        if os.path.exists(os.path.join(args.frames_dir, "faces_dict.json")):
            with open(os.path.join(args.frames_dir, "faces_dict.json"), 'r') as file:
                faces_dict = json.load(file)
        else:
            faces_dict = detect_faces_mediapipe(args.frames_dir)

        print(len(faces_dict))
    pass

if __name__ == '__main__':
    main()