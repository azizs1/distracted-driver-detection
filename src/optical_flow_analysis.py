import numpy as np
import cv2
import tqdm
import os
import sys
import json
import argparse
import mediapipe as mp
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

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

            # # the following code just saves images with boxes on faces for debug purposes
            # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            # cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 4)

            # facebox_dir = os.path.join(DATA_DIR, "faceboxes")
            # print(facebox_dir)
            # os.makedirs(facebox_dir, exist_ok=True)
            # out_path = os.path.join(facebox_dir, "facebox_" + img_file)
            # cv2.imwrite(out_path, img_color)

    # dump dict into JSON file for debug
    out_json = os.path.join(frames_dir, "1faces_debug.json")
    with open(out_json, "w") as f:
        json.dump(faces_dict, f, indent=4)

    return faces_dict

# Based on https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
def detect_faces_mediapipe(frames_dir):
    def pad_laterally(box, scale_factor):
        x, y, w, h = box

        new_w = int(scale_factor*w)
        new_x = int(x-0.5*(new_w-w))

        return (new_x, y, new_w, h)
    
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
                
                # pick the largest bounding box
                biggest_box = max(all_boxes, key=lambda f: f[2]*f[3])

                # in order to better capture behaviors like drinking water and taking calls, its better to pad this
                # on the sides
                # biggest_box = pad_laterally(biggest_box, 2)
                faces_dict[img_file] = biggest_box

                # draw green rectangle to debug
                (x, y, bw, bh) = biggest_box
                cv2.rectangle(img_rgb, (x, y), (x+bw, y+bh), (0, 255, 0), 4)

                facebox_dir = os.path.join(DATA_DIR, "faceboxes")
                os.makedirs(facebox_dir, exist_ok=True)
                out_path = os.path.join(facebox_dir, "facebox_" + img_file)
                cv2.imwrite(out_path, img_rgb)

    # dump dict into JSON file to cache
    with open(os.path.join(frames_dir, "faces_dict.json"), "w") as f:
        json.dump(faces_dict, f, indent=4)

    return faces_dict

# https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
# https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
def compute_optical_flow(frames_dir, faces_dict, alg="lkt-mp"):
    # the valid options for this func are farneback, lkt, and lkt-mp
    # lkt-mp uses mediapipes face mesh to find features to track
    flow_maps = {}
    prev_name = ""
    prev_img = None

    if alg == "farneback":
        for file_name in tqdm.tqdm(sorted(faces_dict.keys(), key=lambda x: int(Path(x).stem.replace("frame_", ""))), desc="Processing frames"):
            if faces_dict[file_name] is not None:
                img = cv2.imread(os.path.join(frames_dir, file_name), cv2.IMREAD_GRAYSCALE)
                if prev_img is not None:
                    # trying to figure out sweet spot between accuracy and speed
                    op_flow = cv2.calcOpticalFlowFarneback(prev_img, img, None, pyr_scale=0.5, levels=3, winsize=10, 
                                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                    
                    x, y, w, h = faces_dict[file_name]
                    mask = np.zeros_like(op_flow[...,0], dtype=np.uint8)
                    mask[y:y+h, x:x+w] = 1
                    flow_roi = op_flow*mask[...,None]

                    flow_maps[f"{prev_name},{file_name}"] = flow_roi

                    # draw imgs to debug
                    # the hue indicates direction
                    # angle 0 to the right is red
                    # angle 90 to down is green
                    # angle 180 to left is blue
                    # angle 270 to up is purple
                    mag, ang = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1])
                    hsv = np.zeros((flow_roi.shape[0], flow_roi.shape[1], 3), dtype=np.uint8)
                    hsv[..., 0] = ang*180/np.pi/2
                    hsv[..., 1] = 255
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    farneback_dir = os.path.join(DATA_DIR, "op_flow_farneback")
                    os.makedirs(farneback_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(farneback_dir, 
                                                f"op_flow_{alg}_{Path(prev_name).stem}_{Path(file_name).stem}.png"), bgr)
            prev_name = file_name
            prev_img = img
    elif alg == "lkt":
        for file_name in tqdm.tqdm(sorted(faces_dict.keys(), key=lambda x: int(Path(x).stem.replace("frame_", ""))), desc="Processing frames"):
            if faces_dict[file_name] is not None:
                img = cv2.imread(os.path.join(frames_dir, file_name), cv2.IMREAD_GRAYSCALE)
                if prev_img is not None:
                    x, y, w, h = faces_dict[file_name]
                    roi_prev = prev_img[y:y+h, x:x+w]
                    roi = img[y:y+h, x:x+w]

                    p0 = cv2.goodFeaturesToTrack(roi_prev, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                    if p0 is not None:
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(roi_prev, roi, p0, None, winSize=(15,15), maxLevel=2,
                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                        
                        if p1 is not None:
                            good_new = p1[st==1]
                            good_old = p0[st==1]

                        # store displacements
                        displacements = good_new-good_old
                        flow_maps[f"{prev_name},{file_name}"] = displacements

                        debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                        for (new, old) in zip(good_new, good_old):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            debug_img = cv2.line(debug_img, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
                            debug_img = cv2.circle(debug_img, (int(a), int(b)), 3, (0,0,255), -1)

                        lkt_dir = os.path.join(DATA_DIR, "op_flow_lkt")
                        os.makedirs(lkt_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(lkt_dir, f"op_flow_{alg}_{Path(prev_name).stem}_{Path(file_name).stem}.png"), debug_img)
            prev_name = file_name
            prev_img = img
    elif alg == "lkt-mp":
        # https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        for file_name in tqdm.tqdm(sorted(faces_dict.keys(), key=lambda x: int(Path(x).stem.replace("frame_", ""))), desc="Processing frames"):
            if faces_dict[file_name] is not None:
                img = cv2.imread(os.path.join(frames_dir, file_name))
                if prev_img is not None:
                    x, y, w, h = faces_dict[file_name]
                    roi_prev = prev_img[y:y+h, x:x+w]
                    roi = img[y:y+h, x:x+w]

                    rgb_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2RGB)
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                    results_prev = face_mesh.process(rgb_prev)
                    results = face_mesh.process(rgb)

                    if results_prev.multi_face_landmarks and results.multi_face_landmarks:
                        lm_prev = results_prev.multi_face_landmarks[0].landmark
                        h_roi, w_roi, _ = roi.shape

                        p0 = np.array([[lm.x*w_roi, lm.y*h_roi] for lm in lm_prev], dtype=np.float32)
                        p0 = p0.reshape(-1,1,2)

                        p1, st, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY), 
                                                            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), p0, None, winSize=(15,15), maxLevel=2,
                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                        
                        if p1 is not None:
                            good_new = p1[st==1]
                            good_old = p0[st==1]

                        # store displacements
                        flow_maps[f"{prev_name},{file_name}"] = good_new-good_old

                        # # imgs for debug
                        # for (new, old) in zip(good_new, good_old):
                        #     a, b = new.ravel()
                        #     c, d = old.ravel()
                        #     a_full, b_full = int(a + x), int(b + y)
                        #     c_full, d_full = int(c + x), int(d + y)
                        #     img = cv2.line(img, (a_full, b_full), (c_full, d_full), (0,255,0), 2)
                        #     img = cv2.circle(img, (a_full, b_full), 3, (0,0,255), -1)

                        # lkt_dir = os.path.join(DATA_DIR, "op_flow_lkt_mp")
                        # os.makedirs(lkt_dir, exist_ok=True)
                        # cv2.imwrite(os.path.join(lkt_dir, f"op_flow_{alg}_{Path(prev_name).stem}_{Path(file_name).stem}.png"), img)
            prev_name = file_name
            prev_img = img

    # save it in data directory for quicker runs later
    np.savez(os.path.join(DATA_DIR, "flow_maps.npz"), **flow_maps)

    return flow_maps

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

        if os.path.exists(os.path.join(args.frames_dir, "faces_dict.json")):
            with open(os.path.join(args.frames_dir, "faces_dict.json"), 'r') as file:
                faces_dict = json.load(file)
        else:
            # faces_dict = detect_faces(args.frames_dir)
            faces_dict = detect_faces_mediapipe(args.frames_dir)

        compute_optical_flow(args.frames_dir, faces_dict)
        print(len(faces_dict))
        # data = np.load(out_path)
        # flow = data["frame1->frame2"]
    pass

if __name__ == '__main__':
    main()