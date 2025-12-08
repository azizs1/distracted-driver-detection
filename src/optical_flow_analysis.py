import numpy as np
import cv2
import tqdm
import os
import sys
import json
import re
import argparse
import mediapipe as mp
from pathlib import Path
import optical_flow_live

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
        lm_dict = {}
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
                        p0 = p0.reshape(-1, 1, 2)

                        p1, st, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY), 
                                                            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), p0, None, winSize=(15,15), maxLevel=2,
                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                        
                        if p1 is not None:
                            good_new = p1[st==1]
                            good_old = p0[st==1]

                        # store displacements
                        flow_maps[f"{prev_name}, {file_name}"] = good_new-good_old

                        # save landmarks for this frame
                        if "landmarks" not in flow_maps:
                            flow_maps["landmarks"] = {}
                        lm_curr = results.multi_face_landmarks[0].landmark
                        lm_dict[file_name] = [(int(lm.x*w_roi), int(lm.y*h_roi)) for lm in lm_curr]

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
    np.savez(os.path.join(DATA_DIR, "flow_lm.npz"), flow_maps=flow_maps, landmarks=lm_dict)

    return flow_maps, lm_dict

# The landmark indices for each eye for EAR were brought in from here:
# https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV
def detect_distractions(flow_maps, lm_dict):
    decisions = {}
    closed_seq = []
    prev_state = "straight"
    distracted_count = 0

    # lm indices for eyes
    left_eye_indices = [33, 159, 158, 133, 153, 145]
    right_eye_indices = [362, 380, 374, 263, 386, 385]

    # https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib
    def ear(eye):
        p1, p2, p3, p4, p5, p6 = eye
        return (np.linalg.norm(np.array(p2)-np.array(p6)) + np.linalg.norm(np.array(p3)-np.array(p5))) / \
               (2.0*np.linalg.norm(np.array(p1)-np.array(p4)))
    
    for frame_name, lms in lm_dict.items():
        # use eye aspect ratio for eye stuff
        left_eye = [lms[i] for i in left_eye_indices]
        right_eye = [lms[i] for i in right_eye_indices]
        ear_val = (ear(left_eye)+ear(right_eye))/2.0

        ear_thresh=0.2
        blink_frames=9 # this is mostly ok, may have to adjust?
        decisions.setdefault(frame_name, {})["eye_state"] = None

        if ear_val < ear_thresh:
            closed_seq.append(frame_name)
            state = "closed (blink?)"
            # if closure is longer than threshold, flip to closed
            if len(closed_seq) > blink_frames:
                state = "closed"

            decisions.setdefault(frame_name, {})["eye_state"] = state
        else:
            decisions.setdefault(frame_name, {})["eye_state"] = "open"
            closed_seq = []

        # https://medium.com/@abhishekjainindore24/different-ways-to-calculate-head-pose-estimation-ypr-yaw-pitch-and-roll-c3542bac03dc
        nose = lms[1]
        left_eye_corner = lms[33]
        right_eye_corner = lms[263]
        yaw_thresh = 0.1
        pitch_thresh = 0.6

        dx = nose[0]-(left_eye_corner[0]+right_eye_corner[0])/2
        dy = nose[1]-(left_eye_corner[1]+right_eye_corner[1])/2
        eye_dist = right_eye_corner[0]-left_eye_corner[0]
        yaw = dx/eye_dist
        pitch = dy/eye_dist

        if abs(yaw) < yaw_thresh and abs(pitch) < pitch_thresh:
            head_state = "straight"
        elif yaw > yaw_thresh:
            head_state = "right"
        elif yaw < -yaw_thresh:
            head_state = "left"
        elif pitch > pitch_thresh:
            head_state = "down"
        else:
            head_state = "idk"
        decisions.setdefault(frame_name, {})["transition_state"] = head_state

        flow_key = next((k for k in flow_maps.keys() if frame_name in k), None)
        if flow_key is not None:
            flow = flow_maps[flow_key]
            mean_dx = np.mean(flow[..., 0])
            mean_dy = np.mean(flow[..., 1])

            dx_thresh = 3
            dy_thresh = 5

            transition_state = ""
            if abs(mean_dx) > dx_thresh:
                if mean_dx > 0:
                    transition_state = "turning right"
                else:
                    transition_state = "turning left"
            elif abs(mean_dy) > dy_thresh:
                if mean_dy > 0:
                    transition_state = "looking down"
                else:
                    transition_state = "looking up"
            decisions[frame_name]["transition_state"] = transition_state
            print(f"{frame_name}, mean_dx:{mean_dx}, mean_dy:{mean_dy}, head_state:{head_state}, transition_state:{transition_state}")


        # make the decision here. add some smoothing so that it is distracted_maybe for at least 4-5 frames before becoming distracted
        distracted = True
        decisions.setdefault(frame_name, {})["focus_state"] = "good"

        if head_state == "straight" and decisions[frame_name]["eye_state"] in ("open", "closed (blink?)"):
            if transition_state in ("turning left", "turning right", "nodding down", "looking up"):
                distracted_count += 1
                decisions[frame_name]["focus_state"] = "careful"
            else:
                distracted_count = 0
                decisions[frame_name]["focus_state"] = "good"
        elif head_state != "straight":
            distracted_count += 1
            decisions[frame_name]["focus_state"] = "careful"
        
        if distracted_count > 3:
            decisions[frame_name]["focus_state"] = "distracted"

        print(f"{frame_name}, yaw:{yaw}, pitch:{pitch}, head_state:{head_state}, eye_state:{decisions[frame_name]["eye_state"]}, decision:{decisions[frame_name]["focus_state"]}")
        decisions.setdefault(frame_name, {})["head_state"] = head_state

    # # this loop is just for debug, to write imgs with the labels on it
    # for frame_name in lm_dict.keys():
    #     eye_state = decisions[frame_name].get("eye_state")
    #     head_state = decisions[frame_name].get("head_state")
    #     transition_state = decisions[frame_name].get("transition_state")
    #     focus_state = decisions[frame_name].get("focus_state")

    #     img_path = os.path.join(frames_dir, frame_name)
    #     img_rgb = cv2.imread(img_path)
    #     cv2.putText(img_rgb, f"Eyes: {eye_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    #     cv2.putText(img_rgb, f"Head: {head_state}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    #     cv2.putText(img_rgb, transition_state, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    #     cv2.putText(img_rgb, re.findall(r'\d+', frame_name)[0], (img_rgb.shape[1]-150, img_rgb.shape[0]-150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    #     if focus_state == "good":
    #         cv2.putText(img_rgb, f"{focus_state}", (50, img_rgb.shape[0]-150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 3, cv2.LINE_AA)
    #     elif focus_state == "careful":
    #         cv2.putText(img_rgb, f"{focus_state}", (50, img_rgb.shape[0]-150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 3, cv2.LINE_AA)
    #     else:
    #         cv2.putText(img_rgb, f"{focus_state}", (50, img_rgb.shape[0]-150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3, cv2.LINE_AA)

    #     final_dir = os.path.join(DATA_DIR, "final")
    #     os.makedirs(final_dir, exist_ok=True)
    #     out_path = os.path.join(final_dir, "final_" + frame_name)
    #     cv2.imwrite(out_path, img_rgb)

    with open(os.path.join(BASE_DIR, "decisions.json"), "w") as f:
        json.dump(decisions, f, indent=4)
    return decisions

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--live', action='store_true', default=False)
    p.add_argument('--frames_dir')
    args = p.parse_args()

    if (args.live):
        cap = cv2.VideoCapture(0)

        # have to set MJPG format otherwise it tries mpeg4 and doesnt work
        # use 'v4l2-ctl --device=/dev/video0 --list-formats-ext' to look at available formats for camera
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = optical_flow_live.detect_distractions_live(frame)
            cv2.imshow("camera", frame)
            # quit on q like ffmpeg
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
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

        if os.path.exists(os.path.join(DATA_DIR, "flow_lm.npz")):
            data = np.load(os.path.join(DATA_DIR, "flow_lm.npz"), allow_pickle=True)
            flow_maps = data["flow_maps"].item()
            lm_dict   = data["landmarks"].item()
        else:
            flow_maps, lm_dict = compute_optical_flow(args.frames_dir, faces_dict)

        detect_distractions(flow_maps, lm_dict)

        print(len(faces_dict))
    pass

if __name__ == '__main__':
    main()