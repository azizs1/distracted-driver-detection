# This file just contains the same functions as optical_flow_analysis, except for single frames.
# completely for live analysis.

import os
import cv2
import mediapipe as mp
import numpy as np

detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

prev_img = None
distracted_count = 0
closed_count = 0

def detect_distractions_live(frame):
    global prev_img, distracted_count, closed_count
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(img_rgb)

    if results.detections:
        # FACE DETECTION ---------------------------------------------------------
        all_boxes = []
        for detection in results.detections:
            rbb = detection.location_data.relative_bounding_box
            all_boxes.append((int(rbb.xmin*w), int(rbb.ymin*h), int(rbb.width*w), int(rbb.height*h)))
        
        # pick the largest bounding box
        x, y, bw, bh = max(all_boxes, key=lambda f: f[2]*f[3])

        # draw green rectangle to debug
        # cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 4)
        # FACE DETECTION ---------------------------------------------------------

        # OPTICAL FLOW -----------------------------------------------------------
        transition_state = ""
        if prev_img is not None:
            roi_prev = prev_img[y:y+bh, x:x+bw]
            roi = frame[y:y+bh, x:x+bw]

            if roi_prev.shape == roi.shape and roi.size > 0:
                rgb_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2RGB)
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                results_prev = face_mesh.process(rgb_prev)
                results_curr = face_mesh.process(rgb)

                if results_prev.multi_face_landmarks and results_curr.multi_face_landmarks:
                    h_roi, w_roi, _ = roi.shape
                    lm_prev = results_prev.multi_face_landmarks[0].landmark
                    p0 = np.array([[lm.x*w_roi, lm.y*h_roi] for lm in lm_prev], dtype=np.float32).reshape(-1, 1, 2)

                    p1, st, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                                                         p0, None, winSize=(15,15), maxLevel=2, 
                                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    if p1 is not None:
                        good_new = p1[st==1]
                        good_old = p0[st==1]
                        flow = good_new-good_old
                        mean_dx = np.mean(flow[..., 0])
                        mean_dy = np.mean(flow[..., 1])
                        dx_thresh = 3
                        dy_thresh = 5

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
                    # debug
                    #     for (new, old) in zip(good_new, good_old):
                    #         a, b = new.ravel()
                    #         c, d = old.ravel()
                    #         cv2.line(frame, (int(a+x), int(b+y)), (int(c+x), int(d+y)), (0,255,0), 2)
                    #         cv2.circle(frame, (int(a+x), int(b+y)), 3, (0,0,255), -1)
        # OPTICAL FLOW -----------------------------------------------------------
        # DISTRACTION DETECTION --------------------------------------------------
        # if this doesn't get clamped, it will crash if face goes out of frame
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w-x)
        bh = min(bh, h-y)

        roi_rgb = cv2.cvtColor(frame[y:y+bh, x:x+bw], cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(roi_rgb)
        if mesh_results.multi_face_landmarks:
            h_roi, w_roi, _ = roi_rgb.shape
            lms = [(int(lm.x*w_roi)+x, int(lm.y*h_roi)+y) for lm in mesh_results.multi_face_landmarks[0].landmark]

            left_eye_indices = [33,159,158,133,153,145]
            right_eye_indices = [362,380,374,263,386,385]

            def ear(eye):
                p1, p2, p3, p4, p5, p6 = eye
                return (np.linalg.norm(np.array(p2)-np.array(p6)) + np.linalg.norm(np.array(p3)-np.array(p5))) / \
                    (2.0*np.linalg.norm(np.array(p1)-np.array(p4)))
        
            left_eye = [lms[i] for i in left_eye_indices]
            right_eye = [lms[i] for i in right_eye_indices]
            ear_val = (ear(left_eye)+ear(right_eye))/2.0

            eye_state = "open"
            if ear_val < 0.2:
                closed_count += 1
                if closed_count > 9:
                    eye_state = "closed"
                else:
                    eye_state = "closed (blink?)"
            else:
                closed_count = 0
                eye_state = "open"

            # here we do the yaw and pitch of the entire face
            nose = lms[1]
            left_eye_corner = lms[33]
            right_eye_corner = lms[263]
            yaw_thresh = 0.1
            pitch_thresh = 0.6
            
            dx = nose[0]-(left_eye_corner[0]+right_eye_corner[0])/2
            dy = nose[1]-(left_eye_corner[1]+right_eye_corner[1])/2
            eye_dist = max(1e-6, right_eye_corner[0]-left_eye_corner[0])
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

            focus_state="good"
            if head_state == "straight" and eye_state in ("open","closed (blink?)"):
                if transition_state in ("turning left","turning right","looking down","looking up"):
                    distracted_count += 1
                    focus_state = "careful"
                else:
                    distracted_count = 0
                    focus_state = "good"
            elif head_state != "straight":
                distracted_count += 1
                focus_state = "careful"
            
            if distracted_count > 30:
                focus_state = "distracted"

            # overlay some current stats with final decision
            cv2.putText(frame, f"Eyes: {eye_state}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Head: {head_state}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{transition_state}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            if focus_state == "good":
                cv2.putText(frame, f"{focus_state}", (50, frame.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            elif focus_state == "careful":
                cv2.putText(frame, f"{focus_state}", (50, frame.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"{focus_state}", (50, frame.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # DISTRACTION DETECTION --------------------------------------------------

    # update previous frame
    prev_img = frame.copy()
    return frame