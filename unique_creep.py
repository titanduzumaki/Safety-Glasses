from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

import cv2
import time
import pyrealsense2 as rs
import numpy as np


people_db = {}
PERSON_ID = 0

MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

attention_score = 0

print("RealSense camera started")

cv2.namedWindow("Smart Gaze System", cv2.WINDOW_NORMAL)

while True:
    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.resize(frame, (800, 500))

        #  ALWAYS CREATE PANEL (FIXED)
        panel = np.zeros((500, 300, 3), dtype=np.uint8)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                h, w, _ = frame.shape
                
                # FACE CENTER
                cx = int(face_landmarks[1].x * w)
                cy = int(face_landmarks[1].y * h)

                #  MATCH PERSON (TRACKING)
                matched_id = None

                for pid, data in people_db.items():
                    px, py = data["pos"]

                    if abs(cx - px) < 50 and abs(cy - py) < 50:
                        matched_id = pid
                        break

                if matched_id is None:
                    
                    PERSON_ID += 1
                    matched_id = PERSON_ID

                    people_db[matched_id] = {
                        "pos": (cx, cy),
                        "timestamps": []
                    }

                # Update position
                people_db[matched_id]["pos"] = (cx, cy)

                # DEPTH AVERAGE (same as before)
                depth_values = []
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        px = cx + i
                        py = cy + j

                        if 0 <= px < w and 0 <= py < h:
                            d = depth_frame.get_distance(px, py)
                            if d > 0:
                                depth_values.append(d)

                distance = sum(depth_values) / len(depth_values) if depth_values else 0

                if distance < 0.5:
                    continue

                # DRAW LANDMARKS
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), 1, (0,255,0), -1)

                # EYE CORNERS
                lx1 = int(face_landmarks[33].x * w)
                lx2 = int(face_landmarks[133].x * w)

                # IRIS
                iris_x = int((
                    face_landmarks[468].x +
                    face_landmarks[469].x +
                    face_landmarks[470].x +
                    face_landmarks[471].x
                ) / 4 * w)

                eye_width = (lx2 - lx1)
                if eye_width == 0:
                    continue

                ratio = (iris_x - lx1) / eye_width

                if ratio < 0.35:
                    gaze = "LEFT"
                elif ratio > 0.65:
                    gaze = "RIGHT"
                else:
                    gaze = "CENTER"

                #  TRACK LOOK EVENTS
                if gaze == "CENTER":
                    current_time = time.time()
                    people_db[matched_id]["timestamps"].append(current_time)

                #  CLEAN OLD DATA (1 min)
                one_min_ago = time.time() - 60

                people_db[matched_id]["timestamps"] = [
                    t for t in people_db[matched_id]["timestamps"]
                    if t > one_min_ago
                ]

                count = len(people_db[matched_id]["timestamps"])

                #  CREEP DETECTION
                if count > 10:
                    label = f"ID {matched_id}: CREEP ALERT!"
                    color = (0,0,255)
                else:
                    label = f"ID {matched_id}: Looks {count}"
                    color = (0,255,0)

                #  DRAW LABEL ON FACE
                cv2.putText(frame, label,
                            (cx, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

                # PANEL UPDATE
                cv2.putText(panel, f"Person {matched_id}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.putText(panel, f"Looks: {count}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            #  NO FACE CASE
            cv2.putText(panel, "No Face Detected",
                        (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)

        #  COMBINE
        combined = np.hstack((frame, panel))
        cv2.imshow("Smart Gaze System", combined)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.01)

    except Exception as e:
        print("ERROR:", e)
        continue

pipeline.stop()
cv2.destroyAllWindows()