last_upload_time = {}

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

import cv2
import time
import pyrealsense2 as rs
import numpy as np
import os
import cloudinary
import cloudinary.uploader


cloudinary.config(
    cloud_name="",
    api_key="",
    api_secret=""
)

MODE = "IMAGE"   # "IMAGE" or "CAMERA"
IMAGE_FOLDER = "images"


people_db = {}
PERSON_ID = 0

MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=6
)

detector = vision.FaceLandmarker.create_from_options(options)

# =========================
# MODE HANDLER
# =========================

def process_frame(frame, panel, depth_frame=None):
    global people_db, PERSON_ID

    frame = cv2.resize(frame, (800, 500))
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.face_landmarks:

        for face_landmarks in result.face_landmarks:

            cx = int(face_landmarks[1].x * w)
            cy = int(face_landmarks[1].y * h)
            
            x1 = max(0, cx - 100)
            y1 = max(0, cy - 100)
            x2 = min(w, cx + 100)
            y2 = min(h, cy + 100)
            
            face_crop = frame[y1:y2, x1:x2]
            
            

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

            people_db[matched_id]["pos"] = (cx, cy)

            x1 = max(0, cx - 100)
            y1 = max(0, cy - 100)
            x2 = min(w, cx + 100)
            y2 = min(h, cy + 100)
            
            face_crop = frame[y1:y2, x1:x2]
            
            current_time = time.time()
            if matched_id not in last_upload_time:
                last_upload_time[matched_id] = 0
            
            if current_time - last_upload_time[matched_id] > 5:
                if face_crop.size != 0:
                    temp_file = f"temp_{matched_id}.jpg"
                    cv2.imwrite(temp_file, face_crop)
                    try:
                        response = cloudinary.uploader.upload(temp_file)
                        url = response["secure_url"]
                        print(f"Uploaded (ID {matched_id}):", url)
                        
                    except Exception as e:
                        print("Upload failed:", e)
                        
                    os.remove(temp_file)
                    last_upload_time[matched_id] = current_time

            # 🔹 SKIP depth if not available (IMAGE MODE)
            distance = 1
            if depth_frame is not None:
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

            # EYE LOGIC
            lx1 = int(face_landmarks[33].x * w)
            lx2 = int(face_landmarks[133].x * w)

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

            if gaze == "CENTER":
                people_db[matched_id]["timestamps"].append(time.time())

            one_min_ago = time.time() - 60

            people_db[matched_id]["timestamps"] = [
                t for t in people_db[matched_id]["timestamps"]
                if t > one_min_ago
            ]

            count = len(people_db[matched_id]["timestamps"])

            if count > 10:
                label = f"ID {matched_id}: CREEP ALERT!"
                color = (0,0,255)
            else:
                label = f"ID {matched_id}: Looks {count}"
                color = (0,255,0)

            cv2.putText(frame, label, (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(panel, f"Person {matched_id}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, f"Looks: {count}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    else:
        cv2.putText(panel, "No Face Detected", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return frame, panel

# =========================
# RUN MODES
# =========================

if MODE == "IMAGE":

    print("Running in IMAGE mode...")

    for filename in os.listdir(IMAGE_FOLDER):

        path = os.path.join(IMAGE_FOLDER, filename)

        frame = cv2.imread(path)

        if frame is None:
            continue

        panel = np.zeros((500, 300, 3), dtype=np.uint8)

        frame, panel = process_frame(frame, panel)

        combined = np.hstack((frame, panel))

        cv2.imshow("Smart Gaze System", combined)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


elif MODE == "CAMERA":

    print("Running in CAMERA mode...")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    while True:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            panel = np.zeros((500, 300, 3), dtype=np.uint8)

            frame, panel = process_frame(frame, panel, depth_frame)

            combined = np.hstack((frame, panel))
            cv2.imshow("Smart Gaze System", combined)

            if cv2.waitKey(1) == 27:
                break

        except Exception as e:
            print("ERROR:", e)

    pipeline.stop()
    cv2.destroyAllWindows()