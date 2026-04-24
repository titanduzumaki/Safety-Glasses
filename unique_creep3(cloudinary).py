from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

import cv2
import time
import numpy as np
import os
import cloudinary
import cloudinary.uploader


# =========================
# 🔹 CLOUDINARY CONFIG (EDIT THIS)
# =========================
cloudinary.config(
    cloud_name="dpvifpivf",
    api_key="958488816558334",
    api_secret="wv9HDjNkGO6yHGlupEnrZ6TsCsw"
)


people_db = {}
PERSON_ID = 0
last_upload_time = {}

MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# 🔥 detect multiple faces
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=5   # 👈 increase (try 5–10)
)

detector = vision.FaceLandmarker.create_from_options(options)


# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

print("Webcam started")

cv2.namedWindow("Smart Gaze System", cv2.WINDOW_NORMAL)


while True:
    try:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (800, 500))
        frame = cv2.flip(frame, 1)

        panel = np.zeros((500, 300, 3), dtype=np.uint8)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                # =========================
                # FACE CENTER
                # =========================
                cx = int(face_landmarks[1].x * w)
                cy = int(face_landmarks[1].y * h)

                # =========================
                # PERSON TRACKING
                # =========================
                matched_id = None

                for pid, data in people_db.items():
                    px, py = data["pos"]

                    if abs(cx - px) < 60 and abs(cy - py) < 60:
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

                # =========================
                # 🔥 BETTER FACE CROP (using landmarks)
                # =========================
                xs = [int(lm.x * w) for lm in face_landmarks]
                ys = [int(lm.y * h) for lm in face_landmarks]

                x1 = max(0, min(xs) - 20)
                y1 = max(0, min(ys) - 20)
                x2 = min(w, max(xs) + 20)
                y2 = min(h, max(ys) + 20)

                face_crop = frame[y1:y2, x1:x2]

                # =========================
                # 🔹 CLOUDINARY UPLOAD (controlled)
                # =========================
                current_time = time.time()

                if matched_id not in last_upload_time:
                    last_upload_time[matched_id] = 0

                # upload every 5 sec per person
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

                # =========================
                # DRAW LANDMARKS
                # =========================
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), 1, (0,255,0), -1)

                # =========================
                # GAZE LOGIC
                # =========================
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

                # ALERT
                if count > 10:
                    label = f"ID {matched_id}: CREEP ALERT!"
                    color = (0,0,255)
                else:
                    label = f"ID {matched_id}: Looks {count}"
                    color = (0,255,0)

                cv2.putText(frame, label,
                            (cx, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(panel, f"Person {matched_id}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.putText(panel, f"Looks: {count}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        else:
            cv2.putText(panel, "No Face Detected",
                        (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)

        combined = np.hstack((frame, panel))
        cv2.imshow("Smart Gaze System", combined)

        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("ERROR:", e)
        continue


cap.release()
cv2.destroyAllWindows()