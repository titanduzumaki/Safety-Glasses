from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat  # 🔥 FIXED

import cv2
import time

MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(2)

looking_frames = 0
NOT_LOOKING_THRESHOLD = 5
attention_score = 0

print("Camera opened:", cap.isOpened())

cv2.namedWindow("Face Landmarks", cv2.WINDOW_NORMAL)

print("Starting loop...")



while True:
    try:
        ret, frame = cap.read()
        print("Loop running... Frame:", ret)

        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #  FINAL FIX
        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if result.face_landmarks:
            print("Face detected")

            for face_landmarks in result.face_landmarks:

                h, w, _ = frame.shape
                
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0,255,0), -1)

                # LEFT EYE CORNERS
                lx1 = int(face_landmarks[33].x * w)
                ly1 = int(face_landmarks[33].y * h)

                lx2 = int(face_landmarks[133].x * w)
                ly2 = int(face_landmarks[133].y * h)

                # RIGHT EYE CORNERS
                rx1 = int(face_landmarks[362].x * w)
                ry1 = int(face_landmarks[362].y * h)

                rx2 = int(face_landmarks[263].x * w)
                ry2 = int(face_landmarks[263].y * h)

                # Draw eye points
                cv2.circle(frame, (lx1, ly1), 4, (0,0,255), -1)
                cv2.circle(frame, (lx2, ly2), 4, (0,0,255), -1)
                cv2.circle(frame, (rx1, ry1), 4, (0,0,255), -1)
                cv2.circle(frame, (rx2, ry2), 4, (0,0,255), -1)

                # 🔥 IRIS CENTER (LEFT EYE)
                iris_x = int((
                    face_landmarks[468].x +
                    face_landmarks[469].x +
                    face_landmarks[470].x +
                    face_landmarks[471].x
                ) / 4 * w)

                # 🔥 SAFE DIVISION
                eye_width = (lx2 - lx1)
                if eye_width == 0:
                    continue

                ratio = (iris_x - lx1) / eye_width

                # 🔥 GAZE DETECTION
                if ratio < 0.35:
                    gaze = "LEFT"
                elif ratio > 0.65:
                    gaze = "RIGHT"
                else:
                    gaze = "CENTER"

                # 🔥 ATTENTION SYSTEM

                if gaze == "CENTER":
                    attention_score += 1
                else:
                    attention_score -= 1

                # clamp value
                attention_score = max(0, min(attention_score, 20))

                # 🔥 STATUS
                if attention_score > 12:
                    status = "ATTENTIVE"
                    color = (0,255,0)

                elif attention_score > 5:
                    status = "NORMAL"
                    color = (0,255,255)

                else:
                    status = "DISTRACTED"
                    color = (0,0,255)

                cv2.putText(frame, f"Attention: {status}",
                            (30,130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color, 3)

                # DISPLAY
                cv2.putText(frame, f"Gaze: {gaze}",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,255), 2)

                cv2.putText(frame, status,
                            (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color, 3)

        cv2.imshow("Face Landmarks", frame)

        key = cv2.waitKey(1)
        if key == 27:
            print("ESC pressed")
            break

        time.sleep(0.01)

    except Exception as e:
        print("ERROR:", e)
        continue

cap.release()
cv2.destroyAllWindows()