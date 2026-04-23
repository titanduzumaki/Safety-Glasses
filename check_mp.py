from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

MODEL_PATH = "face_landmarker.task"

# Load model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening")
    exit()

# 🔥 Create stable window
cv2.namedWindow("Face Landmarks", cv2.WINDOW_NORMAL)

print("Starting camera... Press ESC to exit")

while True:
    ret, frame = cap.read()

    print("Frame read:", ret)

    if not ret:
        print("Camera failed")
        continue

    # Flip + convert
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = vision.Image(
        image_format=vision.ImageFormat.SRGB,
        data=rgb
    )

    # Timestamp for VIDEO mode
    timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp)

    # Draw landmarks
    if result.face_landmarks:
        print("Face detected")

        for face_landmarks in result.face_landmarks:
            for lm in face_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])

                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Show frame
    cv2.imshow("Face Landmarks", frame)

    # 🔥 Stable key handling
    key = cv2.waitKey(1)

    if key != -1:
        print("Key pressed:", key)

    if key == 27:  # ESC
        print("ESC pressed, exiting...")
        break

    # Small delay for stability
    time.sleep(0.01)

# Cleanup
cap.release()
cv2.destroyAllWindows()