#!/usr/bin/env python3
from flask import Flask, request, Response
import argparse
import cv2
import numpy as np
import threading
import time
import os
import uuid
from deepface import DeepFace

app = Flask(__name__)

# ===== GLOBAL STATE =====
processing_frame = None
processed_frame = None
lock = threading.Lock()

# ===== CONFIG =====
FACE_DIR = "faces"
os.makedirs(FACE_DIR, exist_ok=True)

MODEL_NAME = "Facenet512"
THRESHOLD = 7.5
MAX_PEOPLE = 10
COOLDOWN = 2
FRAME_SKIP = 5

# ===== DATABASE =====
face_db = {}   # {id: [embeddings]}
last_seen = {}
frame_counter = 0

# ===== FACE DETECTOR =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===== GET EMBEDDING =====
def get_embedding(face_img):
    try:
        result = DeepFace.represent(
            face_img,
            model_name=MODEL_NAME,
            enforce_detection=False
        )
        return np.array(result[0]["embedding"])
    except:
        return None

# ===== MATCH =====
def match_face(embedding):
    best_id = None
    best_dist = 999

    for fid, embeddings in face_db.items():
        avg = np.mean(embeddings, axis=0)
        dist = np.linalg.norm(embedding - avg)

        if dist < best_dist:
            best_dist = dist
            best_id = fid

    if best_dist < THRESHOLD:
        return best_id
    return None

# ===== REGISTER =====
def register_face(face_img):
    global face_db, last_seen

    if face_img.shape[0] < 80 or face_img.shape[1] < 80:
        return None

    embedding = get_embedding(face_img)
    if embedding is None:
        return None

    face_id = match_face(embedding)

    if face_id:
        if face_id in last_seen and time.time() - last_seen[face_id] < COOLDOWN:
            return face_id

        face_db[face_id].append(embedding)
        last_seen[face_id] = time.time()
        return face_id

    # NEW PERSON
    if len(face_db) >= MAX_PEOPLE:
        return None

    face_id = str(uuid.uuid4())[:8]
    face_db[face_id] = [embedding]
    last_seen[face_id] = time.time()

    cv2.imwrite(os.path.join(FACE_DIR, f"{face_id}.jpg"), face_img)

    print(f"[NEW FACE] {face_id}")

    return face_id

# ===== PROCESS FRAME =====
def process_frame(frame):
    global frame_counter
    frame_counter += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        4,
        minSize=(30,30)
    )

    for (x, y, w, h) in faces:

        if w < 80 or h < 80:
            continue

        face_crop = frame[y:y+h, x:x+w]

        if frame_counter % FRAME_SKIP == 0:
            face_id = register_face(face_crop)
        else:
            face_id = None

        label = f"ID:{face_id}" if face_id else "..."

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.putText(frame, f"People: {len(face_db)}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return frame

# ===== BACKGROUND THREAD =====
def processing_worker():
    global processing_frame, processed_frame

    while True:
        if processing_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = processing_frame.copy()

        frame = process_frame(frame)

        with lock:
            processed_frame = frame

# ===== RECEIVE =====
@app.route("/upload", methods=["POST"])
def upload():
    global processing_frame

    data = request.data
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return "Error", 400

    with lock:
        processing_frame = frame.copy()

    return "OK"

# ===== STREAM =====
def generate():
    while True:
        with lock:
            frame = processed_frame

        if frame is None:
            blank = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(blank, "Waiting...",
                        (30,240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,255), 2)
            ret, buffer = cv2.imencode(".jpg", blank)
        else:
            ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

# ===== ROUTES =====
@app.route("/")
def index():
    return """
    <h2>Threaded Face System</h2>
    <img src="/video" width="720">
    """

@app.route("/video")
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # START BACKGROUND THREAD
    threading.Thread(target=processing_worker, daemon=True).start()

    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()