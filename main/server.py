#!/usr/bin/env python3
from flask import Flask, request, Response
import argparse
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

# Shared state
latest_frame = None
lock = threading.Lock()
last_update_ts = 0.0

def process_frame(frame):
    """
    Put your AI here.
    Example: face detection + box.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

@app.route("/upload", methods=["POST"])
def upload():
    global latest_frame, last_update_ts
    data = request.data
    if not data:
        return "No data", 400

    npimg = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return "Decode failed", 400

    frame = process_frame(frame)

    with lock:
        latest_frame = frame
        last_update_ts = time.time()

    return "OK", 200

def mjpeg_generator(fps):
    """
    Streams the latest frame as MJPEG.
    """
    delay = 1.0 / max(fps, 0.1)
    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            # Send a blank frame or wait
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for frames...", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", blank)
        else:
            ok, buf = cv2.imencode(".jpg", frame)

        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

        time.sleep(delay)

@app.route("/")
def index():
    return """
    <html>
      <head><title>Live Stream</title></head>
      <body style="background:black;color:white;text-align:center;">
        <h2>Live Feed</h2>
        <img src="/video" width="720"/>
      </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(mjpeg_generator(app.config["STREAM_FPS"]),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def main():
    ap = argparse.ArgumentParser(description="Flask video server")
    ap.add_argument("--host", default="0.0.0.0", help="Bind host")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--stream-fps", type=float, default=15.0)
    args = ap.parse_args()

    app.config["STREAM_FPS"] = args.stream_fps
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()