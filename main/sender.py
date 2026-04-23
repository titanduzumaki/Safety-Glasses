#!/usr/bin/env python3
import cv2
import requests
import argparse
import time

def main():
    ap = argparse.ArgumentParser(description="Camera sender")
    ap.add_argument("--server", required=True, help="Server base URL, e.g. http://127.0.0.1:5000")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=10.0, help="Send FPS")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality (0-100)")
    ap.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout seconds")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    url = args.server.rstrip("/") + "/upload"
    delay = 1.0 / max(args.fps, 0.1)

    print(f"[INFO] Sending to {url}")
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed")
                continue

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if not ok:
                print("[WARN] Encode failed")
                continue

            try:
                r = requests.post(url, data=buf.tobytes(),
                                  headers={"Content-Type": "image/jpeg"},
                                  timeout=args.timeout)
                # Optional: print(r.status_code)
            except requests.exceptions.RequestException as e:
                print(f"[WARN] POST failed: {e}")

            # FPS control
            elapsed = time.time() - t0
            if elapsed < delay:
                time.sleep(delay - elapsed)
    finally:
        cap.release()

if __name__ == "__main__":
    main()