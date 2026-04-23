from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    img = request.data

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as f:
        f.write(img)

    print("Saved:", filename)

    return jsonify({"status": "saved", "file": filename})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)