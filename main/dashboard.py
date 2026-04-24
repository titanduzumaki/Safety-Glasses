from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory
import os
import json

app = Flask(__name__)

# 👉 USE faces folder
IMAGE_DIR = "faces"
STATUS_FILE = "status.json"

# ===== LOAD / SAVE STATUS =====
def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_status(data):
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ===== SERVE IMAGES =====
@app.route('/faces/<filename>')
def images(filename):
    return send_from_directory(IMAGE_DIR, filename)

# ===== DASHBOARD =====
@app.route('/')
def dashboard():
    files = sorted(os.listdir(IMAGE_DIR), reverse=True)
    status = load_status()

    return render_template_string("""
    <html>
    <head>
        <title>Face Dashboard</title>
        <style>
            body { background:#111; color:white; font-family:Arial; }
            .card {
                display:inline-block;
                margin:15px;
                padding:10px;
                background:#222;
                border-radius:10px;
                text-align:center;
            }
            img { width:200px; border-radius:8px; }
            button {
                margin-top:10px;
                padding:8px;
                border:none;
                cursor:pointer;
                border-radius:5px;
            }
            .sus { background:red; color:white; }
            .ok { background:green; color:white; }
        </style>
    </head>
    <body>

    <h2>Face Dashboard</h2>

    {% for file in files %}
        <div class="card">
            <img src="/faces/{{file}}"><br><br>

            <b>ID:</b> {{file}}<br>

            <b>Status:</b>
            {{ "Suspicious" if status.get(file) else "Normal" }}<br><br>

            <form action="/toggle" method="post">
                <input type="hidden" name="file" value="{{file}}">
                {% if status.get(file) %}
                    <button class="ok">Mark Normal</button>
                {% else %}
                    <button class="sus">Mark Suspicious</button>
                {% endif %}
            </form>
        </div>
    {% endfor %}

    </body>
    </html>
    """, files=files, status=status)

# ===== TOGGLE =====
@app.route('/toggle', methods=['POST'])
def toggle():
    file = request.form.get("file")
    status = load_status()

    status[file] = not status.get(file, False)

    save_status(status)

    return redirect(url_for('dashboard'))

# ===== RUN =====
if __name__ == "__main__":
    app.run(port=7000, debug=True)