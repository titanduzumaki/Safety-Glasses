from flask import Flask, request, jsonify

app = Flask(__name__)

suspicious_db = {}

# receive suspicious person
@app.route("/report", methods=["POST"])
def report():
    data = request.json
    face_id = data["face_id"]
    image = data["image"]  # base64 later

    suspicious_db[face_id] = data

    print("New suspicious person:", face_id)

    return jsonify({"status": "received"})

# devices fetch suspicious list
@app.route("/get_suspicious")
def get_suspicious():
    return jsonify(suspicious_db)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)