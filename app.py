"""
Violence Detection - Backend (Flask)

Flow:
  Dashboard  --(video upload)-->  /api/violence/upload
      -> video temporarily save
      -> detect_violence(...)  (aapka model)
      -> result events.json me save
      -> result JSON dashboard ko wapas
"""
import os
import json
import uuid
import threading

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Aapka model wala file (same folder me hona chahiye)
from violence_detector import detect_violence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
EVENTS_FILE = os.path.join(BASE_DIR, "events.json")
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # taaki Person 5 ka dashboard (alag port/laptop se) is API ko call kar sake
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # max 100 MB video

file_lock = threading.Lock()  # do requests ek saath file na bigaad dein


# ---------- Events save / load (abhi JSON file, baad me database) ----------
def load_events():
    if not os.path.exists(EVENTS_FILE):
        return []
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_event(event):
    """
    Abhi: events.json me save.
    Week 2 me: yahi function badal kar MongoDB me save karwana
    (db.events.insert_one(event)). Baaki code same rahega.
    """
    with file_lock:
        events = load_events()
        events.append(event)
        with open(EVENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)


# ---------- Routes ----------
@app.route("/")
def home():
    # test page
    return send_file(os.path.join(BASE_DIR, "index.html"))


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/violence/upload", methods=["POST"])
def upload_video():
    # 1. Video aayi ya nahi?
    if "video" not in request.files:
        return jsonify({"error": "'video' field missing"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Only {sorted(ALLOWED_EXT)} allowed"}), 400

    camera_id = request.form.get("camera_id", default=1, type=int)

    # 2. Unique naam se temporarily save
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file.save(path)

    # 3. Model chalao
    try:
        result = detect_violence(path, camera_id=camera_id)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)  # kaam ho gaya, video delete

    if result is None:
        return jsonify({"error": "Video could not be read (corrupted?)"}), 422

    # 4. Result me id + original naam add karke save
    result["id"] = uuid.uuid4().hex
    result["video"] = file.filename
    save_event(result)

    return jsonify(result), 200


@app.route("/api/events", methods=["GET"])
def get_events():
    events = load_events()
    events.reverse()  # latest pehle
    return jsonify(events)


if __name__ == "__main__":
    # debug=False: debug mode me TensorFlow model 2 baar load hota hai
    app.run(host="0.0.0.0", port=5000, debug=False)
