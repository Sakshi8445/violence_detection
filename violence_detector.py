import os
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
SEQ_LEN = 16
WINDOW_SEC = 1000        # whole video = one clip (same as training)
MEDIUM_THR = 0.5
HIGH_THR = 0.75
FIGHT_IS_CLASS_1 = True

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "violence_cnn_lstm.keras")

_model = None


def _build_model():
    base_cnn = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    return Sequential([
        TimeDistributed(base_cnn, input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(128),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])


def get_model():
    """Model sirf pehli baar load hota hai, phir reuse hota hai."""
    global _model
    if _model is None:
        _model = _build_model()
        _model.load_weights(WEIGHTS_PATH)
    return _model


def _prep(frame_bgr):
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return preprocess_input(frame.astype("float32"))


def _load_clips(video_path, seq_len=SEQ_LEN, window_sec=WINDOW_SEC):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if total <= 0:
        cap.release()
        return None

    win = max(int(fps * window_sec), seq_len)
    starts = list(range(0, max(total - win, 0) + 1, win)) or [0]

    clips, times = [], []
    for s in starts:
        end = min(s + win, total) - 1
        idxs = np.linspace(s, end, seq_len).astype(int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1])
                    continue
                break
            frames.append(_prep(frame))
        if len(frames) == seq_len:
            clips.append(np.array(frames))
            times.append(round(s / fps, 1))
    cap.release()

    if not clips:
        return None
    return np.array(clips), times


def _risk(p):
    if p >= HIGH_THR:
        return "high"
    if p >= MEDIUM_THR:
        return "medium"
    return "low"


def detect_violence(video_path, camera_id=1):
    loaded = _load_clips(video_path)
    if loaded is None:
        return None
    clips, times = loaded

    raw = get_model().predict(clips, verbose=0).reshape(-1)
    fight_probs = raw if FIGHT_IS_CLASS_1 else 1 - raw

    worst = int(np.argmax(fight_probs))
    fight_p = float(fight_probs[worst])
    label = "Fight" if fight_p >= MEDIUM_THR else "NonFight"
    conf = fight_p if label == "Fight" else 1 - fight_p

    return {
        "event_type": "violence",
        "prediction": label,
        "confidence": round(conf, 3),
        "risk_level": _risk(fight_p),
        "peak_clip_start_sec": times[worst],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": camera_id,
        "video": video_path,
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python violence_detector.py <video_path> [camera_id]")
        sys.exit(1)
    cam = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    result = detect_violence(sys.argv[1], camera_id=cam)
    print(json.dumps(result) if result else "Could not process video.")