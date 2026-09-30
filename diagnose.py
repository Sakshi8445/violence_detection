import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from violence_detector import get_model, IMG_SIZE, SEQ_LEN

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_videos")


def frames_bgr(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, SEQ_LEN).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1])
                continue
            break
        frames.append(cv2.resize(f, (IMG_SIZE, IMG_SIZE)))
    cap.release()
    return np.array(frames) if len(frames) == SEQ_LEN else None


def rgb(x):
    return x[..., ::-1]


variants = {
    "A rgb+mobilenet (current)": lambda x: preprocess_input(rgb(x).astype("float32")),
    "B rgb /255": lambda x: rgb(x).astype("float32") / 255.0,
    "C bgr+mobilenet": lambda x: preprocess_input(x.astype("float32")),
    "D bgr /255": lambda x: x.astype("float32") / 255.0,
}

model = get_model()
print("\nFight score (0 = NonFight, 1 = Fight):\n")
print("%-12s %-9s" % ("Video", "Expected") + "".join("%-28s" % k for k in variants))

for name in sorted(os.listdir(FOLDER)):
    if not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue
    fr = frames_bgr(os.path.join(FOLDER, name))
    if fr is None:
        print(name, "-> could not read")
        continue
    expected = "NonFight" if name.upper().startswith("NV") else "Fight"
    line = "%-12s %-9s" % (name[:11], expected)
    for fn in variants.values():
        p = float(model.predict(fn(fr)[None], verbose=0).reshape(-1)[0])
        line += "%-28.3f" % p
    print(line)