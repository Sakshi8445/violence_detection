import os
import csv
from violence_detector import detect_violence

BASE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(BASE, "test_videos")
os.makedirs(FOLDER, exist_ok=True)

rows = []

for name in sorted(os.listdir(FOLDER)):
    if not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue
    expected = "NonFight" if name.upper().startswith("NV") else "Fight"
    result = detect_violence(os.path.join(FOLDER, name), camera_id=1)
    if result is None:
        rows.append([name, expected, "ERROR", "-", "No"])
        continue
    pred = result["prediction"]
    rows.append([name, expected, pred, result["confidence"],
                 "Yes" if pred == expected else "No"])

# error handling test: kharab video
fake = os.path.join(FOLDER, "fake_corrupt.mp4")
with open(fake, "w") as f:
    f.write("this is not a video")
err = detect_violence(fake, camera_id=1)
os.remove(fake)

print("\n%-15s %-10s %-10s %-10s %-6s" % ("Video", "Expected", "Predicted", "Conf", "Sahi?"))
for r in rows:
    print("%-15s %-10s %-10s %-10s %-6s" % tuple(r))

correct = sum(1 for r in rows if r[4] == "Yes")
print(f"\nAccuracy: {correct}/{len(rows)}")
print("Error test (corrupt file) ->", "PASS (None returned)" if err is None else f"Got: {err}")

with open("results.csv", "w", newline="") as f:
    csv.writer(f).writerows([["Video", "Expected", "Predicted", "Confidence", "Correct"]] + rows)
print("Saved: results.csv")