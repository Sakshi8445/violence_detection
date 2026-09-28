# Violence Detector

Checks a video and tells you if it contains a fight.

## Setup

Python 3.13.0

```
pip install -r requirements.txt
```

Keep `violence_cnn_lstm.keras` in the same folder as `violence_detector.py`.

## Use

```python
from violence_detector import detect_violence

result = detect_violence("video.mp4", camera_id=1)
```

## Result

```json
{
  "event_type": "violence",
  "prediction": "Fight",
  "confidence": 0.912,
  "risk_level": "high",
  "peak_clip_start_sec": 0.0,
  "timestamp": "2026-09-28 14:30:05",
  "camera_id": 1,
  "video": "video.mp4"
}
```

- `prediction`: `Fight` or `NonFight`
- `risk_level`: `low`, `medium` or `high`

## Notes

- Use short clips and a file path (no live streams).
- Returns `None` if the video can't be read.
