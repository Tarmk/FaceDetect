# Attention Tracking (YOLO + OpenCV)

Watches a webcam (or a video file), finds the face with a YOLOv8 face model, estimates where the eyes/pupils are pointing with plain OpenCV heuristics, and emits per-frame attention JSON plus "popup" events when the viewer has looked away for too long.

- **Install** (recommended virtualenv):
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **Model**: On first run the script downloads `yolov8n-face-lindevs.pt` into `weights/` automatically. To use your own face-capable YOLO checkpoint, pass `--model path/to/model.pt`.

- **Run**:
  ```bash
  python attn_tracker.py --overlay --fps-target 30 --start 0 --end 120
  ```

- **Output**: The script prints per-frame JSON lines and popup events to stdout, and a final summary JSON after `END_S`.

- **Key flags**:
  - `--src 0` camera index, or `--src path/to/video.mp4` for a file
  - `--overlay` show on-screen visualization
  - `--start / --end` seconds window
  - Thresholds/tuning: `--px-left`, `--px-right`, `--center-attentive`, `--center-distract`, `--face-miss-ms`, `--popup-after-s`, `--popup-cooldown-s`, `--smoothing`, `--min-face-conf`, `--min-eye-area`

Notes:
- Uses YOLO only for face bbox. Eye/pupil is a lightweight heuristic relying on OpenCV.
- For robust head pose and gaze, you can later swap in landmarks + solvePnP. 