## Attention Tracking (YOLO + OpenCV)

- **Install** (recommended virtualenv):
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **Model**: Provide a face-capable YOLO checkpoint (e.g., `yolov8n-face.pt` or your custom face model). Place it in the project root or pass `--model` with its path.

- **Run**:
  ```bash
  python attn_tracker.py --model yolov8n-face.pt --overlay --fps-target 30 --start 0 --end 120
  ```

- **Output**: The script prints per-frame JSON lines and popup events to stdout, and a final summary JSON after `END_S`.

- **Key flags**:
  - `--src 0` camera index (use integer for webcam)
  - `--overlay` show on-screen visualization
  - `--start / --end` seconds window
  - Thresholds/tuning: `--px-left`, `--px-right`, `--center-attentive`, `--center-distract`, `--face-miss-ms`, `--popup-after-s`, `--popup-cooldown-s`, `--smoothing`, `--min-face-conf`, `--min-eye-area`

Notes:
- Uses YOLO only for face bbox. Eye/pupil is a lightweight heuristic relying on OpenCV.
- For robust head pose and gaze, you can later swap in landmarks + solvePnP. 