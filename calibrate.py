"""
Gesture Calibration Tool  (improved)
=====================================
Walks through all 8 gestures one by one.
For each gesture:
  1. Read the on-screen description and hold that hand pose
  2. Watch the bar chart on the right — wait until ONE bar dominates clearly
  3. Press SPACE to lock in the result
  4. Press R to retry if you got a bad reading
  5. Press S to skip, Q to quit

Result is saved to calibration_result.json and auto-loaded by the app.
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import json
from collections import Counter
from config import Config

# ── Load model ────────────────────────────────────────────
print(f"Loading model from {Config.MODEL_PATH} …")
model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
print("Model loaded. Input shape:", model.input_shape)

NUM_CLASSES = model.output_shape[-1]
print(f"Model has {NUM_CLASSES} output classes.\n")

# ── Gesture list with clear descriptions ─────────────────
# (name, description shown on screen)
GESTURES = [
    ('ZERO',        'Closed fist — no fingers up'),
    ('ONE',         'One finger up — index only'),
    ('TWO',         'Two fingers up — index + middle'),
    ('THREE',       'Three fingers up — index + middle + ring'),
    ('FOUR',        'Four fingers up — all except thumb'),
    ('FIVE',        'Open hand — all 5 fingers spread'),
    ('BRIGHT_UP',   'Your BRIGHT UP gesture (thumbs up or custom)'),
    ('BRIGHT_DOWN', 'Your BRIGHT DOWN gesture (thumbs down or custom)'),
]

# ── Mediapipe hand detection ──────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

task_path = os.path.join('engine', 'hand_landmarker.task')
base_opts  = mp_python.BaseOptions(model_asset_path=task_path)
options    = mp_vision.HandLandmarkerOptions(
    base_options=base_opts,
    num_hands=1,
    min_hand_detection_confidence=0.45,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40,
)
detector = mp_vision.HandLandmarker.create_from_options(options)


def get_roi(frame):
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res    = detector.detect(mp_img)
    if not res.hand_landmarks:
        return None, None
    lms = res.hand_landmarks[0]
    xs  = [lm.x for lm in lms]; ys = [lm.y for lm in lms]
    x1  = max(0, int(min(xs) * w));  y1 = max(0, int(min(ys) * h))
    x2  = min(w, int(max(xs) * w)); y2 = min(h, int(max(ys) * h))
    bw  = x2 - x1; bh = y2 - y1
    side = int(max(bw, bh) * 1.5)
    cx  = (x1 + x2) // 2; cy = (y1 + y2) // 2
    x1  = max(0, cx - side // 2); y1 = max(0, cy - side // 2)
    x2  = min(w, x1 + side);      y2 = min(h, y1 + side)
    roi = frame[y1:y2, x1:x2]
    return (roi if roi.size > 0 else None), (x1, y1, x2, y2)


def predict(roi):
    resized = cv2.resize(roi, Config.MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb_roi = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor  = rgb_roi.astype('float32') / 255.0
    tensor  = np.expand_dims(tensor, 0)
    scores  = model(tensor, training=False).numpy()[0]
    return int(np.argmax(scores)), scores


def draw_bars(frame, scores, mapping):
    """Right-side bar chart showing all class probabilities."""
    h, w   = frame.shape[:2]
    bar_w  = 160
    bar_h  = 18
    x_off  = w - bar_w - 8
    y_off  = 8
    top_i  = int(np.argmax(scores))

    for i, s in enumerate(scores):
        lbl    = mapping.get(i, f'cls_{i}')
        y      = y_off + i * (bar_h + 3)
        filled = int(float(s) * bar_w)
        cv2.rectangle(frame, (x_off, y), (x_off + bar_w, y + bar_h), (30, 30, 30), -1)
        color = (0, 220, 80) if i == top_i else (0, 100, 190)
        cv2.rectangle(frame, (x_off, y), (x_off + filled, y + bar_h), color, -1)
        cv2.putText(frame, f'{i}:{lbl[:9]} {s*100:.0f}%',
                    (x_off - 150, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)


# ── Open camera ───────────────────────────────────────────
cam_idx = Config.CAMERA_INDEX
print(f"Opening camera index {cam_idx}  (change CAMERA_INDEX in .env if wrong)")

cap = None
for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    _c = cv2.VideoCapture(cam_idx, backend)
    if _c.isOpened():
        ret, _ = _c.read()
        if ret:
            cap = _c
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Camera {cam_idx} opened with backend {backend}")
            break
        _c.release()

if cap is None:
    print(f"ERROR: Could not open camera index {cam_idx} with any backend.")
    print("Available indices detected: try CAMERA_INDEX=0 or CAMERA_INDEX=1 in .env")
    exit(1)

# ── State ─────────────────────────────────────────────────
calibrated   = {}   # gesture_name → model_index
label_map    = {i: f'cls_{i}' for i in range(NUM_CLASSES)}
gesture_idx  = 0
vote_buffer  = []
COLLECT_N    = 30   # frames to collect before showing result

print("\n=== CALIBRATION STARTED ===")
print("Controls:  SPACE = confirm   R = retry   S = skip   Q = quit\n")

while gesture_idx < len(GESTURES):
    name, desc = GESTURES[gesture_idx]
    ret, frame = cap.read()
    if not ret:
        continue

    frame     = cv2.flip(frame, 1)
    h, w      = frame.shape[:2]
    roi, box  = get_roi(frame)
    pred_idx  = None
    scores    = np.zeros(NUM_CLASSES)
    collected = len(vote_buffer)

    if roi is not None:
        pred_idx, scores = predict(roi)
        vote_buffer.append(pred_idx)
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

    # ── Header panel ─────────────────────────────────────
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)

    step_txt = f"Step {gesture_idx+1}/{len(GESTURES)}"
    cv2.putText(frame, step_txt, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Gesture: {name}", (10, 52),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, desc, (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 80), 1, cv2.LINE_AA)

    # Progress bar
    prog = min(collected, COLLECT_N)
    cv2.rectangle(frame, (10, 90), (10 + int((w - 20) * prog / COLLECT_N), 108),
                  (0, 180, 255), -1)
    cv2.rectangle(frame, (10, 90), (w - 10, 108), (80, 80, 80), 1)

    if collected < COLLECT_N:
        msg = f"Hold gesture clearly… ({collected}/{COLLECT_N} frames)"
        cv2.putText(frame, msg, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    else:
        best_idx  = Counter(vote_buffer).most_common(1)[0][0]
        best_conf = float(scores[best_idx]) * 100
        cv2.putText(frame,
                    f"Ready!  Dominant index = {best_idx} ({best_conf:.0f}%)   "
                    f"SPACE=confirm  R=retry",
                    (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1, cv2.LINE_AA)

    cv2.putText(frame, "SPACE=confirm  R=retry  S=skip  Q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (120, 120, 120), 1, cv2.LINE_AA)

    draw_bars(frame, scores, label_map)
    cv2.imshow('Gesture Calibration', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit early.")
        break

    elif key == ord('s'):
        print(f"  –  {name} skipped")
        vote_buffer.clear()
        gesture_idx += 1

    elif key == ord('r'):
        vote_buffer.clear()
        print(f"  ↺  Retrying {name}…")

    elif key == ord(' ') and collected >= COLLECT_N:
        best_idx = Counter(vote_buffer).most_common(1)[0][0]
        calibrated[name] = best_idx
        label_map[best_idx] = name
        print(f"  ✓  {name:12s} → model index {best_idx}   "
              f"(distribution: {dict(Counter(vote_buffer))})")
        vote_buffer.clear()
        gesture_idx += 1

    elif key == ord(' ') and collected < COLLECT_N:
        print(f"  Wait — only {collected}/{COLLECT_N} frames collected. Keep holding the gesture.")

cap.release()
cv2.destroyAllWindows()

# ── Save result ───────────────────────────────────────────
if not calibrated:
    print("\nNo gestures calibrated. Nothing saved.")
else:
    corrected = {idx: gname for gname, idx in calibrated.items()}

    print("\n=== CALIBRATION RESULT ===")
    print("Corrected GESTURE_LABELS:")
    for i in sorted(corrected):
        print(f"    {i}: '{corrected[i]}'")

    out = {
        'GESTURE_LABELS': {str(k): v for k, v in corrected.items()},
        'note': 'Auto-generated by calibrate.py — loaded automatically by config.py'
    }
    with open('calibration_result.json', 'w') as f:
        json.dump(out, f, indent=2)

    missing = [g for g, _ in GESTURES if g not in calibrated]
    if missing:
        print(f"\nWARNING: These gestures were skipped: {missing}")
        print("Re-run calibrate.py to fill them in.")
    else:
        print("\nAll 8 gestures calibrated successfully!")

    print("\ncalibration_result.json saved.")
    print("Restart app.py — config.py will load the new mapping automatically.")
