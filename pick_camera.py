"""
Camera Picker
=============
Shows a live preview of every available camera side by side.
Press the number key matching the camera you want to use.
The choice is saved to .env automatically.

IMPORTANT — before running, make sure:
  - Iriun app is OPEN on your phone
  - Phone and PC are on the SAME WiFi network (or USB connected)
  - Iriun PC driver is installed
Only then will Iriun appear as a live camera feed.
"""

import cv2
import numpy as np
import re

ENV_FILE = '.env'

def update_env(index: int):
    with open(ENV_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'CAMERA_INDEX=\d+', f'CAMERA_INDEX={index}', content)
    with open(ENV_FILE, 'w') as f:
        f.write(content)
    print(f"\n.env updated: CAMERA_INDEX={index}")


# ── Scan all cameras 0-5 ──────────────────────────────────
print("Scanning cameras... (make sure Iriun is running on phone!)\n")
caps = {}
for i in range(6):
    c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if c.isOpened():
        ret, f = c.read()
        if ret:
            caps[i] = c
            print(f"  Camera [{i}] - FOUND and streaming")
        else:
            c.release()
            print(f"  Camera [{i}] - driver found but NO FEED (Iriun not connected?)")
    else:
        print(f"  Camera [{i}] - not available")

if not caps:
    print("\nERROR: No cameras found at all.")
    exit(1)

print(f"\nFound {len(caps)} camera(s). Showing preview...")
print("Press the NUMBER KEY of the camera you want to use, then press Q to save & quit.")
print("Press Q without a number to quit without saving.\n")

selected = None
THUMB_W, THUMB_H = 320, 240

while True:
    thumbs = []
    labels = []

    for idx, cap in caps.items():
        ret, frame = cap.read()
        if ret and frame is not None:
            thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
        else:
            thumb = np.zeros((THUMB_H, THUMB_W, 3), dtype='uint8')
            cv2.putText(thumb, 'NO FEED', (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)

        label = f"[{idx}] Press {idx} to select"
        color = (0, 255, 80) if idx == selected else (200, 200, 200)
        cv2.putText(thumb, label, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(thumb, label, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if idx == selected:
            cv2.rectangle(thumb, (0, 0), (THUMB_W-1, THUMB_H-1), (0, 255, 80), 4)

        thumbs.append(thumb)

    # Pad to even number for display
    while len(thumbs) % 2 != 0 and len(thumbs) > 1:
        thumbs.append(np.zeros((THUMB_H, THUMB_W, 3), dtype='uint8'))

    if len(thumbs) == 1:
        combined = thumbs[0]
    else:
        rows = []
        for i in range(0, len(thumbs), 2):
            rows.append(np.hstack(thumbs[i:i+2]))
        combined = np.vstack(rows)

    # Footer
    footer_h = 36
    footer   = np.zeros((footer_h, combined.shape[1], 3), dtype='uint8')
    msg = (f"Selected: [{selected}]  Press Q to save & quit"
           if selected is not None else "Press a number key to select a camera, then Q to save")
    cv2.putText(footer, msg, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    display = np.vstack([combined, footer])

    cv2.imshow('Camera Picker — pick your Iriun cam', display)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q') or key == 27:
        break
    elif chr(key).isdigit():
        num = int(chr(key))
        if num in caps:
            selected = num
            print(f"Selected camera index {num}")
        else:
            print(f"Index {num} is not available")

for c in caps.values():
    c.release()
cv2.destroyAllWindows()

if selected is not None:
    update_env(selected)
    print(f"Done! CAMERA_INDEX={selected} saved to .env")
    print("Now run:  python app.py")
else:
    print("No camera selected. .env not changed.")
