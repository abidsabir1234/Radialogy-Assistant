# CNN (gesture_v2.h5) was trained on Leap Motion infrared data — domain gap causes
# misclassification on a real webcam. MediaPipe landmarks are used for classification
# instead; CNN scores are computed only to power the debug HUD on the dashboard.
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf
import os
from config import Config


class GestureRecognizer:
    def __init__(self):
        self.model      = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
        self.labels     = Config.GESTURE_LABELS   # {0:'BRIGHT_DOWN', ...}
        self.threshold  = Config.CONFIDENCE_THRESHOLD
        self.input_size = Config.MODEL_INPUT_SIZE  # (96, 96)

        task_path = os.path.join('engine', 'hand_landmarker.task')
        base_opts = mp_python.BaseOptions(model_asset_path=task_path)
        options   = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        self._vote_buf  = []
        self._vote_size = 5

    def process_frame(self, frame):
        """Returns: label, confidence, landmarks, roi_coords, scores(CNN debug only)."""
        h, w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            self._vote_buf.clear()
            return 'NONE', 0.0, None, None, np.zeros(8)

        lm_list = results.hand_landmarks[0]

        xs = [lm.x for lm in lm_list]
        ys = [lm.y for lm in lm_list]
        x1 = max(0, int(min(xs) * w))
        y1 = max(0, int(min(ys) * h))
        x2 = min(w, int(max(xs) * w))
        y2 = min(h, int(max(ys) * h))

        bw   = x2 - x1
        bh   = y2 - y1
        side = int(max(bw, bh) * 1.5)
        cx   = (x1 + x2) // 2
        cy   = (y1 + y2) // 2
        x1   = max(0, cx - side // 2)
        y1   = max(0, cy - side // 2)
        x2   = min(w, x1 + side)
        y2   = min(h, y1 + side)

        roi_coords = (x1, y1, x2, y2)
        roi        = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return 'UNCERTAIN', 0.0, lm_list, roi_coords, np.zeros(8)

        roi_96  = cv2.resize(roi, self.input_size, interpolation=cv2.INTER_LINEAR)
        roi_rgb = cv2.cvtColor(roi_96, cv2.COLOR_BGR2RGB)
        tensor  = roi_rgb.astype('float32') / 255.0
        tensor  = np.expand_dims(tensor, axis=0)
        scores  = self.model(tensor, training=False).numpy()[0]

        lm_label, lm_conf = self._classify_from_landmarks(lm_list, h, w)

        vote_entry = lm_label if lm_label is not None else 'UNCERTAIN'
        self._vote_buf.append(vote_entry)
        if len(self._vote_buf) > self._vote_size:
            self._vote_buf.pop(0)

        voted_label = max(set(self._vote_buf), key=self._vote_buf.count)

        if voted_label == 'UNCERTAIN':
            return 'UNCERTAIN', lm_conf, lm_list, roi_coords, scores

        return voted_label, lm_conf, lm_list, roi_coords, scores

    def draw_overlay(self, frame, lm_list, roi_coords, label, confidence,
                     scores=None, debug=False):
        h, w = frame.shape[:2]

        if lm_list:
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17),
            ]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (0, 180, 80), 1, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 255, 120), -1, cv2.LINE_AA)

        if roi_coords:
            x1, y1, x2, y2 = roi_coords
            col = (0, 255, 100) if label not in ('NONE', 'UNCERTAIN') else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

        text  = f"{label}  {confidence*100:.0f}%"
        color = (0, 255, 100) if label not in ('NONE', 'UNCERTAIN') else (80, 80, 80)
        cv2.putText(frame, text, (16, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (16, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)

        return frame

    def _classify_from_landmarks(self, lm_list, h, w):
        # Distance from wrist rather than y-coord: rotation-invariant, works at any hand angle
        pts = np.array([(lm.x * w, lm.y * h) for lm in lm_list])

        def d(a, b):
            return float(np.linalg.norm(pts[a] - pts[b]))

        RATIO = 1.3

        # Thumb uses 1.4 — its range of motion is shorter than other fingers
        thumb_ext  = d(4, 0) > d(2, 0) * 1.4

        index_up   = d(8,  0) > d(6,  0) * RATIO
        middle_up  = d(12, 0) > d(10, 0) * RATIO
        ring_up    = d(16, 0) > d(14, 0) * RATIO
        pinky_up   = d(20, 0) > d(18, 0) * RATIO

        non_thumb = [index_up, middle_up, ring_up, pinky_up]
        cnt       = sum(non_thumb)

        if not thumb_ext and cnt == 0:
            return 'ZERO', 0.95

        if thumb_ext and cnt == 4:
            return 'FIVE', 0.95

        if not thumb_ext and cnt == 4:
            return 'FOUR', 0.95

        # pts[4]=thumb tip, pts[5]=index MCP; tip above knuckle = thumb pointing up
        if thumb_ext and cnt == 0 and pts[4][1] < pts[5][1]:
            return 'BRIGHT_UP', 0.95

        if not thumb_ext and not index_up and not middle_up and not ring_up and pinky_up:
            return 'BRIGHT_DOWN', 0.95

        if not thumb_ext and index_up and cnt == 1:
            return 'ONE', 0.95

        if not thumb_ext and index_up and middle_up and cnt == 2:
            return 'TWO', 0.95

        if not thumb_ext and index_up and middle_up and ring_up and cnt == 3:
            return 'THREE', 0.95

        return None, 0.0
