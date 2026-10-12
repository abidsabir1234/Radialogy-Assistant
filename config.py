import os
import json
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'change_me_in_production')
    DEBUG = True

    # Show per-class CNN score bars on the live feed — set False once labels are confirmed correct
    DEBUG_SCORES = os.environ.get('DEBUG_SCORES', 'true').lower() == 'true'

    CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', 0))

    MONGODB_URL = os.environ.get('MONGODB_URL')
    DB_NAME = 'Radiology_Assistant'

    MODEL_PATH = 'gesture_v2.h5'
    MODEL_INPUT_SIZE = (96, 96)

    CONFIDENCE_THRESHOLD = 0.80

    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'gif', 'jfif', 'jpe'}

    # CRITICAL: Keras flow_from_directory sorts class folders ALPHABETICALLY.
    # Classes in order: BRIGHT_DOWN, BRIGHT_UP, FIVE, FOUR, ONE, THREE, TWO, ZERO
    # Run calibrate.py if gestures are mis-labelled — it overwrites this via calibration_result.json.
    _DEFAULT_GESTURE_LABELS = {
        0: 'BRIGHT_DOWN',
        1: 'BRIGHT_UP',
        2: 'FIVE',
        3: 'FOUR',
        4: 'ONE',
        5: 'THREE',
        6: 'TWO',
        7: 'ZERO'
    }

    _CAL_FILE = 'calibration_result.json'
    if os.path.exists(_CAL_FILE):
        with open(_CAL_FILE) as _f:
            _cal = json.load(_f)
        _cal_labels = {int(k): v for k, v in _cal['GESTURE_LABELS'].items()}
        GESTURE_LABELS = {**_DEFAULT_GESTURE_LABELS, **_cal_labels}
        print(f"[Config] Loaded {len(_cal_labels)} calibrated labels from {_CAL_FILE}")
    else:
        GESTURE_LABELS = _DEFAULT_GESTURE_LABELS

    GESTURE_ACTIONS = {
        'BRIGHT_UP':   'Brightness ▲  (Thumb Up)',
        'BRIGHT_DOWN': 'Brightness ▼  (Pinky Up)',
        'ZERO':        'Reset All  (Fist)',
        'ONE':         'Zoom In  (Index Finger)',
        'TWO':         'Rotate 2°  (V-Sign)',
        'THREE':       'Gaussian Blur  (3 Fingers)',
        'FOUR':        'Zoom Out  (4 Fingers)',
        'FIVE':        'Grayscale Toggle  (Open Palm)',
        'NONE':        'No Hand Detected',
        'UNCERTAIN':   'Hold gesture steadier…',
    }
