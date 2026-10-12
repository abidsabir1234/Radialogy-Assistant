import os
import time
import threading
from functools import wraps

import cv2
import numpy as np
from flask import (Blueprint, render_template, Response, request,
                   session, redirect, url_for, jsonify)

from config import Config
from engine.camera      import Camera
from engine.recognizer  import GestureRecognizer
from engine.transformer import ImageTransformer

main_bp = Blueprint('main', __name__)

_camera      = Camera(source=Config.CAMERA_INDEX)
_recognizer  = GestureRecognizer()
_transformer = ImageTransformer()

_state_lock = threading.Lock()
_state = {
    'gesture':    'NONE',
    'confidence': 0.0,
    'scores':     [0.0] * 8,
}

# Module-level (not session) so generator threads can access it
_scan_lock  = threading.Lock()
_scan_state = {
    'path': None,
    'img':  None,
}

# Transform output cache — avoids recomputing the same frame repeatedly
_transform_cache = {
    'gesture':  None,
    'img_id':   None,   # id() of the current img ndarray
    'buf':      None,   # last encoded JPEG bytes
}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated


@main_bp.route('/')
def home():
    return render_template('home.html')


@main_bp.route('/about')
def about():
    return render_template('about.html')


@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    sent = False
    if request.method == 'POST':
        sent = True
    return render_template('contact.html', sent=sent)


@main_bp.route('/dashboard')
@login_required
def dashboard():
    with _scan_lock:
        has_image = _scan_state['path'] is not None
        
    if has_image:
        _camera.start()   # idempotent
        
    return render_template('dashboard.html',
                           user=session['user'],
                           actions=Config.GESTURE_ACTIONS,
                           has_image=has_image)


@main_bp.route('/gesture_status')
@login_required
def gesture_status():
    with _state_lock:
        s = dict(_state)
    with _scan_lock:
        has_image = _scan_state['path'] is not None
    return jsonify({
        'gesture':    s['gesture'],
        'confidence': round(s['confidence'] * 100, 1),
        'scores':     [round(float(v) * 100, 1) for v in s['scores']],
        'labels':     list(Config.GESTURE_LABELS.values()),
        'action':     Config.GESTURE_ACTIONS.get(s['gesture'], ''),
        'has_image':  has_image,
    })


@main_bp.route('/reset_transform', methods=['POST'])
@login_required
def reset_transform():
    _transformer.reset()
    return jsonify({'ok': True})


@main_bp.route('/video_feed')
@login_required
def video_feed():
    return Response(_gen_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _gen_video():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    _frame_counter = 0          # used for CNN skip — every 3rd frame only
    _last_label  = 'NONE'
    _last_conf   = 0.0
    _last_scores = np.zeros(8)
    _last_lm     = None
    _last_roi    = None

    while True:
        with _scan_lock:
            has_image = _scan_state['path'] is not None
            
        if not has_image:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera off until image upload",
                        (80, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (60, 80, 100), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode('.jpg', placeholder, encode_params)
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
            time.sleep(1.0)
            continue

        frame = _camera.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        _frame_counter += 1

        # Run full recognition every frame (landmarks are fast);
        # reuse previous CNN scores on non-sampled frames to cut CPU usage.
        label, conf, lm, roi, scores = _recognizer.process_frame(frame)

        # Cache results for overlay on skipped frames
        _last_label  = label
        _last_conf   = conf
        _last_lm     = lm
        _last_roi    = roi
        _last_scores = scores

        with _state_lock:
            _state['gesture']    = label
            _state['confidence'] = conf
            _state['scores']     = scores.tolist()

        frame = _recognizer.draw_overlay(frame, lm, roi, label, conf,
                                         scores=scores,
                                         debug=Config.DEBUG_SCORES)

        ok, buf = cv2.imencode('.jpg', frame, encode_params)
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


@main_bp.route('/transform_feed')
@login_required
def transform_feed():
    return Response(_gen_transform(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _gen_transform():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]

    while True:
        with _scan_lock:
            img     = _scan_state['img']
            img_id  = id(img)           # changes only when a new image is uploaded

        if img is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Upload an image to begin",
                        (80, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (60, 80, 100), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode('.jpg', placeholder, encode_params)
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
            # No active image — sleep longer to save CPU
            time.sleep(0.5)
            continue

        with _state_lock:
            gesture = _state['gesture']

        # --- Transform cache: skip recompute if gesture + image are unchanged ---
        # Continuous gestures (BRIGHT_UP/DOWN, ONE, FOUR, TWO) update transformer
        # state on every call, so we always recompute when they're active.
        # For static / no-op gestures we can safely return the cached frame.
        _CONTINUOUS = {'BRIGHT_UP', 'BRIGHT_DOWN', 'ONE', 'FOUR', 'TWO'}
        cache = _transform_cache
        if (
            cache['buf'] is not None
            and gesture not in _CONTINUOUS
            and cache['gesture'] == gesture
            and cache['img_id'] == img_id
        ):
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + cache['buf'] + b'\r\n')
            # Idle — sleep a bit longer so we don't busy-loop
            time.sleep(0.08)
            continue

        transformed = _transformer.apply(img, gesture)

        ok, buf = cv2.imencode('.jpg', transformed, encode_params)
        if not ok:
            continue

        encoded_bytes = buf.tobytes()
        # Update cache
        cache['gesture'] = gesture
        cache['img_id']  = img_id
        cache['buf']     = encoded_bytes

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + encoded_bytes + b'\r\n')

        time.sleep(0.033)


@main_bp.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        return jsonify({'error': 'File type not allowed'}), 400

    upload_dir = os.path.abspath(Config.UPLOAD_FOLDER)
    os.makedirs(upload_dir, exist_ok=True)
    save_path  = os.path.join(upload_dir, 'current_scan.jpg')

    np_arr = np.frombuffer(file.read(), dtype=np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Cannot decode image'}), 400

    cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with _scan_lock:
        _scan_state['path'] = save_path
        _scan_state['img']  = img.copy()

    _transformer.reset()

    return redirect(url_for('main.dashboard'))


@main_bp.route('/download_scan')
@login_required
def download_scan():
    with _scan_lock:
        img = _scan_state['img']
    if img is None:
        return "No image uploaded", 404

    transformed = _transformer.apply(img, 'NONE')
    
    ok, buf = cv2.imencode('.jpg', transformed, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if not ok:
        return "Encoding failed", 500

    return Response(buf.tobytes(), 
                    mimetype='image/jpeg',
                    headers={"Content-Disposition": "attachment;filename=transformed_scan.jpg"})
