import cv2
import threading
import time


def _open_camera(source):
    # DSHOW must be first — Iriun virtual webcam fails with MSMF
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"[Camera] Opened index {source} via backend {backend}")
                return cap
            cap.release()
    return None


def _scan_all_cameras():
    print("[Camera] Scanning indices 0-4 for any active camera (DSHOW)...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"[Camera] Found active camera at index {i} — "
                      f"set CAMERA_INDEX={i} in .env to persist this choice")
                return cap, i
            cap.release()
        else:
            cap.release()
    print("[Camera] No active cameras found during scan (is Iriun running on your phone?)")
    return None, None


class Camera:
    def __init__(self, source=0):
        self.source   = source
        self._lock    = threading.Lock()
        self._frame   = None
        self._running = False
        self._cap     = None
        self._thread  = None

    def start(self):
        if self._running:
            return
        self._cap = _open_camera(self.source)
        if self._cap is None:
            print(f"[Camera] Index {self.source} not available — trying fallback scan...")
            cap, found_idx = _scan_all_cameras()
            if cap is not None:
                self._cap   = cap
                self.source = found_idx   # use the found index for recovery retries
            else:
                print(f"[Camera] WARNING: no camera found — will keep retrying every 2 s")
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()

    def is_running(self):
        return self._running

    def _capture_loop(self):
        _fail_count = 0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                _fail_count += 1
                time.sleep(2.0)
                self._cap = _open_camera(self.source)
                if self._cap is None and _fail_count % 5 == 0:
                    cap, found_idx = _scan_all_cameras()
                    if cap is not None:
                        self._cap   = cap
                        self.source = found_idx
                        _fail_count = 0
                continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                _fail_count = 0
                with self._lock:
                    self._frame = frame
            else:
                self._cap.release()
                self._cap = None
                time.sleep(1.0)

    def __del__(self):
        self.stop()
