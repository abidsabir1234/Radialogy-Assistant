import cv2
import numpy as np


class ImageTransformer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.brightness = 0       # -100 … +100
        self.rotation   = 0       # degrees
        self.zoom       = 1.0     # 0.5 … 3.0
        self.grayscale  = False
        self.blur       = False
        self._last_gesture = None

    def apply(self, image, gesture: str):
        if image is None:
            return None

        self._update_state(gesture)

        img = image.copy()
        h, w = img.shape[:2]

        # 1. Grayscale
        if self.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 2. Gaussian Blur
        if self.blur:
            img = cv2.GaussianBlur(img, (15, 15), 0)

        # 3. Brightness (HSV V-channel)
        if self.brightness != 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + self.brightness, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 4. Rotation + Zoom
        if self.rotation != 0 or self.zoom != 1.0:
            center = (w // 2, h // 2)
            M   = cv2.getRotationMatrix2D(center, self.rotation, self.zoom)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        return img

    def get_state(self):
        return {
            'brightness': self.brightness,
            'rotation':   self.rotation,
            'zoom':       round(self.zoom, 2),
            'grayscale':  self.grayscale,
            'blur':       self.blur,
        }

    def _update_state(self, gesture: str):
        # Continuous gestures (apply every frame)
        if gesture == 'BRIGHT_UP':
            self.brightness = min(100, self.brightness + 3)
        elif gesture == 'BRIGHT_DOWN':
            self.brightness = max(-100, self.brightness - 3)
        elif gesture == 'ONE':
            self.zoom = min(3.0, self.zoom + 0.03)
        elif gesture == 'FOUR':
            self.zoom = max(0.3, self.zoom - 0.03)
        elif gesture == 'TWO':
            self.rotation = (self.rotation + 2) % 360
        elif gesture == 'ZERO':
            self.reset()

        # Toggle gestures (fire once per gesture change)
        if gesture != self._last_gesture:
            if gesture == 'FIVE':
                self.grayscale = not self.grayscale
            elif gesture == 'THREE':
                self.blur = not self.blur

        self._last_gesture = gesture
