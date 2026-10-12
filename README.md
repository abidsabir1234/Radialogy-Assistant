<div align="center">

# 🩻 Radiology Assistant

### Touchless, Hand-Gesture-Controlled Medical Image Manipulation

*Final Year Project — Computer Vision · Deep Learning · Human–Computer Interaction*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-0097A7?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)

</div>

---

## 📌 Overview

In operating theatres and interventional radiology suites, surgeons often need to inspect X-ray, CT, MRI or ultrasound images **during** a procedure. Touching a mouse, keyboard or touchscreen breaks the sterile field, so they have to ask an assistant to manipulate the image for them. That is slow, error-prone and interrupts the workflow.

**Radiology Assistant** removes that bottleneck. A standard webcam watches the clinician's hand. The system recognises **eight static hand gestures in real time** and maps each one to an image operation: zoom, rotate, adjust brightness, blur, convert to grayscale, or reset. The radiology scan updates live in the browser, without anything being touched.

## ✨ Key Features

- **Real-time gesture recognition** from an ordinary webcam (or a phone used as a webcam via Iriun), at about 30 FPS on a CPU.
- **Eight-gesture vocabulary** mapped to clinically useful image operations.
- **Live side-by-side dashboard**: an annotated camera feed with the hand skeleton and bounding box, next to the transformed scan.
- **Temporal smoothing**: a 5-frame majority vote suppresses flicker and accidental triggers.
- **Rotation-invariant classification**: finger states are measured as distances from the wrist, not screen coordinates, so gestures work at any hand angle.
- **Secure user accounts**: sign-up and login backed by MongoDB Atlas, with PBKDF2-SHA256 password hashing and session-protected routes.
- **Upload any scan** (PNG, JPG, BMP, TIFF, WEBP, GIF) and **download the transformed result**.
- **Robust camera handling**: tries several capture backends, scans other camera indices automatically, and reconnects on its own if the feed drops.
- **Calibration and camera-picker utilities** for adapting the system to new hardware.

## ✋ Gesture Vocabulary

| Gesture | Hand Pose | Action | Behaviour |
|:---:|---|---|---|
| ✊ `ZERO` | Closed fist | **Reset** all transformations | Instant |
| ☝️ `ONE` | Index finger up | **Zoom in** (up to 3×) | Continuous while held |
| ✌️ `TWO` | Index + middle (V-sign) | **Rotate** +2° per frame | Continuous while held |
| 🤟 `THREE` | Index + middle + ring | **Gaussian blur** on/off | Toggles once per gesture |
| 🖖 `FOUR` | Four fingers, thumb folded | **Zoom out** (down to 0.3×) | Continuous while held |
| 🖐️ `FIVE` | Open palm | **Grayscale** on/off | Toggles once per gesture |
| 👍 `BRIGHT_UP` | Thumb up | **Brightness +** | Continuous while held |
| 🤙 `BRIGHT_DOWN` | Pinky up | **Brightness −** | Continuous while held |

## 🏗️ System Architecture

```
 ┌──────────────┐   frames   ┌──────────────────────┐  gesture label  ┌───────────────────┐
 │  Webcam /    │──────────▶│  GestureRecognizer   │───────────────▶│  ImageTransformer │
 │  Iriun phone │  (thread)  │  • MediaPipe 21-pt   │                 │  • grayscale      │
 │  camera.py   │            │    hand landmarks    │                 │  • Gaussian blur  │
 └──────────────┘            │  • geometric finger  │                 │  • HSV brightness │
                             │    state classifier  │                 │  • affine rotate  │
                             │  • CNN score head    │                 │    + zoom         │
                             │  • 5-frame vote      │                 └─────────┬─────────┘
                             └──────────┬───────────┘                           │
                                        │ annotated frame                       │ transformed scan
                                        ▼                                       ▼
                             ┌──────────────────────────────────────────────────────────┐
                             │  Flask (Blueprints)  —  MJPEG streams + JSON status API  │
                             │  /video_feed   /transform_feed   /gesture_status          │
                             └──────────────────────────┬───────────────────────────────┘
                                                        ▼
                                      Browser dashboard (HTML / CSS / JS)
                                      Auth ◀──▶ MongoDB Atlas (PBKDF2 hashes)
```

### Recognition Pipeline

1. **Capture**: a background thread grabs 640×480 frames, so the web server is never blocked by camera I/O.
2. **Hand detection**: the MediaPipe Hand Landmarker (`hand_landmarker.task`) finds 21 3-D landmarks on one hand.
3. **Region of interest**: a square box, 1.5× the size of the hand, is cropped and resized to 96×96.
4. **Classification**: each finger is judged *extended* when its tip-to-wrist distance is more than 1.3× its knuckle-to-wrist distance (1.4× for the thumb). The resulting finger pattern is mapped to one of the eight gestures.
5. **CNN head**: a custom Keras CNN (`gesture_v2.h5`, 96×96 RGB input, 8-class softmax) scores the same ROI. Its per-class scores are shown in the dashboard as a diagnostic HUD.
6. **Smoothing**: a sliding majority vote over the last 5 predictions produces the final, stable label.
7. **Transformation**: the label updates the transform state, and OpenCV re-renders the scan. A frame cache skips recomputation when nothing has changed.

## 🔬 Research Insight: The Domain Gap

The CNN was first trained on a **Leap Motion infrared** hand-gesture dataset. On ordinary RGB webcam frames its accuracy dropped sharply. This is a textbook case of **domain shift**: the training data (greyscale IR silhouettes, controlled lighting) differs from the deployment data (colour images, cluttered backgrounds, variable lighting).

Instead of collecting and labelling a new dataset, I redesigned the pipeline as a **hybrid system**. MediaPipe's landmark model, which was trained on large and diverse RGB data, does the hand localisation. A lightweight, interpretable **geometric rule layer** then makes the final decision. The CNN is kept as a diagnostic signal. This made recognition reliable under real-world conditions with no retraining. `calibrate.py` is included to re-map CNN class indices when the model is retrained or the hardware changes.

**Possible extensions:** fine-tuning the CNN on webcam data, or training a small classifier directly on landmark vectors.

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Computer Vision** | OpenCV, MediaPipe Tasks (Hand Landmarker) |
| **Deep Learning** | TensorFlow / Keras (custom CNN) |
| **Backend** | Python, Flask (application factory and Blueprints), multithreading |
| **Database** | MongoDB Atlas (PyMongo), passlib PBKDF2-SHA256 |
| **Frontend** | HTML5, CSS3, JavaScript, Jinja2 templates, MJPEG streaming |

## 📂 Project Structure

```
Radialogy-Assistant/
├── app.py                  # Flask application factory & entry point
├── config.py               # Central config (model, thresholds, gesture map, env vars)
├── calibrate.py            # Interactive tool to re-map CNN class indices → gestures
├── pick_camera.py          # Live preview of all cameras; saves choice to .env
├── gesture_v2.h5           # Trained Keras CNN (8 classes, 96×96 input)
├── engine/
│   ├── camera.py           # Threaded capture, multi-backend fallback, auto-reconnect
│   ├── recognizer.py       # MediaPipe landmarks + geometric classifier + CNN + voting
│   ├── transformer.py      # Stateful OpenCV image transformation engine
│   └── hand_landmarker.task# MediaPipe hand landmark model
├── routes/
│   ├── main.py             # Dashboard, video/transform streams, upload/download, status API
│   └── auth.py             # Signup / login / logout
├── database/
│   └── auth.py             # MongoDB user store with hashed passwords
├── templates/              # Jinja2 pages (home, about, contact, dashboard, auth/)
└── static/                 # CSS, gesture illustrations, sample scans (X-ray, CT, MRI, US)
```

## 🚀 Getting Started

### Prerequisites

- Python **3.12** (recommended)
- A webcam, or a phone running [Iriun Webcam](https://iriun.com/)
- A free [MongoDB Atlas](https://www.mongodb.com/atlas) cluster

### 1. Clone and install

```bash
git clone https://github.com/abidsabir1234/Radialogy-Assistant.git
cd Radialogy-Assistant

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
MONGODB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
SECRET_KEY=<random-string>
CAMERA_INDEX=0
DEBUG_SCORES=true
```

Generate a secure secret key with:

```bash
python -c "import secrets; print(secrets.token_hex(24))"
```

> The app creates the database `Radiology_Assistant` and the collection `Account` automatically on first sign-up.

### 3. (Optional) Choose your camera

```bash
python pick_camera.py
```

### 4. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**, create an account, upload a scan from the dashboard, and start gesturing.

## 🎯 Usage Tips

- Keep your hand **30–60 cm** from the camera, with your palm facing it.
- Use even, front-facing lighting. Strong backlight reduces landmark accuracy.
- Hold each gesture steadily for about 0.2 s. The 5-frame vote needs consistent frames.
- Make a **fist** at any time to reset the image.

## 🔭 Future Work

- DICOM support and windowing / level adjustment for CT.
- Dynamic gestures such as swipe-to-scroll through image series.
- A landmark-based neural classifier, so the system learns gestures instead of relying on hand-written rules.
- Multi-user session isolation and deployment on hospital intranet hardware.
- A clinical usability study measuring task time against a mouse or an assistant.

## 👤 Author

**Abid Ali**
Final Year Project, BS
GitHub: [@abidsabir1234](https://github.com/abidsabir1234)

---

<div align="center">
<sub>Built to make medical imaging more hygienic, faster and more intuitive.</sub>
</div>
