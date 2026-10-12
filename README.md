<div align="center">

# 🩻 Radiology Assistant

**A touchless, hand-gesture-controlled system for viewing and manipulating medical images**

*Final Year Project (2025–2026) · Computer Vision · Deep Learning · Human–Computer Interaction*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/CNN-MobileNetV2-4285F4)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-0097A7?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)

</div>

---

## Table of Contents

1. [Motivation](#-motivation)
2. [What It Does](#-what-it-does)
3. [Gesture Vocabulary](#-gesture-vocabulary)
4. [Deep Learning Model — MobileNetV2 Transfer Learning](#-deep-learning-model--mobilenetv2-transfer-learning)
5. [Real-Time Recognition Pipeline](#-real-time-recognition-pipeline)
6. [Image Transformation Engine](#-image-transformation-engine)
7. [System Architecture](#-system-architecture)
8. [Engineering Challenges & Solutions](#-engineering-challenges--solutions)
9. [Tech Stack](#-tech-stack)
10. [Project Structure](#-project-structure)
11. [Installation & Setup](#-installation--setup)
12. [Usage](#-usage)
13. [Future Work](#-future-work)
14. [Author](#-author)

---

## 💡 Motivation

During surgery and interventional procedures, clinicians often need to zoom into, rotate or adjust the contrast of X-ray, CT, MRI and ultrasound images. Touching a mouse, keyboard or touchscreen breaks the **sterile field**. As a result, surgeons usually rely on an assistant to operate the workstation for them, which is slow, error-prone and interrupts their concentration.

**Radiology Assistant** lets the clinician control the image directly with **hand gestures in front of an ordinary webcam**. No contact, special gloves or depth sensor is needed.

## 🎯 What It Does

- **Gesture recognition in real time** (about 30 FPS on a laptop CPU) from a standard webcam, or from a phone camera via Iriun Webcam.
- **Eight static hand gestures** mapped to image operations: zoom, rotation, brightness, blur, grayscale and reset.
- **Live dashboard**: an annotated camera feed (21-point hand skeleton, ROI box, predicted label and confidence) shown next to the live transformed scan.
- **Upload any scan** (PNG, JPG, BMP, TIFF, WEBP, GIF) and **download the transformed result** as a JPEG.
- **Secure authentication**: sign-up and login backed by MongoDB Atlas, passwords hashed with PBKDF2-SHA256, and session-protected routes.
- **Developer tooling**: a calibration tool for the CNN's class mapping and a live camera picker.

## ✋ Gesture Vocabulary

| Gesture | Hand Pose | Action | Mode |
|:---:|---|---|---|
| ✊ `ZERO` | Closed fist | **Reset** all transformations | Instant |
| ☝️ `ONE` | Index finger up | **Zoom in** (up to 3.0×) | Continuous while held |
| ✌️ `TWO` | Index + middle (V-sign) | **Rotate** +2° per frame | Continuous while held |
| 🤟 `THREE` | Index + middle + ring | **Gaussian blur** on/off | Toggle |
| 🖖 `FOUR` | Four fingers, thumb folded | **Zoom out** (down to 0.3×) | Continuous while held |
| 🖐️ `FIVE` | Open palm | **Grayscale** on/off | Toggle |
| 👍 `BRIGHT_UP` | Thumb up | **Brightness +** | Continuous while held |
| 🤙 `BRIGHT_DOWN` | Pinky up | **Brightness −** | Continuous while held |

*Continuous* gestures apply a small change on every frame while held, which gives smooth and controllable adjustment. *Toggle* gestures fire once each time the gesture is first made, so holding the gesture doesn't make the effect flicker on and off.

---

## 🧠 Deep Learning Model — MobileNetV2 Transfer Learning

The gesture CNN (`gesture_v2.h5`) is built with **TensorFlow / Keras 3** using **transfer learning** on top of **MobileNetV2**.

### Why MobileNetV2?

MobileNetV2 (Sandler et al., Google, 2018) is designed for **real-time inference on resource-constrained hardware**, which is exactly what a live webcam application running on a CPU needs.

- **Depthwise-separable convolutions** split a standard convolution into a per-channel *depthwise* filter and a 1×1 *pointwise* projection. This cuts computation by roughly 8–9× compared with standard 3×3 convolutions, with little loss in accuracy.
- **Inverted residual blocks** expand a thin bottleneck to a wider representation (expansion factor 6), filter it with a depthwise convolution, then project it back down. Shortcut connections join the thin bottlenecks, which keeps memory traffic low.
- **Linear bottlenecks** remove the non-linearity on the final projection of each block. This avoids destroying information in low-dimensional feature spaces, where ReLU would zero out useful signal.
- **ReLU6** activations are bounded to [0, 6], which makes the model stable and suitable for low-precision and quantised deployment.
- With about **2.2 M parameters** in the backbone, it is small enough to run every frame alongside MediaPipe and OpenCV without a GPU.

### Architecture

```
Input  (96 × 96 × 3, RGB, scaled to [0, 1])
  │
  ▼
MobileNetV2 backbone  (α = 1.0, input 96×96, include_top = False)
  • 17 inverted-residual bottleneck blocks
  • 17 DepthwiseConv2D layers + 1×1 Conv2D expansion / projection layers
  • BatchNormalization + ReLU6 throughout
  • ❄ Frozen — ImageNet-pretrained weights used as a fixed feature extractor
  │   output: 3 × 3 × 1280 feature map
  ▼
GlobalAveragePooling2D            → 1280-d feature vector
BatchNormalization
Dense(512, activation='relu')
Dropout(0.5)
Dense(256, activation='relu')
Dropout(0.3)
Dense(8, activation='softmax')    → probability over 8 gesture classes
```

| Component | Configuration |
|---|---|
| Framework | TensorFlow 2.x / Keras 3 |
| Backbone | `MobileNetV2(alpha=1.0, input_shape=(96, 96, 3), include_top=False)` |
| Backbone weights | ImageNet pre-trained, **frozen** (`trainable = False`) |
| Trainable head | GAP → BN → Dense 512 → Dropout 0.5 → Dense 256 → Dropout 0.3 → Dense 8 |
| Output | 8-way softmax |
| Optimizer | **Adam**, learning rate **5 × 10⁻⁴** |
| Loss | **Categorical cross-entropy** (one-hot labels) |
| Metric | Accuracy |
| Data pipeline | `ImageDataGenerator.flow_from_directory` (one folder per class) |
| Saved format | Keras HDF5 (`.h5`) |

### Design decisions

- **Transfer learning with a frozen backbone.** ImageNet features such as edges, contours and part shapes transfer well to hand silhouettes. Freezing the backbone means only the small head is trained, which converges quickly, needs far less data, and resists overfitting.
- **96×96 input.** This is the smallest resolution for which Keras ships pre-trained MobileNetV2 weights. It keeps per-frame inference cheap while still resolving individual fingers.
- **Two-stage dense head with decreasing dropout (0.5 → 0.3).** Strong regularisation where the feature vector is widest, lighter regularisation near the output.
- **BatchNormalization after pooling** stabilises the scale of the pooled features before the dense layers and speeds up head training.
- **A lower Adam learning rate (5e-4)** gives smoother convergence of the new head on top of fixed pre-trained features.
- **Class-index mapping.** `flow_from_directory` assigns class indices **alphabetically** (`BRIGHT_DOWN, BRIGHT_UP, FIVE, FOUR, ONE, THREE, TWO, ZERO`). `config.py` mirrors this order, and `calibrate.py` can automatically re-derive the mapping from live predictions if the model is retrained.

### Training data and the domain gap

The CNN was trained on an **infrared (Leap Motion) hand-gesture dataset**. Deployed on a standard RGB webcam, it showed a clear **domain shift**: IR training images are near-silhouettes under controlled lighting, while webcam frames are colour images with cluttered backgrounds and variable illumination. This finding shaped the final hybrid pipeline described next.

---

## ⚙️ Real-Time Recognition Pipeline

To get robust real-world performance, the system combines **learned landmark detection**, **geometric reasoning** and the **CNN**:

1. **Threaded capture**: a background thread continuously reads 640×480 frames, so the web server never blocks on camera I/O.
2. **Hand landmark detection**: **MediaPipe Hand Landmarker** (`hand_landmarker.task`) locates **21 hand keypoints**. It is trained on large and diverse RGB data, so it generalises to any webcam and lighting.
3. **ROI extraction**: a square region 1.5× the size of the hand bounding box is cropped and resized to **96×96**, then normalised to [0, 1].
4. **MobileNetV2 inference**: the ROI is passed through the CNN, producing an 8-class softmax score vector. The scores are streamed to the dashboard's live diagnostic panel (`/gesture_status`).
5. **Geometric gesture classification**: the final label comes from the landmarks. A finger counts as *extended* when its **tip-to-wrist distance exceeds 1.3× its PIP-joint-to-wrist distance** (1.4× for the thumb, whose range of motion is shorter). Because this uses distances, not raw coordinates, it is **rotation- and scale-invariant**. The resulting five-finger pattern maps directly to a gesture.
6. **Temporal smoothing**: a **5-frame majority vote** removes single-frame flicker and accidental transitions.
7. **Transformation and streaming**: the stable label drives the image transformer, and both feeds are streamed to the browser as **MJPEG** over HTTP.

This hybrid design keeps the result **accurate and explainable** under real webcam conditions, while the MobileNetV2 scores stay visible for analysis and future fine-tuning.

## 🖼️ Image Transformation Engine

`engine/transformer.py` keeps a persistent transform state and re-renders the scan with **OpenCV** in a fixed order:

| Step | Operation | Implementation |
|---|---|---|
| 1 | Grayscale | `cvtColor` BGR → GRAY → BGR |
| 2 | Gaussian blur | 15×15 kernel |
| 3 | Brightness | Shift of the **V channel in HSV space** (−100 … +100), which preserves hue |
| 4 | Rotation + zoom | A single **affine warp** (`getRotationMatrix2D` + `warpAffine`) with reflected borders |

A **frame cache** skips re-encoding when neither the gesture nor the image has changed, which greatly reduces idle CPU usage.

---

## 🏗️ System Architecture

```
 ┌──────────────┐  frames   ┌────────────────────────────┐  label  ┌────────────────────┐
 │ Webcam/Iriun │─────────▶│  GestureRecognizer          │───────▶│  ImageTransformer  │
 │ (bg thread)  │           │  • MediaPipe 21 landmarks   │         │  • grayscale       │
 └──────────────┘           │  • ROI → MobileNetV2 (Keras)│         │  • Gaussian blur   │
                            │  • geometric classifier     │         │  • HSV brightness  │
                            │  • 5-frame majority vote    │         │  • rotate + zoom   │
                            └─────────────┬──────────────┘         └─────────┬──────────┘
                                          │ annotated frame                  │ transformed scan
                                          ▼                                  ▼
                      ┌──────────────────────────────────────────────────────────────┐
                      │ Flask app factory + Blueprints                               │
                      │ /video_feed  /transform_feed  (MJPEG)   /gesture_status (JSON)│
                      │ /upload  /download_scan  /reset_transform  /login /signup    │
                      └────────────────────────────┬─────────────────────────────────┘
                                                   ▼
                             Browser dashboard (HTML · CSS · JavaScript)
                             Auth  ◀──▶  MongoDB Atlas (PBKDF2-SHA256)
```

## 🧩 Engineering Challenges & Solutions

| Challenge | Solution |
|---|---|
| CNN trained on IR data misclassified webcam frames (**domain shift**) | Hybrid pipeline: MediaPipe landmarks and geometric rules decide the label, while the MobileNetV2 scores are kept for diagnostics |
| Predictions flickered between neighbouring gestures | 5-frame sliding majority vote |
| Gestures failed when the hand was tilted | Wrist-relative **distance ratios** instead of y-coordinates (rotation-invariant) |
| Toggle actions fired repeatedly while a gesture was held | Edge-triggered toggles (fire only when the gesture changes) |
| Camera I/O blocked HTTP responses | Dedicated capture thread with lock-protected frame buffer |
| Phone webcams (Iriun) failed on some Windows backends | DirectShow → MSMF → ANY backend fallback, automatic index scan and auto-reconnect |
| High CPU usage when idle | Transform cache and adaptive sleep in the stream generators |
| Class order mismatch after retraining | `calibrate.py` re-learns the index → label mapping and saves it to `calibration_result.json` |

## 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| **Deep Learning** | TensorFlow, Keras 3, MobileNetV2 (transfer learning), NumPy |
| **Computer Vision** | OpenCV, MediaPipe Tasks — Hand Landmarker |
| **Backend** | Python 3.12, Flask (application factory, Blueprints), multithreading, MJPEG streaming |
| **Database & Security** | MongoDB Atlas, PyMongo, passlib (PBKDF2-SHA256), python-dotenv |
| **Frontend** | HTML5, CSS3, JavaScript, Jinja2 |

## 📂 Project Structure

```
Radialogy-Assistant/
├── app.py                   # Flask application factory & entry point
├── config.py                # Model path, input size, thresholds, gesture/action maps, env vars
├── gesture_v2.h5            # MobileNetV2-based Keras gesture classifier (8 classes, 96×96×3)
├── calibrate.py             # Interactive CNN class-mapping calibration tool
├── pick_camera.py           # Live multi-camera preview; saves chosen index to .env
├── requirements.txt
├── engine/
│   ├── camera.py            # Threaded capture, backend fallback, auto-reconnect
│   ├── recognizer.py        # MediaPipe landmarks + MobileNetV2 inference + classifier + voting
│   ├── transformer.py       # Stateful OpenCV transformation engine
│   └── hand_landmarker.task # MediaPipe hand landmark model
├── routes/
│   ├── main.py              # Dashboard, MJPEG streams, upload/download, status API
│   └── auth.py              # Sign-up / login / logout
├── database/
│   └── auth.py              # MongoDB user store with hashed passwords
├── templates/               # Jinja2 pages: home, about, contact, dashboard, auth/
└── static/                  # CSS, gesture illustrations, sample X-ray / CT / MRI / ultrasound scans
```

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.12**
- A webcam, or a smartphone running [Iriun Webcam](https://iriun.com/)
- A free [MongoDB Atlas](https://www.mongodb.com/atlas) cluster

### 1. Clone and install dependencies

```bash
git clone https://github.com/abidsabir1234/Radialogy-Assistant.git
cd Radialogy-Assistant

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
MONGODB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
SECRET_KEY=<random-hex-string>
CAMERA_INDEX=0
DEBUG_SCORES=true
```

Generate a secret key:

```bash
python -c "import secrets; print(secrets.token_hex(24))"
```

> The database `Radiology_Assistant` and collection `Account` are created automatically on the first sign-up.

### 3. (Optional) Select a camera

```bash
python pick_camera.py
```

### 4. (Optional) Calibrate the CNN class mapping

```bash
python calibrate.py
```

### 5. Run the application

```bash
python app.py
```

Open **http://127.0.0.1:5000**, create an account, upload a scan on the dashboard, and control it with your hand.

## 🖐️ Usage

- Keep your hand **30–60 cm** from the camera, palm facing it.
- Use even, front-facing light. Avoid strong backlight.
- Hold each gesture steadily for about **0.2 s**, so the 5-frame vote can lock on.
- Make a **fist** at any time to reset the image.
- Click **Download** to save the transformed scan.

## 🔭 Future Work

- **Fine-tune MobileNetV2 on RGB webcam data**: unfreeze the top inverted-residual blocks and train with a low learning rate, to close the IR→RGB domain gap so the CNN can serve as the primary classifier.
- **Landmark-based neural classifier**: a small MLP or 1-D CNN over the 21×3 landmark coordinates.
- **Model optimisation**: TensorFlow Lite conversion with INT8 post-training quantisation for edge devices.
- **Dynamic gestures**: swipe and pinch recognition with temporal models (LSTM / temporal convolution) for scrolling through CT and MRI slice stacks.
- **DICOM support** with window/level control for CT Hounsfield units.
- A **clinical usability study** comparing task completion time against mouse and assistant-operated workflows.

## 👤 Author

**Abid Ali**
Final Year Project, 2025–2026
GitHub: [@abidsabir1234](https://github.com/abidsabir1234)

---

<div align="center">
<sub>Making medical imaging more hygienic, faster and more intuitive.</sub>
</div>
