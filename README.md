# 🔬 Kinetic Forensics — AI-Powered Deepfake Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Run%20%2B%20Functions-4285F4?logo=googlecloud&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Hosting%20%2B%20Firestore%20%2B%20Storage-FFCA28?logo=firebase&logoColor=black)
![Gemini](https://img.shields.io/badge/Gemini-Vertex%20AI%20%7C%20AI%20Studio-8E44AD?logo=google&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarker-0097A7?logo=google&logoColor=white)

**A physics-based deepfake detection system that ignores surface pixels and instead analyzes the laws of motion.**

*"Generative AI has perfected visual realism. But it cannot perfectly simulate physics."*

[Architecture](#-architecture) • [Cloud Deployment](#-production-cloud-deployment) • [Local Setup](#-run-locally-zero-cost) • [API Reference](#-api-reference) • [How It Works](#-how-the-physics-engine-works)

</div>

---

## 🎯 The Problem This Solves

Traditional deepfake detectors look for **visual artifacts** — blurry edges, mismatched lighting, face warping. But modern generative AI (diffusion models, GAN-based face-swappers) has made these detectors obsolete. The fakes look perfect.

**Kinetic Forensics takes a fundamentally different approach:**

> Instead of asking *"does this face look real?"*, we ask *"does this face move like a real human?"*

Human faces are governed by **biology and physics**. They have mass, inertia, biological micro-tremors, and involuntary eye movements (saccades). AI-generated faces, no matter how realistic, consistently fail to replicate these exact physical signatures. We call these failures **"Kinetic Dissonance."**

---

## 🏗️ Architecture

### Production Cloud Architecture (GCP + Firebase)

This system was **fully deployed and operational** on Google Cloud Platform during the Google AI Challenge. The complete event-driven cloud pipeline is:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION CLOUD STACK                          │
└─────────────────────────────────────────────────────────────────────┘

User Browser (Firebase Hosting)
        │
        │  HTTPS POST /analyze  (video file upload)
        ▼
┌──────────────────────────────────────────────────┐
│  FastAPI Backend  ──  Google Cloud Run           │
│  • Containerized Python (Docker)                 │
│  • MediaPipe Face Landmarker (478-point 3D mesh) │
│  • Extracts: chin, iris centers, eyebrows        │
│  • Output: timestamped JSON landmark array       │
└────────────┬─────────────────────────────────────┘
             │
             │  Upload (video + landmarks JSON)
             ▼
┌──────────────────────────────────────────────────┐
│  Firebase Cloud Storage                          │
│  analyzed_videos/<timestamp>.mp4                 │
│  analyzed_videos/<timestamp>_landmarks.json      │
└────────────┬─────────────────────────────────────┘
             │
             │  onObjectFinalized trigger (landmarks.json)
             ▼
┌──────────────────────────────────────────────────┐
│  Firebase Cloud Functions (Node.js v2)           │
│  • Downloads landmark JSON from Storage          │
│  • Samples frames intelligently (max 240)        │
│  • Calls Vertex AI with video + landmark prompt  │
└────────────┬─────────────────────────────────────┘
             │
             │  Multi-modal prompt (video + landmark JSON)
             ▼
┌──────────────────────────────────────────────────┐
│  Google Vertex AI — Gemini 3.1 Pro               │
│  System Persona: Forensic Biometrics Analyst     │
│  Analyzes: chin acceleration, saccadic gaze,     │
│            biological jitter, temporal lag       │
│  Output: { authenticity_score, anomalies,        │
│            forensic_explanation }                │
└────────────┬─────────────────────────────────────┘
             │
             │  Write to Firestore "analyses" collection
             ▼
┌──────────────────────────────────────────────────┐
│  Cloud Firestore (NoSQL Database)                │
│  Document: { video_reference, analysis,          │
│              total_frames, analyzed_at }         │
└────────────┬─────────────────────────────────────┘
             │
             │  Real-time polling (4s intervals, 5min timeout)
             ▼
User Browser — Results Dashboard
  • Deepfake Probability Gauge
  • Kinetic Jitter SVG Time-Series Chart
  • AI Forensic Explanation (typewriter animation)
```

### Dual-Mode Architecture (v2.0)

After the GCP free trial expired, the system was upgraded to support **graceful degradation** — a pattern used in production-grade distributed systems. It now operates identically in two modes without any code changes:

```
┌─────────────────────────────────────────────────────────────┐
│                     MODE SELECTION                          │
│                                                             │
│  Environment Variables Set?   →   Mode Used                 │
│  ─────────────────────────────────────────────             │
│  gcloud ADC + GCP Project     →   Production Cloud Mode    │
│  GEMINI_API_KEY only          →   AI Studio Mode (free)    │
│  Nothing set                  →   Physics Heuristic Mode   │
└─────────────────────────────────────────────────────────────┘
```

| Component | Production Cloud | AI Studio Mode | Heuristic Mode |
|---|---|---|---|
| **Compute** | Google Cloud Run | Local (any machine) | Local |
| **AI Engine** | Vertex AI Gemini 3.1 Pro | Gemini 2.5 Flash | Physics algorithm |
| **Storage** | Firebase Cloud Storage | Firebase (if avail.) | Local temp files |
| **Database** | Cloud Firestore | Firestore (if avail.) | Direct API response |
| **Frontend** | Firebase Hosting | `npm run dev` | `npm run dev` |
| **Cost** | GCP billing required | Free tier | Zero cost |
| **Analysis Quality** | Full multi-modal AI | High quality AI | Deterministic physics |

---

## 🔬 How the Physics Engine Works

### The Four Laws of Kinetic Forensics

Human movement is governed by biology that AI models struggle to perfectly replicate:

**1. The Law of Inertia — Chin & Jaw Velocity Variance**
```
Human head movements have mass. Acceleration = Δv / Δt must follow physical limits.
AI-generated faces exhibit "weightless" transitions — instantaneous position changes
that violate Newton's second law. We measure:

  jitter_ratio = std_dev(chin_velocity) / mean(chin_velocity)

  Real human:  jitter_ratio > 0.15  (noisy, organic)
  AI deepfake: jitter_ratio < 0.15  (suspiciously smooth)
```

**2. Saccadic Eye Movement — Main Sequence Violations**
```
Real human eyes move in discrete, high-velocity jumps (saccades), NOT linear slides.
AI-generated irises interpolate linearly between positions.

  We count saccade_peaks where |Δeye_position| > 0.005 per frame.
  Genuine: multiple large discrete jumps
  Synthetic: continuous linear motion (saccade_peaks < 2 over 30+ frames)
```

**3. Biological Noise — Micro-Tremor Presence (3–7 Hz)**
```
Real humans have involuntary micro-tremors from muscle activity.
Perfect smoothness in landmark trajectories = synthetic interpolation.
```

**4. Temporal Continuity — Face Detection Rate**
```
Face-swap models often produce warping artifacts causing frame-level face-loss.
Detection rate < 70% across frames flags synthetic boundary failures.
```

### Data Pipeline

```
Video File (MP4, MOV, WebM)
      │
      ▼  [OpenCV — frame-by-frame decode]
BGR Frame Buffer
      │
      ▼  [cv2.cvtColor — BGR→RGB]
RGB Image
      │
      ▼  [MediaPipe FaceLandmarker.detect_for_video()]
478 3D Normalized Landmarks (x, y, z ∈ [0.0, 1.0])
      │
      ▼  [Landmark Selection — 5 key anatomical points]
  • Chin        (index 152)
  • Left Iris   (indices 468–472, averaged to center)
  • Right Iris  (indices 473–477, averaged to center)
  • Left Eyebrow  (index 70)
  • Right Eyebrow (index 300)
      │
      ▼  [Optional EMA Smoothing — α configurable]
      │
      ▼  [Optional Velocity — Euclidean distance per frame]
      │
      ▼  [JSON Serialization]
output_landmarks.json  →  Gemini AI Analysis / Physics Heuristic
```

---

## ☁️ Production Cloud Deployment

### Deploy Backend to Cloud Run

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project deepfake-detector-494710

# Build and deploy (Docker auto-detected from Dockerfile)
gcloud run deploy sensor-backend \
  --source . \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT_ID=deepfake-detector-494710

# Configure CORS for Cloud Storage
gsutil cors set cors.json gs://deepfake-detector-494710.firebasestorage.app
```

### Deploy Firebase Functions & Hosting

```bash
# Install Firebase CLI
npm install -g firebase-tools
firebase login

# Deploy Cloud Functions (Gemini analysis trigger)
firebase deploy --only functions

# Deploy frontend to Firebase Hosting
cd Frontend && npm run build && cd ..
firebase deploy --only hosting
```

### Live Production URLs (when GCP billing active)
- **Frontend**: `https://deepfake-detector-494710.web.app`
- **Backend API**: `https://sensor-backend-521504670907.asia-southeast1.run.app`

---

## 💻 Run Locally (Zero Cost)

You can run the complete deepfake detection system on your local machine in under 5 minutes with **no GCP billing, no Firebase project, no cloud setup required.**

### Prerequisites

- Python 3.8+
- Node.js 18+ and npm

### Step 1 — Backend (FastAPI + MediaPipe)

```bash
# Clone and enter project
git clone https://github.com/RiteeshTM/sensor-module
cd sensor-module

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# The face_landmarker.task model (30MB) is already included.
# If missing, it auto-downloads on first run.

# ─── OPTION A: Free Gemini AI via Google AI Studio ────────────────
# Get a free API key at: https://aistudio.google.com/apikey
$env:GEMINI_API_KEY="your-api-key-here"   # Windows PowerShell
# export GEMINI_API_KEY="your-api-key-here"  # macOS/Linux

# ─── OPTION B: No API key — runs physics heuristic engine ─────────
# (No environment variables needed at all)

# Start the server
python server.py
# → Server running at http://localhost:8080
# → Visit http://localhost:8080 to see mode (Firebase/Gemini status)
```

### Step 2 — Frontend (React + Vite)

```bash
# In a new terminal
cd Frontend
npm install
npm run dev
# → App running at http://localhost:5173
```

Open `http://localhost:5173`, upload any face video (< 60 seconds), and click **Analyze Video**.

### How the Frontend Auto-Detects Local Mode

The React app intelligently routes API calls:
```javascript
// Auto-detects localhost → routes to local backend
// No config.js changes needed
const BACKEND_URL = window.location.hostname === "localhost"
  ? "http://localhost:8080"          // Local FastAPI server
  : "https://sensor-backend-...run.app";  // Cloud Run (production)
```

The backend returns results **directly in the HTTP response** when running locally — no Firestore polling needed.

### Verify Your Backend Mode

```bash
curl http://localhost:8080/
# {
#   "status": "running",
#   "firebase": "unavailable (local mode)",
#   "analysis_engine": "Google AI Studio (gemini-2.5-flash)",
#   "version": "2.0.0"
# }
```

---

## 📡 API Reference

### `GET /`
Health check and mode status.

**Response:**
```json
{
  "status": "running",
  "firebase": "connected | unavailable (local mode)",
  "analysis_engine": "Google AI Studio (gemini-2.5-flash) | Vertex AI (gemini-3.1-pro) | Local Heuristic Mock",
  "version": "2.0.0"
}
```

### `POST /analyze`
Upload a video for deepfake analysis. Supports both Cloud and Local modes transparently.

**Request:**
```bash
curl -X POST http://localhost:8080/analyze \
  -F "video=@your_video.mp4"
```

**Response (Local Mode — immediate):**
```json
{
  "message": "Analysis completed successfully",
  "videoUri": null,
  "result": {
    "probability": 23.5,
    "confidence": 76.5,
    "framesAnalyzed": 412,
    "status": "Real",
    "report": "Physics-based kinetic forensics analysis of 412 frames..."
  }
}
```

**Response (Cloud Mode — triggers async pipeline):**
```json
{
  "message": "Analysis completed successfully",
  "videoUri": "gs://deepfake-detector-494710.firebasestorage.app/analyzed_videos/backend_20250428_142530.mp4",
  "result": { ... }
}
```

---

## 🗂️ Project Structure

```
sensor-module/
│
├── 📄 sensor.py               Core MediaPipe landmark extraction engine
│                              CLI + importable module
│
├── 📄 server.py               FastAPI REST API server
│                              Handles video upload, orchestrates pipeline,
│                              returns results directly or via Firebase
│
├── 📄 deepfake_detection.py   AI analysis engine (3-tier fallback)
│                              1. Google AI Studio (GEMINI_API_KEY)
│                              2. Vertex AI (GCP credentials)
│                              3. Physics heuristic (no API needed)
│
├── 📄 firebase_utils.py       Firebase Admin SDK wrapper
│                              Graceful fallback if credentials absent
│
├── 📄 main.py                 CLI end-to-end workflow script
│
├── 📄 Dockerfile              Cloud Run container definition
│
├── 📁 Frontend/               React + Vite web application
│   ├── src/App.jsx            Main app (upload, analysis, results UI)
│   ├── src/components/        Typewriter effect component
│   ├── src/index.css          Dark glassmorphism design system
│   └── public/config.js       Runtime backend URL override
│
├── 📁 functions/              Firebase Cloud Functions (Node.js)
│   └── index.js               onObjectFinalized → Gemini analysis trigger
│
├── 📄 firebase.json           Firebase project config (Hosting + Functions)
├── 📄 storage.rules           Firebase Storage security rules
├── 📄 cors.json               GCS CORS config for browser uploads
├── 📄 pyproject.toml          Python project metadata (uv compatible)
└── 📄 requirements.txt        Python dependencies
```

---

## 🧪 Sensor Module CLI Usage

The `sensor.py` module works as a standalone CLI tool for landmark extraction, independent of the web stack:

```bash
# Basic extraction
python sensor.py input.mp4

# With all options
python sensor.py input.mp4 \
  --output landmarks.json \
  --smoothing-alpha 0.7 \
  --include-velocity \
  --verbose

# Batch processing
Get-ChildItem *.mp4 | ForEach-Object { python sensor.py $_.Name }
```

**Output Format (per frame):**
```json
[
  {
    "frame": 0,
    "time": 0.0,
    "face_detected": true,
    "chin":           [0.500, 0.642, -0.021],
    "left_eye":       [0.349, 0.401,  0.001],
    "right_eye":      [0.651, 0.401,  0.001],
    "left_eyebrow":   [0.299, 0.252,  0.000],
    "right_eyebrow":  [0.701, 0.252,  0.000],
    "chin_velocity":  0.0043
  }
]
```

All coordinates are normalized `[0.0, 1.0]`. `z` is a depth estimate (negative = closer to camera).

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description | Default |
|---|---|---|---|
| `GEMINI_API_KEY` | No | Google AI Studio key (free) — enables Gemini 2.5 Flash without GCP billing | *not set → physics mode* |
| `GOOGLE_CLOUD_PROJECT_ID` | No | GCP project ID for Vertex AI and Firebase | `deepfake-detector-494710` |

### Sensor CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `face_landmarker.task` | Path to MediaPipe model file |
| `--output` | `<input>_landmarks.json` | Output JSON path |
| `--smoothing-alpha` | `0.8` | EMA filter strength `[0.0–1.0]`. `0` = raw, `0.95` = very smooth |
| `--include-velocity` | off | Append `chin_velocity` field (units/second) |
| `--verbose` | off | Print FPS, frame count, detection stats |

---

## 🛠️ Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | React 18 + Vite | SPA with drag-and-drop video upload |
| **Animations** | Framer Motion | Smooth result reveals, scanning line |
| **Icons** | Lucide React | Shield, Alert, CheckCircle icons |
| **Backend** | FastAPI + Uvicorn | Async REST API, multipart video upload |
| **CV Engine** | MediaPipe Face Landmarker | 478-point 3D facial landmark extraction |
| **Video Decode** | OpenCV (headless) | Frame-by-frame BGR→RGB pipeline |
| **AI Analysis** | Google Gemini (multi-modal) | Video + landmark reasoning (Vertex AI / AI Studio) |
| **Hosting** | Firebase Hosting | CDN-backed static frontend |
| **Compute** | Google Cloud Run | Serverless, auto-scaling containerized backend |
| **Functions** | Firebase Cloud Functions v2 | Event-driven Gemini trigger |
| **Database** | Cloud Firestore | Real-time NoSQL results storage |
| **Storage** | Firebase Cloud Storage | Video and landmark file persistence |
| **Containers** | Docker | Cloud Run deployment packaging |
| **Auth** | GCP Application Default Credentials | Secure service-to-service auth |

---

## 📊 Performance

| Hardware | Landmark Extraction FPS | Memory Usage |
|---|---|---|
| M1/M2 MacBook | 45–50 fps | ~250 MB |
| Intel i7 (10th gen) | 30–35 fps | ~300 MB |
| Intel i5 (8th gen) | 15–20 fps | ~350 MB |
| Google Colab (CPU) | 25–30 fps | ~400 MB |

**Model sizes:**
- `face_landmarker.task`: ~30 MB (MediaPipe, Float16, included in repo)
- Docker image: ~2.1 GB (Python + OpenCV + MediaPipe layers)

---

## 🔧 Troubleshooting

### Backend won't start
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Check face_landmarker.task exists
ls face_landmarker.task

# If missing, download manually:
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### CORS errors in browser
The FastAPI CORS middleware allows `localhost:5173` by default. If you're running the frontend on a different port, add it to the `allow_origins` list in `server.py`.

### Firebase init warning on startup
```
Warning: Firebase initialization failed: ...
The application will run in local/standalone mode
```
This is **expected and intentional** when running without GCP credentials. The system will still function fully using the physics engine or Google AI Studio.

### No faces detected
- Ensure faces are clearly visible and frontal
- Recommended resolution: 640×480 or higher
- Recommended frame rate: 24+ fps

---

## 🏛️ Design Decisions

### Why MediaPipe Face Landmarker (not Face Mesh)?
| Aspect | Face Landmarker | Face Mesh (deprecated) |
|---|---|---|
| API Type | Tasks API (2024+) | Solutions API (legacy) |
| Built-in tracking | ✅ | ❌ Manual |
| Performance | Better | Slower |
| Maintenance | Active | Deprecated |

### Why Iris Averaging?
MediaPipe outputs 5 perimeter points per iris. Averaging gives a **stable pupil center** with reduced jitter — essential for velocity analysis:
```
Raw iris points: [0.348, 0.402, 0.001], [0.349, 0.401, 0.003], ... (high variance)
Averaged center: [0.350, 0.401, 0.001]  (stable)
```

### Why Optional EMA Smoothing?
Raw MediaPipe output has inherent ±2–3 pixel jitter. EMA with `α=0.8` preserves real motion while suppressing measurement noise. Making it optional preserves the raw data for research pipelines.

### Why Preserve All Frames (Including Misses)?
Skipping frames without detected faces breaks time-series alignment. By recording `face_detected: false` with `null` landmark values, downstream models can interpolate, detect blink events, and maintain frame-accurate correspondence.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built for the **Google AI Challenge** · Powered by **Google Gemini** · Deployed on **Google Cloud**

*"In a world where pixels lie, physics cannot."*

</div>
