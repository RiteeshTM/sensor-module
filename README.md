# Kinetic Forensics — Physics-Based Deepfake Detection

A deepfake detector that ignores how a face *looks* and analyses how it *moves*.

Generative models have solved visual realism. They have not solved physics. Real faces have
mass, inertia, involuntary micro-tremors and discrete saccadic eye jumps. Synthesised faces
interpolate. This project extracts 3D facial landmarks frame by frame, measures the motion,
and reports where that motion stops behaving like biology.

Everything runs on a laptop with no cloud account, no API key and no billing. Add a free
Gemini key for AI-assisted analysis, or full GCP credentials for the original cloud pipeline —
the same code handles all three.

---

## Table of contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Operating modes](#operating-modes)
- [Local setup (detailed)](#local-setup-detailed)
- [Cloud setup (detailed)](#cloud-setup-detailed)
- [Reading the results](#reading-the-results)
- [API reference](#api-reference)
- [Command-line tools](#command-line-tools)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [License](#license)

---

## Quick start

Two terminals, about five minutes. No account of any kind is required.

**Prerequisites:** Python 3.10+ and Node.js 18+.

### Terminal 1 — backend

```bash
git clone https://github.com/RiteeshTM/sensor-module
cd sensor-module

python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / CMD
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python server.py
```

The backend starts on **http://localhost:8000**. On the first run it downloads the MediaPipe
Face Landmarker model (`face_landmarker.task`, ~3.7 MB) into the project folder.

Confirm it is healthy and see which analysis engine is active:

```bash
curl http://localhost:8000/
```

```json
{
  "status": "running",
  "version": "2.1.0",
  "firebase": "unavailable (local mode)",
  "analysis_engine": "Local physics heuristic (kinetic-forensics-v2)",
  "engine": "Local physics heuristic",
  "model": "kinetic-forensics-v2",
  "model_file_present": true,
  "max_upload_mb": 50.0
}
```

### Terminal 2 — frontend

```bash
cd Frontend
npm install
npm run dev
```

Open **http://localhost:5173**, drop in a video of a face (under 60 seconds and 50 MB), and
press **Analyze Video**.

> The frontend hard-codes `http://localhost:8000` whenever it is served from `localhost` or
> `127.0.0.1`, so a dev session can never accidentally hit the production backend. No config
> file needs editing.

### Optional — turn on free Gemini analysis

Grab a key from [Google AI Studio](https://aistudio.google.com/apikey) (free tier, no billing
account), then restart the backend with it set:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"

# macOS / Linux
export GEMINI_API_KEY="your-key-here"

python server.py
```

`GET /` will now report `Google AI Studio (gemini-2.5-flash)`. Nothing else changes.

---

## How it works

### The four kinetic signals

| Signal | What is measured | What a fake looks like |
|---|---|---|
| **Inertia** | Frame-to-frame chin velocity, then its variance | Unnaturally smooth motion — mass-less transitions |
| **Biological noise** | `jitter_ratio = σ(velocity) / μ(velocity)` | `< 0.15` — the 3–7 Hz micro-tremor is missing |
| **Saccades** | Count of per-frame eye displacements above 0.005 | Near zero — gaze slides linearly instead of jumping |
| **Temporal continuity** | Share of frames with a detected face | `< 70%` — face-swap warping breaks detection |

### Pipeline

```
video.mp4
    |
    v  OpenCV — decode frame by frame, BGR -> RGB
RGB frames
    |
    v  MediaPipe Face Landmarker (VIDEO mode) — 478 3D landmarks per frame
    |
    v  Select 5 anatomical points
       chin (152) · left iris (468-472, averaged) · right iris (473-477, averaged)
       left eyebrow (70) · right eyebrow (300)
    |
    v  Optional EMA smoothing (alpha = 0.8 by default)
    v  Per-frame chin velocity
    |
    v  landmarks.json  ->  analysis_utils.compute_metrics()
    |                        jitter ratio, detection rate, saccade count,
    |                        and the per-frame velocity series
    |
    v  Analysis engine (AI Studio / Vertex AI / local physics heuristic)
    |
    v  { authenticity_score, flagged_anomalies, forensic_explanation }
    |
    v  Frontend: gauge, real metrics, and a chart of the actual velocity series
```

**Why iris averaging?** MediaPipe returns 5 perimeter points per iris. Averaging them yields a
stable pupil centroid, which matters because the saccade test measures small displacements.

**Why keep frames with no face?** They are recorded as `face_detected: false` with `null`
coordinates rather than dropped. Dropping them would silently compress the timeline and
destroy the velocity calculation; keeping them makes the detection rate measurable.

**Why is the chart real?** Every point plotted is a measured chin velocity from the uploaded
clip, downsampled by bucket-averaging to 160 points. The dashed line is the measured mean and
the shaded band is ±1σ — the same numbers that drive the verdict.

---

## Operating modes

The backend picks a mode at startup by looking at the environment. There is no flag to set and
no code to change.

| | Environment | Analysis engine | Storage | Results reach the browser via |
|---|---|---|---|---|
| **Local (default)** | nothing set | Local physics heuristic | none | HTTP response |
| **AI Studio** | `GEMINI_API_KEY` | `gemini-2.5-flash` | none (or Firebase if available) | HTTP response |
| **Cloud** | GCP credentials | Vertex AI `gemini-3.1-pro-preview` | Cloud Storage + Firestore | HTTP response, plus Firestore |

Fallback is automatic and one-directional: if a Gemini call fails (quota, network, auth), the
request does **not** fail. The local physics engine produces the verdict, the response sets
`usedFallback: true`, and the UI says so explicitly.

`GET /` always reports the engine that is currently active, and every verdict carries the
engine that actually produced it. The UI never claims a model it did not use.

---

## Local setup (detailed)

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

`google-genai` and `firebase-admin` are listed in `requirements.txt` but are **optional at
runtime** — the app degrades to the physics engine if they are missing or unconfigured.

### 2. The MediaPipe model

`face_landmarker.task` is **not** committed to the repository (it is in `.gitignore`). It
downloads automatically on first use. To fetch it manually:

```bash
curl -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 3. Run the backend

```bash
python server.py                 # http://localhost:8000, auto-reload on
```

Or with uvicorn directly, which is what the Docker image uses:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Set `PORT` to use a different port. If you do, also update the localhost URL in
`Frontend/src/App.jsx`.

### 4. Run the frontend

```bash
cd Frontend
npm install
npm run dev                      # http://localhost:5173
```

To preview a production bundle locally:

```bash
npm run build && npm run preview
```

### 5. Optional: environment file

Copy `.env.example` to `.env` and edit it. Load it however you prefer — `python-dotenv`,
`direnv`, or your shell — the app reads plain environment variables.

---

## Cloud setup (detailed)

> The hosted deployment described here is currently **suspended** (the GCP trial expired).
> These instructions are complete and correct for anyone bringing it back up, but nothing in
> the local path depends on any of it.

### Architecture

```
Browser (Firebase Hosting)
    |  POST /analyze
    v
FastAPI on Cloud Run  -->  MediaPipe landmark extraction
    |                              |
    |  upload video + landmarks    |
    v                              v
Firebase Cloud Storage  --onObjectFinalized-->  Cloud Function (Node 22)
                                                       |
                                                       v
                                            Vertex AI gemini-3.1-pro-preview
                                                       |
                                                       v
                                            Firestore "analyses" collection
                                                       |
                            Browser polls every 4s <---+
```

The Cloud Function triggers on any object ending in `_landmarks.json` and derives the sibling
video path by swapping that suffix for `.mp4`. The two filenames must stay in lock-step —
`firebase_utils.upload_file_to_storage()` is what guarantees that.

### 1. Prerequisites

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

npm install -g firebase-tools
firebase login
```

Enable the APIs: Cloud Run, Cloud Build, Vertex AI, Cloud Functions, Firestore, Cloud Storage.

Replace `deepfake-detector-494710` with your own project ID in `.firebaserc`, `cors.json` and
`Frontend/public/config.js`, and set `GOOGLE_CLOUD_PROJECT_ID` in the deploy commands below.

### 2. Deploy the backend to Cloud Run

```bash
gcloud run deploy sensor-backend \
  --source . \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 600 \
  --set-env-vars GOOGLE_CLOUD_PROJECT_ID=YOUR_PROJECT_ID
```

The container listens on `$PORT` (Cloud Run sets 8080; the `Dockerfile` matches).

> **Important:** `.gcloudignore` and `.dockerignore` control what gets uploaded. Every Python
> module the server imports — `server.py`, `sensor.py`, `analysis_utils.py`,
> `deepfake_detection.py`, `firebase_utils.py` — must stay out of those ignore files, or the
> container builds successfully and then fails at request time with `ImportError`.

### 3. Configure Storage CORS

```bash
gsutil cors set cors.json gs://YOUR_PROJECT_ID.firebasestorage.app
```

### 4. Deploy the Cloud Function

```bash
cd functions && npm install && cd ..
firebase deploy --only functions
```

### 5. Build and deploy the frontend

Firebase Hosting serves the **`public/`** directory, but Vite builds into **`Frontend/dist/`**.
Nothing copies between them automatically, so this step is mandatory — skipping it deploys a
stale bundle:

```bash
cd Frontend
npm run build
cd ..

# Windows PowerShell
Remove-Item -Recurse -Force public\assets -ErrorAction SilentlyContinue
Copy-Item -Recurse -Force Frontend\dist\* public\

# macOS / Linux
rm -rf public/assets && cp -r Frontend/dist/* public/

firebase deploy --only hosting
```

Point the deployed frontend at your backend by editing `public/config.js` after the copy — it
is read at runtime, so no rebuild is needed:

```javascript
window.__APP_CONFIG__ = { BACKEND_URL: "https://your-backend-url.run.app" };
```

### 6. Lock down Storage before going public

`storage.rules` currently ships as `allow read, write: if true`, which is fine for a demo and
unacceptable for anything long-lived. Restrict it before exposing a real deployment.

---

## Reading the results

| Field shown in the UI | Meaning | Rough interpretation |
|---|---|---|
| **Deepfake Probability** | `100 − authenticity_score` | ≥ 50% is reported as *Fake* |
| **Confidence** | Distance of the verdict from a coin flip | Higher is a more decisive call |
| **Frames Analyzed** | Frames decoded and passed through MediaPipe | — |
| **Face Detection Rate** | Frames with a located face | < 70% is flagged |
| **Kinetic Jitter (σ/μ)** | Chin-velocity noise ratio | < 0.15 too smooth · > 1.5 flickering · 0.15–1.5 normal |
| **Saccadic Events** | Discrete eye jumps above threshold | Near zero over a long clip is suspicious |
| **Analysis Engine** | Which tier produced the verdict | Heuristic · AI Studio · Vertex AI |

### The "Inconclusive" verdict

If the analysis engine returns something the server cannot parse into a numeric score, the
result is **Inconclusive** — probability and confidence are `null` and the UI shows `N/A`. It
is deliberately not treated as "Real". The measured kinetic metrics and the velocity chart are
still shown, because those come from the landmark data and never depend on the model replying
correctly.

---

## API reference

Base URL: `http://localhost:8000` locally, or your Cloud Run URL.

### `GET /` and `GET /health`

Health check plus active-mode report. Both paths return the same body (see
[Quick start](#quick-start) for a sample).

### `POST /analyze`

Multipart upload of a single video under the field name `video`.

```bash
curl -X POST http://localhost:8000/analyze -F "video=@sample.mp4"
```

**200 — success**

```json
{
  "message": "Analysis completed successfully",
  "videoUri": null,
  "result": {
    "status": "Real",
    "probability": 15.0,
    "confidence": 85.0,
    "authenticityScore": 85.0,
    "engine": "Local physics heuristic",
    "model": "kinetic-forensics-v2",
    "usedFallback": false,
    "framesAnalyzed": 412,
    "framesWithFace": 408,
    "detectionRate": 0.9903,
    "jitterRatio": 0.5583,
    "meanVelocity": 0.109353,
    "stdVelocity": 0.061056,
    "saccadePeaks": 9,
    "jitterSeries": [{ "t": 0.0, "v": 0.0 }, { "t": 0.033, "v": 0.1399 }],
    "anomalies": ["..."],
    "report": "Physics-based kinetic forensics analysis of 412 frames..."
  }
}
```

`videoUri` is a `gs://` path when Firebase is connected, and `null` in local mode.
`jitterSeries` is the per-frame chin velocity, bucket-averaged to at most 160 points — this is
what the chart plots.

**Error responses**

| Status | Condition |
|---|---|
| `400` | Empty upload |
| `413` | File exceeds `MAX_UPLOAD_BYTES` (50 MB default) |
| `422` | No face detected in any frame |
| `500` | Unexpected processing error (message in `error`) |

Each request works in its own temporary directory, which is removed on the way out —
concurrent uploads cannot collide and nothing is left behind.

---

## Command-line tools

### `sensor.py` — landmark extraction

Standalone and independent of the web stack.

```bash
python sensor.py input.mp4
python sensor.py input.mp4 --output landmarks.json --include-velocity --verbose
python sensor.py input.mp4 --smoothing-alpha 0    # raw, unsmoothed output
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `face_landmarker.task` | Model path; downloaded if missing |
| `--output` | `<input>_landmarks.json` | Output JSON path |
| `--smoothing-alpha` | `0.8` | EMA strength, `0.0`–`1.0`. `0` disables smoothing |
| `--include-velocity` | off | Add `chin_velocity` (normalized units/second) |
| `--verbose` | off | Log FPS, frame count and progress |

Output, one record per frame:

```json
{
  "frame": 42,
  "time": 1.4,
  "face_detected": true,
  "chin": [0.501, 0.641, -0.020],
  "left_eye": [0.349, 0.400, 0.001],
  "right_eye": [0.651, 0.400, 0.001],
  "left_eyebrow": [0.299, 0.251, 0.000],
  "right_eyebrow": [0.701, 0.251, 0.000],
  "chin_velocity": 0.0043
}
```

Coordinates are normalized to `[0.0, 1.0]`; `z` is a relative depth estimate (more negative is
closer to the camera). Frames with no face keep their slot with `face_detected: false` and
`null` coordinates.

### `main.py` — full pipeline in one command

```bash
python main.py sample.mp4
python main.py sample.mp4 --landmarks existing_landmarks.json
python main.py sample.mp4 --no-upload      # never touch Firebase
```

Extracts landmarks, runs the best available engine, prints the metrics and the verdict, and
backs up to Firebase only when credentials exist and `--no-upload` was not passed.

### `deepfake_detection.py` — analysis only

```bash
python deepfake_detection.py sample.mp4 sample_landmarks.json
```

Runs the tier-selection logic against landmarks you already have and prints which engine
answered.

---

## Configuration

Every variable is optional. With none of them set, the app runs entirely locally.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | unset | Google AI Studio key. Enables `gemini-2.5-flash` |
| `GOOGLE_CLOUD_PROJECT_ID` | `deepfake-detector-494710` | GCP project for Vertex AI and Firebase |
| `GEMINI_LOCATION` | `global` | Vertex AI region |
| `GEMINI_MAX_FRAMES` | `240` | Landmark frames sampled into the prompt |
| `GEMINI_MAX_TEXT_CHARS` | `20000` | Hard cap on prompt landmark text |
| `PORT` | `8000` | Server port. Cloud Run sets this automatically |
| `MAX_UPLOAD_BYTES` | `52428800` | Upload size limit (50 MB) |
| `ALLOWED_ORIGINS` | unset | Extra CORS origins, comma separated |
| `FACE_LANDMARKER_MODEL` | `face_landmarker.task` | Model file path |
| `GOOGLE_APPLICATION_CREDENTIALS` | unset | Service-account key. Omit to use ADC |
| `FIREBASE_STORAGE_BUCKET` | derived from project ID | Storage bucket override |

Landmark files are sampled before being embedded in a prompt: a 60-second clip is roughly 1800
frames and ~800 KB of JSON, which is far more than the model needs. Sampling keeps that under
20 KB without discarding the shape of the motion.

---

## Project structure

```
sensor-module/
│
├── sensor.py                 MediaPipe landmark extraction (CLI + importable)
├── analysis_utils.py         Shared metrics: velocity series, jitter ratio,
│                             saccade counting, detection rate, downsampling
├── deepfake_detection.py     Three-tier analysis engine + local physics heuristic
├── firebase_utils.py         Firebase Admin wrapper with graceful degradation
├── server.py                 FastAPI app: GET / and POST /analyze
├── main.py                   One-command CLI pipeline
│
├── Dockerfile                Cloud Run container (listens on $PORT, default 8080)
├── requirements.txt          Python dependencies
├── pyproject.toml            Project metadata (uv compatible)
├── .env.example              Every supported environment variable, documented
│
├── Frontend/                 React 18 + Vite single-page app
│   ├── src/App.jsx           Upload, analysis, results, real jitter chart
│   ├── src/components/       Typewriter effect
│   ├── src/index.css         Dark glassmorphism design system
│   ├── public/config.js      Runtime backend URL for deployed builds
│   └── dist/                 Build output (gitignored)
│
├── functions/                Firebase Cloud Functions (Node 22)
│   └── index.js              onObjectFinalized -> Vertex AI -> Firestore
│
├── public/                   Firebase Hosting root — copy Frontend/dist/* here
├── firebase.json             Hosting + Functions + Storage config
├── storage.rules             Storage security rules
└── cors.json                 Storage CORS config for browser uploads
```

---

## Troubleshooting

**`ModuleNotFoundError` on startup**
The virtualenv is not active, or dependencies are not installed. Re-run
`pip install -r requirements.txt` inside the activated environment.

**Backend starts, frontend says "failed to fetch"**
Confirm the backend is up: `curl http://localhost:8000/`. If you changed `PORT`, update the
localhost URL in `Frontend/src/App.jsx` to match.

**CORS error in the browser console**
`localhost:5173`, `:3000`, `:5000`, `:8000` and `:8080` are allowed out of the box. For any
other origin, set `ALLOWED_ORIGINS=http://localhost:4173` (comma separated for several).

**`422 No face was detected`**
The clip needs a clearly visible, front-facing subject. Aim for 640×480 or higher and 24+ fps.
Heavy motion blur, extreme angles, strong backlighting and very small faces all break
detection. Check the reported detection rate — a low value is itself a signal.

**`Firebase unavailable (...): running in local mode`**
Expected without GCP credentials. Not an error; the app works fully in this state.

**Gemini quota or auth errors**
Also expected, and handled: the request falls back to the local physics engine and the response
marks `usedFallback: true`. Check the engine name shown with the result to see what ran.

**Deployed site does not show recent changes**
You rebuilt into `Frontend/dist/` but did not copy into `public/`. See
[step 5 of the cloud setup](#5-build-and-deploy-the-frontend).

**Slow analysis**
Landmark extraction is CPU-bound at roughly 15–50 fps depending on hardware, so a 30-second
clip takes about 20–60 seconds. Shorter clips are proportionally faster.

---

## Limitations

Worth stating plainly, because a detector that oversells itself is worse than none:

- **The physics engine is a heuristic, not a trained classifier.** Its thresholds are
  hand-tuned against the behaviours described above, not learned from a labelled corpus. It has
  not been benchmarked against a standard deepfake dataset.
- **A very still subject looks smooth.** Someone holding rigidly still produces low velocity
  variance, which is the same signature the inertia test flags. Natural head motion gives far
  more reliable results.
- **Smoothing interacts with the saccade test.** The saccade threshold is an absolute
  displacement, so a high `--smoothing-alpha` (0.9+) can damp real saccades below it. The
  default 0.8 is fine; the jitter ratio itself is scale-invariant and unaffected.
- **Single face only.** `num_faces=1`; the first detected face wins.
- **Re-encoding matters.** Aggressive compression and frame interpolation applied by messaging
  apps and social platforms alter exactly the motion signals being measured.
- **This is a research prototype**, not a legal or journalistic authentication tool. Treat
  every verdict as one signal among many.

---

## License

MIT — see [LICENSE](LICENSE).

Built for the Google AI Challenge. Powered by MediaPipe and Google Gemini.
