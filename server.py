"""FastAPI backend for the Kinetic Forensics deepfake detector.

Exposes two endpoints:

    GET  /          health check + which analysis engine is currently active
    POST /analyze   upload a video, get a forensic verdict back

The same code path serves local and cloud deployments. When Firebase
credentials are present the video, landmarks and verdict are also persisted to
Cloud Storage and Firestore; when they are not, everything stays on the local
machine and the verdict is returned directly in the HTTP response.
"""

import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from analysis_utils import compute_metrics, downsample_series
from deepfake_detection import analyze, describe_engine
from firebase_utils import (
    initialize_firebase,
    is_firebase_available,
    save_analysis_to_firestore,
    storage_bucket_name,
    upload_file_to_storage,
)
from sensor import process_video

APP_VERSION = "2.1.0"

# Upload guard. The frontend enforces the same limit client-side, but never
# trust the client.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))

# Extra browser origins may be added with a comma-separated ALLOWED_ORIGINS var.
DEFAULT_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
    "https://deepfake-detector-494710.web.app",
    "https://deepfake-detector-494710.firebaseapp.com",
]
EXTRA_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

MODEL_PATH = Path(os.environ.get("FACE_LANDMARKER_MODEL", "face_landmarker.task"))

app = FastAPI(title="Kinetic Forensics Backend", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS + EXTRA_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase on startup. This is deliberately non-fatal: a missing
# credential just means the server runs in local mode.
initialize_firebase(os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "deepfake-detector-494710"))


# --- Helpers ------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Parse a model response that may be wrapped in a markdown code fence.

    Returns ``{"raw_output": text}`` when nothing parseable is found, so callers
    always get a dict and can check for the absence of real fields.
    """
    if not text:
        return {"raw_output": ""}

    candidate = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass

    # Last resort: grab the outermost {...} block from a chatty response.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass

    return {"raw_output": text}


def coerce_score(value) -> Optional[float]:
    """Turn a model-supplied authenticity score into a clamped 0-100 float.

    Returns ``None`` when the value is missing or unusable. Crucially, there is
    no default of 100 here: an unparseable response must surface as
    "Inconclusive", never as a confident "Real".
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        score = float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if not match:
                return None
            score = float(match.group())
        else:
            return None
    return max(0.0, min(100.0, score))


def build_result_payload(structured: dict, metrics: dict, engine_info: dict) -> dict:
    """Assemble the response body the frontend renders."""
    score = coerce_score(structured.get("authenticity_score"))

    anomalies = structured.get("flagged_anomalies") or []
    if isinstance(anomalies, str):
        anomalies = [anomalies]
    anomalies = [str(a) for a in anomalies][:12]

    report = (
        structured.get("forensic_explanation")
        or structured.get("raw_output")
        or ""
    )

    payload = {
        "engine": engine_info.get("engine"),
        "model": engine_info.get("model"),
        "usedFallback": bool(engine_info.get("fallback")),
        "framesAnalyzed": metrics["totalFrames"],
        "framesWithFace": metrics["framesWithFace"],
        "detectionRate": metrics["detectionRate"],
        "jitterRatio": metrics["jitterRatio"],
        "meanVelocity": metrics["meanVelocity"],
        "stdVelocity": metrics["stdVelocity"],
        "saccadePeaks": metrics["saccadePeaks"],
        "jitterSeries": downsample_series(metrics["series"]),
        "anomalies": anomalies,
        "report": report,
    }

    if score is None:
        # Point of failure made visible instead of silently reported as "Real".
        payload.update(
            {
                "status": "Inconclusive",
                "probability": None,
                "confidence": None,
                "report": report
                or "The analysis engine did not return a usable authenticity score. "
                "The kinetic measurements below were still computed from the video.",
                "warning": "No authenticity score was returned by the analysis engine.",
            }
        )
    else:
        prob_fake = round(100.0 - score, 1)
        payload.update(
            {
                "status": "Fake" if prob_fake >= 50.0 else "Real",
                "probability": prob_fake,
                "confidence": round(max(score, prob_fake), 1),
                "authenticityScore": round(score, 1),
            }
        )

    if engine_info.get("fallback_reason"):
        payload["fallbackReason"] = engine_info["fallback_reason"]

    return payload


# --- Endpoints ----------------------------------------------------------------


@app.get("/")
@app.get("/health")
def health():
    """Report liveness plus which analysis engine and storage backend are active."""
    engine_info = describe_engine()
    return {
        "status": "running",
        "version": APP_VERSION,
        "firebase": "connected" if is_firebase_available() else "unavailable (local mode)",
        "analysis_engine": f"{engine_info['engine']} ({engine_info['model']})",
        "engine": engine_info["engine"],
        "model": engine_info["model"],
        "model_file_present": MODEL_PATH.exists(),
        "max_upload_mb": round(MAX_UPLOAD_BYTES / (1024 * 1024), 1),
    }


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """Extract landmarks from an uploaded video and return a forensic verdict."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # A unique working directory per request: two uploads in the same second no
    # longer collide, and cleanup is a single rmtree.
    work_dir = Path(tempfile.mkdtemp(prefix=f"kf_{timestamp}_{uuid.uuid4().hex[:8]}_"))

    try:
        safe_name = Path(video.filename or "upload.mp4").name
        video_path = work_dir / safe_name
        landmarks_path = work_dir / "landmarks.json"

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        size = video_path.stat().st_size
        if size == 0:
            return JSONResponse(status_code=400, content={"error": "The uploaded file is empty."})
        if size > MAX_UPLOAD_BYTES:
            limit_mb = round(MAX_UPLOAD_BYTES / (1024 * 1024), 1)
            return JSONResponse(
                status_code=413,
                content={"error": f"Video exceeds the {limit_mb} MB upload limit."},
            )

        if not MODEL_PATH.exists():
            print(f"Face Landmarker model not found at {MODEL_PATH}; it will be downloaded now.")

        print(f"Processing video {video_path.name} ({size / 1e6:.1f} MB)...")
        records = process_video(
            input_video=video_path,
            output_json=landmarks_path,
            model_path=MODEL_PATH,
            include_velocity=True,
        )

        metrics = compute_metrics(records)

        if metrics["framesWithFace"] == 0:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "No face was detected in this video. Use a clip with a clearly "
                    "visible, front-facing subject at 640x480 or higher."
                },
            )

        print("Running deepfake analysis...")
        engine_info = analyze(str(video_path), str(landmarks_path))
        structured = extract_json(engine_info["raw"])
        result_payload = build_result_payload(structured, metrics, engine_info)

        # Optional cloud persistence. Failures here never fail the request.
        video_uri = None
        if is_firebase_available():
            try:
                video_blob = upload_file_to_storage(
                    file_path=str(video_path),
                    bucket_folder="analyzed_videos",
                    is_video=True,
                    timestamp=timestamp,
                )
                upload_file_to_storage(
                    file_path=str(landmarks_path),
                    bucket_folder="analyzed_videos",
                    is_video=False,
                    timestamp=timestamp,
                )
                video_uri = f"gs://{storage_bucket_name()}/{video_blob}"
                save_analysis_to_firestore(
                    video_uri, engine_info["raw"], total_frames=metrics["totalFrames"]
                )
            except Exception as fe:  # noqa: BLE001
                print(f"Warning: failed to save to Firebase: {fe}")

        return JSONResponse(
            {
                "message": "Analysis completed successfully",
                "videoUri": video_uri,
                "result": result_payload,
            }
        )

    except Exception as exc:  # noqa: BLE001
        print(f"Error processing video: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    # Cloud Run injects PORT; locally we default to 8000 (what the frontend
    # expects when it detects localhost).
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
