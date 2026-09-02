"""Firebase Admin SDK wrapper with graceful degradation.

Every function here is safe to call whether or not Firebase credentials exist.
:func:`initialize_firebase` never raises; callers gate cloud work behind
:func:`is_firebase_available`, so the same code path runs locally and in GCP.
"""

import json
import os
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, storage

_firebase_available = False

DEFAULT_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "deepfake-detector-494710")


def storage_bucket_name() -> str:
    """Cloud Storage bucket for this project, overridable via env."""
    explicit = os.environ.get("FIREBASE_STORAGE_BUCKET")
    if explicit:
        return explicit
    return f"{DEFAULT_PROJECT_ID}.firebasestorage.app"


def initialize_firebase(project_id: str = DEFAULT_PROJECT_ID) -> bool:
    """Initialize the Admin SDK using Application Default Credentials.

    Returns True when Firebase is usable. A failure is logged and swallowed:
    the application then runs in local/standalone mode.
    """
    global _firebase_available

    if firebase_admin._apps:
        _firebase_available = True
        return True

    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(
            cred,
            {"projectId": project_id, "storageBucket": storage_bucket_name()},
        )
        _firebase_available = True
        print(f"Firebase initialized for project: {project_id}")
    except Exception as exc:  # noqa: BLE001 - absence of credentials is expected
        _firebase_available = False
        print(f"Firebase unavailable ({exc.__class__.__name__}): running in local mode.")

    return _firebase_available


def is_firebase_available() -> bool:
    """True when the Admin SDK initialized successfully."""
    return _firebase_available and bool(firebase_admin._apps)


def upload_file_to_storage(
    file_path: str,
    bucket_folder: str = "analyzed_videos",
    is_video: bool = True,
    timestamp: str = None,
) -> str:
    """Upload a video or landmarks file to Cloud Storage.

    The naming scheme matters: the Cloud Function triggers on
    ``<prefix>_landmarks.json`` and derives the sibling video path by replacing
    that suffix with ``.mp4``, so the two names must stay in lock-step.

    Returns the destination blob name.
    """
    if not is_firebase_available():
        raise RuntimeError("Firebase is not initialized or unavailable.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    if is_video:
        blob_name = f"{bucket_folder}/backend_{timestamp}.mp4"
        content_type = "video/mp4"
    else:
        blob_name = f"{bucket_folder}/backend_{timestamp}_landmarks.json"
        content_type = "application/json"

    blob = storage.bucket().blob(blob_name)
    print(f"Uploading {os.path.basename(file_path)} to Cloud Storage as {blob_name}...")
    blob.upload_from_filename(file_path, content_type=content_type)
    print("Upload complete.")

    return blob_name


def save_analysis_to_firestore(
    video_storage_path: str,
    analysis_result_text: str,
    total_frames: int = 0,
) -> str:
    """Write an analysis result into the ``analyses`` collection.

    Returns the new document ID.
    """
    if not is_firebase_available():
        raise RuntimeError("Firebase is not initialized or unavailable.")

    print("Saving analysis results to Firestore...")

    clean = (analysis_result_text or "").strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        if clean.lower().startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        structured = json.loads(clean)
        if not isinstance(structured, dict):
            structured = {"raw_output": analysis_result_text}
    except Exception:  # noqa: BLE001 - model did not return clean JSON
        structured = {"raw_output": analysis_result_text}

    doc_ref = firestore.client().collection("analyses").document()
    doc_ref.set(
        {
            "video_reference": video_storage_path,
            "analysis": structured,
            "total_frames": total_frames,
            "analyzed_at": firestore.SERVER_TIMESTAMP,
        }
    )

    print(f"Stored in Firestore with document ID: {doc_ref.id}")
    return doc_ref.id
