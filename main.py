"""End-to-end command-line workflow: video -> landmarks -> forensic verdict.

Usage:
    python main.py path/to/video.mp4
    python main.py path/to/video.mp4 --landmarks existing_landmarks.json
    python main.py path/to/video.mp4 --no-upload

Landmark extraction always runs locally. Cloud backup to Firebase Storage and
Firestore happens only when credentials are available and --no-upload was not
passed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from analysis_utils import compute_metrics
from deepfake_detection import analyze
from firebase_utils import (
    initialize_firebase,
    is_firebase_available,
    save_analysis_to_firestore,
    upload_file_to_storage,
)
from sensor import process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full deepfake detection pipeline on a video.")
    parser.add_argument("video", help="Path to the video file to analyze.")
    parser.add_argument(
        "--landmarks",
        default=None,
        help="Reuse an existing landmarks JSON instead of re-extracting. "
        "Default: <video_stem>_landmarks.json (created if absent).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("FACE_LANDMARKER_MODEL", "face_landmarker.task"),
        help="Path to the MediaPipe Face Landmarker model. Downloaded automatically if missing.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip Firebase upload even when credentials are available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1

    landmarks_path = (
        Path(args.landmarks).expanduser()
        if args.landmarks
        else video_path.with_name(f"{video_path.stem}_landmarks.json")
    )

    if landmarks_path.exists() and args.landmarks:
        print(f"Reusing existing landmarks: {landmarks_path}")
        records = json.loads(landmarks_path.read_text(encoding="utf-8"))
    else:
        print(f"Extracting landmarks from {video_path.name}...")
        records = process_video(
            input_video=video_path,
            output_json=landmarks_path,
            model_path=Path(args.model).expanduser(),
            include_velocity=True,
            verbose=True,
        )
        print(f"Landmarks written to {landmarks_path}")

    metrics = compute_metrics(records)
    if metrics["framesWithFace"] == 0:
        print("Error: no face detected in any frame. Try a clearer, front-facing clip.", file=sys.stderr)
        return 1

    storage_path = None
    if not args.no_upload:
        initialize_firebase()
        if is_firebase_available():
            try:
                storage_path = upload_file_to_storage(str(video_path), is_video=True)
                print(f"Video backed up to Cloud Storage at: {storage_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: cloud backup failed ({exc}). Continuing locally.")
        else:
            print("Firebase unavailable. Running analysis locally.")

    print("Starting deepfake analysis...")
    result = analyze(str(video_path), str(landmarks_path))

    if storage_path and is_firebase_available():
        try:
            save_analysis_to_firestore(
                storage_path, result["raw"], total_frames=metrics["totalFrames"]
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not write results to Firestore ({exc}).")

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Engine          : {result['engine']} ({result['model']})")
    if result.get("fallback"):
        print(f"Fallback used   : yes - {result.get('fallback_reason', 'cloud tier unavailable')}")
    print(f"Frames          : {metrics['totalFrames']} "
          f"({metrics['framesWithFace']} with a detected face, "
          f"{metrics['detectionRate'] * 100:.1f}%)")
    print(f"Jitter ratio    : {metrics['jitterRatio']:.3f} (sigma/mu of chin velocity)")
    print(f"Saccadic events : {metrics['saccadePeaks']} across {metrics['eyeSamples']} samples")
    print("-" * 60)

    try:
        print(json.dumps(json.loads(result["raw"]), indent=2))
    except Exception:  # noqa: BLE001
        print(result["raw"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
