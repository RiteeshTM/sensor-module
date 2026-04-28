"""Sensor module for extracting selected face landmarks from a video.

This script uses OpenCV for video decoding and MediaPipe Face Landmarker for
frame-by-frame facial landmark inference.

Default JSON output schema:
[
  {
    "frame": 0,
    "time": 0.0,
    "chin": [x, y, z],
    "left_eye": [x, y, z],
    "right_eye": [x, y, z],
    "left_eyebrow": [x, y, z],
    "right_eyebrow": [x, y, z]
  }
]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


# Selected MediaPipe Face Mesh landmark indices.
# These indices are stable for the Face Landmarker model bundle.
CHIN_INDEX = 152
LEFT_EYE_IRIS_INDICES = (468, 469, 470, 471, 472)
RIGHT_EYE_IRIS_INDICES = (473, 474, 475, 476, 477)
LEFT_EYEBROW_INDEX = 70
RIGHT_EYEBROW_INDEX = 300
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


logger = logging.getLogger("sensor")


LANDMARK_FIELDS = ("chin", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract selected face landmarks from a video using MediaPipe Face Landmarker."
    )
    parser.add_argument(
        "input_video",
        nargs="?",
        default="input.mp4",
        help="Path to the input video file. Default: input.mp4",
    )
    parser.add_argument(
        "--model",
        default="face_landmarker.task",
        help=(
            "Path to the MediaPipe Face Landmarker model file (.task). "
            "If the file is missing, it is downloaded automatically from the official MediaPipe model URL."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output JSON file. Default: <input_stem>_landmarks.json",
    )
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="Add chin_velocity to each output record as an optional extra field.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print basic debug information while processing the video.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.8,
        help="EMA smoothing factor for selected landmarks. Use 0 to disable smoothing. Default: 0.8",
    )
    return parser.parse_args()


def ensure_model_file(model_path: Path) -> Path:
    """Ensure the Face Landmarker model exists locally, downloading it if needed."""

    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Face Landmarker model to %s", model_path)
    urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, model_path)
    return model_path


def create_face_landmarker(model_path: Path) -> vision.FaceLandmarker:
    """Create a configured Face Landmarker instance for video processing."""

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def landmark_to_list(landmark: mp.tasks.components.containers.NormalizedLandmark) -> List[float]:
    """Convert a MediaPipe landmark to a plain Python list."""

    return [float(landmark.x), float(landmark.y), float(landmark.z)]


def mean_landmark(landmarks: Sequence[mp.tasks.components.containers.NormalizedLandmark], indices: Sequence[int]) -> List[float]:
    """Average a set of landmarks into one stable point."""

    points = np.array([[landmarks[index].x, landmarks[index].y, landmarks[index].z] for index in indices], dtype=np.float32)
    center = points.mean(axis=0)
    return [float(center[0]), float(center[1]), float(center[2])]


def extract_selected_landmarks(
    landmarks: Sequence[mp.tasks.components.containers.NormalizedLandmark],
) -> dict:
    """Extract only the requested facial landmarks."""

    return {
        "chin": landmark_to_list(landmarks[CHIN_INDEX]),
        "left_eye": mean_landmark(landmarks, LEFT_EYE_IRIS_INDICES),
        "right_eye": mean_landmark(landmarks, RIGHT_EYE_IRIS_INDICES),
        "left_eyebrow": landmark_to_list(landmarks[LEFT_EYEBROW_INDEX]),
        "right_eyebrow": landmark_to_list(landmarks[RIGHT_EYEBROW_INDEX]),
    }


def compute_chin_velocity(current_chin: Sequence[float], previous_chin: Sequence[float], delta_time: float) -> float:
    """Compute chin speed in normalized coordinate units per second."""

    if delta_time <= 0:
        return 0.0

    current = np.array(current_chin, dtype=np.float32)
    previous = np.array(previous_chin, dtype=np.float32)
    distance = float(np.linalg.norm(current - previous))
    return distance / delta_time


def smooth_point(current: Sequence[float], previous: Sequence[float], alpha: float) -> List[float]:
    """Apply exponential moving average smoothing to a 3D point."""

    current_array = np.array(current, dtype=np.float32)
    previous_array = np.array(previous, dtype=np.float32)
    smoothed = alpha * previous_array + (1.0 - alpha) * current_array
    return [float(smoothed[0]), float(smoothed[1]), float(smoothed[2])]


def smooth_selected_landmarks(
    current: Dict[str, List[float]],
    previous: Dict[str, List[float]],
    alpha: float,
) -> Dict[str, List[float]]:
    """Smooth all selected landmarks using the previous frame as a reference."""

    if alpha <= 0.0:
        return current

    return {
        field: smooth_point(current[field], previous[field], alpha)
        for field in LANDMARK_FIELDS
    }


def process_video(
    input_video: Path,
    output_json: Path,
    model_path: Path,
    include_velocity: bool = False,
    verbose: bool = False,
    smoothing_alpha: float = 0.8,
) -> List[Dict[str, Any]]:
    """Process a video file and return the extracted landmark records."""

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    model_path = ensure_model_file(model_path)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    if smoothing_alpha < 0.0 or smoothing_alpha > 1.0:
        raise ValueError("smoothing_alpha must be between 0.0 and 1.0")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_duration_ms = 1000.0 / fps
    records: List[Dict[str, Any]] = []
    previous_chin: Optional[Sequence[float]] = None
    previous_time: Optional[float] = None
    previous_smoothed_landmarks: Optional[Dict[str, List[float]]] = None

    if verbose:
        logger.info("Video opened: %s", input_video)
        logger.info("FPS: %.3f", fps)
        logger.info("Total frames reported by OpenCV: %d", total_frames)

    with create_face_landmarker(model_path) as landmarker:
        frame_id = 0
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            if frame_bgr is None:
                if verbose:
                    logger.debug("Skipping corrupted frame %d", frame_id)
                records.append(
                    {
                        "frame": frame_id,
                        "time": frame_id / fps,
                        "face_detected": False,
                        "chin": None,
                        "left_eye": None,
                        "right_eye": None,
                        "left_eyebrow": None,
                        "right_eyebrow": None,
                        **({"chin_velocity": None} if include_velocity else {}),
                    }
                )
                frame_id += 1
                continue

            time_seconds = frame_id / fps
            timestamp_ms = int(round(frame_id * frame_duration_ms))

            # MediaPipe expects RGB input.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.face_landmarks:
                if verbose:
                    logger.debug("No face detected on frame %d", frame_id)
                records.append(
                    {
                        "frame": frame_id,
                        "time": time_seconds,
                        "face_detected": False,
                        "chin": None,
                        "left_eye": None,
                        "right_eye": None,
                        "left_eyebrow": None,
                        "right_eyebrow": None,
                        **({"chin_velocity": None} if include_velocity else {}),
                    }
                )
                previous_smoothed_landmarks = None
                previous_chin = None
                previous_time = None
                frame_id += 1
                continue

            # Use the first detected face only.
            landmarks = result.face_landmarks[0]
            selected = extract_selected_landmarks(landmarks)
            if previous_smoothed_landmarks is not None:
                selected = smooth_selected_landmarks(selected, previous_smoothed_landmarks, smoothing_alpha)
            previous_smoothed_landmarks = selected

            record: Dict[str, Any] = {
                "frame": frame_id,
                "time": time_seconds,
                "face_detected": True,
                **selected,
            }

            if include_velocity:
                if previous_chin is None or previous_time is None:
                    record["chin_velocity"] = 0.0
                else:
                    record["chin_velocity"] = compute_chin_velocity(
                        selected["chin"],
                        previous_chin,
                        time_seconds - previous_time,
                    )
                previous_chin = selected["chin"]
                previous_time = time_seconds

            records.append(record)

            if verbose and frame_id % 50 == 0:
                logger.info("Processed frame %d", frame_id)

            frame_id += 1

    capture.release()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    if verbose:
        logger.info("Wrote %d face-detected frames to %s", len(records), output_json)

    return records


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    input_video = Path(args.input_video).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_json = Path(args.output).expanduser().resolve() if args.output else input_video.with_name(f"{input_video.stem}_landmarks.json")

    process_video(
        input_video=input_video,
        output_json=output_json,
        model_path=model_path,
        include_velocity=args.include_velocity,
        verbose=args.verbose,
        smoothing_alpha=args.smoothing_alpha,
    )


if __name__ == "__main__":
    main()