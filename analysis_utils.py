"""Shared helpers for turning raw landmark records into forensic metrics.

Both the FastAPI server and the offline physics engine read from this module so
that the numbers shown in the UI are the *same* numbers the verdict is based on.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# Landmark displacement (in MediaPipe normalized units) above which a per-frame
# eye movement is treated as a discrete saccade rather than smooth drift.
SACCADE_THRESHOLD = 0.005

# Number of points sent to the frontend chart. The raw series can be thousands
# of frames long; this keeps the payload small without hiding the signal.
DEFAULT_SERIES_POINTS = 160


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def detected_frames(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only the frames where MediaPipe actually located a face."""
    return [r for r in records if r.get("face_detected") and r.get("chin") is not None]


def chin_velocity_series(records: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Per-frame chin speed in normalized units per second.

    Uses the ``chin_velocity`` field when ``sensor.process_video`` was run with
    ``include_velocity=True``; otherwise it recomputes the same quantity from the
    raw coordinates so the chart works with either flavour of landmark file.
    """
    series: List[Dict[str, float]] = []
    previous: Optional[Dict[str, Any]] = None

    for record in records:
        if not (record.get("face_detected") and record.get("chin") is not None):
            previous = None
            continue

        time_s = float(record.get("time", 0.0))
        velocity = record.get("chin_velocity")

        if velocity is None:
            if previous is None:
                velocity = 0.0
            else:
                dt = time_s - float(previous.get("time", 0.0))
                velocity = _euclidean(record["chin"], previous["chin"]) / dt if dt > 0 else 0.0

        series.append({"t": round(time_s, 4), "v": float(velocity)})
        previous = record

    return series


def downsample_series(
    series: Sequence[Dict[str, float]],
    max_points: int = DEFAULT_SERIES_POINTS,
) -> List[Dict[str, float]]:
    """Bucket-average a velocity series down to at most ``max_points`` points.

    Averaging (rather than stride sampling) is deliberate: stride sampling
    aliases high-frequency tremor into whatever pattern the stride happens to
    hit, which is exactly the signal this chart exists to show.
    """
    points = list(series)
    if max_points <= 0 or len(points) <= max_points:
        return [{"t": round(p["t"], 4), "v": round(p["v"], 6)} for p in points]

    bucket_size = len(points) / max_points
    out: List[Dict[str, float]] = []

    for i in range(max_points):
        start = int(i * bucket_size)
        end = max(start + 1, int((i + 1) * bucket_size))
        chunk = points[start:end]
        if not chunk:
            continue
        out.append(
            {
                "t": round(sum(p["t"] for p in chunk) / len(chunk), 4),
                "v": round(sum(p["v"] for p in chunk) / len(chunk), 6),
            }
        )

    return out


def compute_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the kinetic statistics that drive both the verdict and the chart."""
    total_frames = len(records)
    detected = detected_frames(records)
    detection_rate = len(detected) / total_frames if total_frames else 0.0

    series = chin_velocity_series(records)
    velocities = [p["v"] for p in series]

    mean_velocity = sum(velocities) / len(velocities) if velocities else 0.0
    if velocities:
        variance = sum((v - mean_velocity) ** 2 for v in velocities) / len(velocities)
        std_velocity = math.sqrt(variance)
    else:
        std_velocity = 0.0

    jitter_ratio = std_velocity / (mean_velocity + 1e-9) if velocities else 0.0

    # Discrete eye jumps: real gaze moves in saccades, synthetic gaze slides.
    saccade_peaks = 0
    eye_samples = 0
    previous = None
    for record in detected:
        left, right = record.get("left_eye"), record.get("right_eye")
        if left and right and previous:
            prev_left, prev_right = previous.get("left_eye"), previous.get("right_eye")
            if prev_left and prev_right:
                magnitude = math.sqrt(
                    (left[0] - prev_left[0]) ** 2
                    + (left[1] - prev_left[1]) ** 2
                    + (right[0] - prev_right[0]) ** 2
                    + (right[1] - prev_right[1]) ** 2
                )
                eye_samples += 1
                if magnitude > SACCADE_THRESHOLD:
                    saccade_peaks += 1
        previous = record

    return {
        "totalFrames": total_frames,
        "framesWithFace": len(detected),
        "detectionRate": round(detection_rate, 4),
        "meanVelocity": round(mean_velocity, 6),
        "stdVelocity": round(std_velocity, 6),
        "jitterRatio": round(jitter_ratio, 4),
        "saccadePeaks": saccade_peaks,
        "eyeSamples": eye_samples,
        "series": series,
    }


def load_records(landmarks_path: str) -> List[Dict[str, Any]]:
    """Read a landmarks JSON file, returning an empty list on any failure."""
    import json
    import os

    if not landmarks_path or not os.path.exists(landmarks_path):
        return []
    try:
        with open(landmarks_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - never let a bad file kill a request
        print(f"Warning: could not parse landmarks JSON ({landmarks_path}): {exc}")
        return []
    return data if isinstance(data, list) else []
