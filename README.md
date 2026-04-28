# Sensor Module: Face Landmark Extraction

A production-grade Python script for extracting selected facial landmarks from video files using MediaPipe Face Landmarker. Designed for deepfake detection systems and facial motion analysis.

## Features

- **Frame-by-frame landmark extraction** using MediaPipe Face Mesh (478-point model)
- **Smart landmark selection**: chin, pupil centers (averaged from iris points), eyebrows
- **Temporal continuity**: preserves timeline even when faces are not detected
- **EMA smoothing**: optional exponential moving average to reduce jitter noise
- **Velocity computation**: optional chin movement speed calculation
- **Automatic model download**: fetches the official MediaPipe model on first run
- **Robust error handling**: skips corrupted frames without breaking the pipeline

## Installation

### Prerequisites

- Python 3.8+
- pip or uv

### Setup

```bash
# Clone or navigate to the sensor-module directory
cd sensor-module

# Install dependencies
pip install -r requirements.txt
# OR with uv:
uv pip install -r requirements.txt
```

## Usage

### Basic Usage

Process a video and extract landmarks to JSON:

```bash
python sensor.py input.mp4
```

Output file: `input_landmarks.json`

### With Custom Output Path

```bash
python sensor.py input.mp4 --output output.json
```

### Verbose Mode (Debug Info)

```bash
python sensor.py input.mp4 --verbose
```

Prints:

- Video FPS and total frame count
- Processing progress (every 50 frames)
- Final frame count with face detection

### With Smoothing

Reduce jitter in landmarks using EMA (exponential moving average):

```bash
python sensor.py input.mp4 --smoothing-alpha 0.8
```

- `alpha=0.0`: no smoothing (raw landmarks)
- `alpha=0.8`: default, strong smoothing
- `alpha=0.95`: lighter smoothing (more responsive)

### With Velocity Computation

Add chin movement speed to output:

```bash
python sensor.py input.mp4 --include-velocity
```

### Using with uv

```bash
uv run python sensor.py input.mp4 --verbose
```

## Output Format

JSON array with one record per frame:

```json
[
  {
    "frame": 0,
    "time": 0.0,
    "face_detected": true,
    "chin": [0.5, 0.6, -0.02],
    "left_eye": [0.35, 0.4, -0.01],
    "right_eye": [0.65, 0.4, -0.01],
    "left_eyebrow": [0.3, 0.25, 0.0],
    "right_eyebrow": [0.7, 0.25, 0.0]
  },
  {
    "frame": 1,
    "time": 0.033,
    "face_detected": true,
    "chin": [0.501, 0.602, -0.019],
    "left_eye": [0.351, 0.401, -0.009],
    "right_eye": [0.651, 0.401, -0.009],
    "left_eyebrow": [0.301, 0.251, 0.001],
    "right_eyebrow": [0.701, 0.251, 0.001]
  },
  {
    "frame": 2,
    "time": 0.066,
    "face_detected": false,
    "chin": null,
    "left_eye": null,
    "right_eye": null,
    "left_eyebrow": null,
    "right_eyebrow": null
  }
]
```

### Fields

| Field           | Type              | Description                                     |
| --------------- | ----------------- | ----------------------------------------------- |
| `frame`         | int               | 0-indexed frame number                          |
| `time`          | float             | Timestamp in seconds (frame_id / fps)           |
| `face_detected` | bool              | Whether a face was detected in this frame       |
| `chin`          | [x, y, z] \| null | Normalized chin coordinates or null if no face  |
| `left_eye`      | [x, y, z] \| null | Left iris center (averaged from 5 iris points)  |
| `right_eye`     | [x, y, z] \| null | Right iris center (averaged from 5 iris points) |
| `left_eyebrow`  | [x, y, z] \| null | Left eyebrow corner                             |
| `right_eyebrow` | [x, y, z] \| null | Right eyebrow corner                            |
| `chin_velocity` | float \| null     | (Optional) Chin movement speed in units/second  |

### Coordinate System

All coordinates are **normalized** (0.0–1.0):

- `x`: left (0.0) to right (1.0)
- `y`: top (0.0) to bottom (1.0)
- `z`: depth estimate from the model (-1.0 to +1.0), where negative = farther from camera

## Landmark Indices

The script extracts the following MediaPipe Face Mesh indices:

| Landmark      | Index(es)                          |
| ------------- | ---------------------------------- |
| Chin          | 152                                |
| Left Iris     | 468, 469, 470, 471, 472 (averaged) |
| Right Iris    | 473, 474, 475, 476, 477 (averaged) |
| Left Eyebrow  | 70                                 |
| Right Eyebrow | 300                                |

## Configuration Options

```
usage: sensor.py [-h] [--model MODEL] [--output OUTPUT]
                  [--include-velocity] [--smoothing-alpha ALPHA] [--verbose]
                  [input_video]

positional arguments:
  input_video              Path to input video (default: input.mp4)

optional arguments:
  -h, --help               Show help message
  --model MODEL            Path to Face Landmarker model (.task file)
                          Auto-downloads if missing
  --output OUTPUT          Path to output JSON file
                          (default: <input_stem>_landmarks.json)
  --include-velocity       Add chin_velocity to each record
  --smoothing-alpha ALPHA  EMA smoothing factor [0.0-1.0]
                          (default: 0.8)
  --verbose                Print debug information
```

## Performance

- **Speed**: ~30-50 fps on modern CPU (M1/M2 MacBook, Intel i7+)
- **Memory**: ~200-300 MB for typical video processing
- **Model size**: ~30 MB (downloaded once)

## Troubleshooting

### Model Download Fails

If the automatic model download fails due to network issues:

1. Download manually:

   ```bash
   wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```

2. Place in the same directory as `sensor.py`

3. Run: `python sensor.py input.mp4`

### No Faces Detected

- Ensure faces are clearly visible in the video
- Check video resolution (recommended: 640x480 or higher)
- Verify frame rate is reasonable (24+ fps)

### Out of Memory

- Process shorter video clips separately
- Reduce video resolution before processing

## Advanced Usage

### Processing Multiple Videos

```bash
for video in *.mp4; do
  python sensor.py "$video" --verbose
done
```

### Post-Processing in Python

```python
import json

with open("input_landmarks.json") as f:
    records = json.load(f)

# Filter only detected frames
detected = [r for r in records if r["face_detected"]]

# Calculate average chin position
chins = [r["chin"] for r in detected]
avg_chin = [sum(c[i] for c in chins) / len(chins) for i in range(3)]

print(f"Detected {len(detected)}/{len(records)} frames")
print(f"Average chin: {avg_chin}")
```

## Architecture

The script follows a modular, production-grade design:

1. **Argument parsing** (`parse_args()`) - CLI interface
2. **Model management** (`ensure_model_file()`) - Auto-download if needed
3. **Initialization** (`create_face_landmarker()`) - MediaPipe setup
4. **Landmark extraction** (`extract_selected_landmarks()`) - Point selection
5. **Processing pipeline** (`process_video()`) - Main loop
6. **Data transformations** - Smoothing, velocity computation
7. **JSON export** - Structured output

## Engineering Notes

### Why Temporal Continuity?

Skipping frames without faces breaks time-series analysis. By recording `face_detected=false`, downstream models can:

- Interpolate missing landmarks
- Detect blink events (rapid face loss)
- Maintain frame-to-frame correspondence

### Why Iris Averaging?

MediaPipe outputs 5 iris points (perimeter). Averaging them:

- Reduces noise from individual point jitter
- Gives a stable pupil center
- Works better for velocity analysis

### Why EMA Smoothing?

MediaPipe's predictions have inherent jitter. EMA filter:

- Reduces high-frequency noise
- Preserves real motion
- Is computationally cheap
- Can be toggled off if raw data is needed

### Why Optional Velocity?

Not all use cases need velocity. Optional flag keeps the default output schema clean while supporting extended features.

## Deepfake Detection with Gemini

Once you've extracted landmarks, analyze them for deepfake indicators using the `deepfake_detection.py` module.

### Setup

Set the Google Cloud project ID:

```bash
export GOOGLE_CLOUD_PROJECT_ID="your-project-id"
```

### Basic Usage

```bash
python deepfake_detection.py video.mp4 landmarks.json
```

### Output

Returns JSON with:

- `authenticity_score` (0-100): Confidence the video is genuine
- `flagged_anomalies`: Timestamps and reasons for suspicion
- `forensic_explanation`: Technical summary of findings

### Analysis Focus

The detector checks for:

- **Snap-to-grid effect**: Unnatural smoothness in chin acceleration
- **Saccadic violations**: Linear eye movements instead of discrete jumps
- **Biological noise loss**: Absence of micro-tremors (3-7 Hz)
- **Temporal lag**: Desync between mouth and eye movements (>50ms)

## License

MIT License

## References

- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Face Mesh Documentation](https://google.github.io/mediapipe/solutions/face_mesh)
- [478-Point Face Landmark Map](https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png)

## Support

For issues or feature requests, refer to the inline code comments in `sensor.py` or the [MediaPipe GitHub](https://github.com/google-ai-edge/mediapipe).
