FROM python:3.11-slim-bookworm

WORKDIR /app

ENV MEDIAPIPE_DISABLE_GPU=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# OpenCV / MediaPipe runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libgles2 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so the layer caches across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. Keep every imported module out of .dockerignore:
# server.py, sensor.py, analysis_utils.py, deepfake_detection.py, firebase_utils.py
COPY . .

EXPOSE 8080

# Shell form so $PORT (injected by Cloud Run) is expanded at runtime.
CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
