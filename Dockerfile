FROM python:3.11-slim-bookworm

# System deps required by mediapipe + opencv
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files the server needs at runtime
COPY server.py sensor.py firebase_utils.py face_landmarker.task ./

# Cloud Run listens on 8080
EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
