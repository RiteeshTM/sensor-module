FROM python:3.10-slim

WORKDIR /app

ENV MEDIAPIPE_DISABLE_GPU=1

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1 \
    libgles2 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
