from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from datetime import datetime
from pathlib import Path

from sensor import process_video
from firebase_utils import initialize_firebase, upload_file_to_storage

app = FastAPI(title="Sensor Module Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:5000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "https://deepfake-detector-494710.web.app",
        "https://deepfake-detector-494710.firebaseapp.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase on startup
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "deepfake-detector-494710")
initialize_firebase(project_id)

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    try:
        # Create a timestamp to use for all generated files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save uploaded video temporarily
        temp_video_path = f"temp_{timestamp}_{video.filename}"
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        temp_landmarks_path = f"temp_{timestamp}_landmarks.json"
        
        # Process the video to extract landmarks
        print(f"Processing video {temp_video_path}...")
        process_video(
            input_video=Path(temp_video_path),
            output_json=Path(temp_landmarks_path),
            model_path=Path("face_landmarker.task")
        )
        
        # Upload video to Firebase Storage
        video_blob = upload_file_to_storage(
            file_path=temp_video_path,
            bucket_folder="analyzed_videos",
            is_video=True,
            timestamp=timestamp
        )
        
        # Upload landmarks JSON to Firebase Storage
        landmarks_blob = upload_file_to_storage(
            file_path=temp_landmarks_path,
            bucket_folder="analyzed_videos",
            is_video=False,
            timestamp=timestamp
        )
        
        # Clean up temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_landmarks_path):
            os.remove(temp_landmarks_path)
            
        # The frontend needs the gs:// URI to poll Firestore
        bucket_name = "deepfake-detector-494710.firebasestorage.app"
        video_uri = f"gs://{bucket_name}/{video_blob}"
        
        return JSONResponse({
            "message": "Files uploaded and analysis triggered successfully",
            "videoUri": video_uri
        })
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
