from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from datetime import datetime
from pathlib import Path

from sensor import process_video
from firebase_utils import initialize_firebase, upload_file_to_storage, is_firebase_available, save_analysis_to_firestore

app = FastAPI(title="Sensor Module Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:5000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1",
        "https://deepfake-detector-494710.web.app",
        "https://deepfake-detector-494710.firebaseapp.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase on startup (resilient)
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "deepfake-detector-494710")
initialize_firebase(project_id)

@app.get("/")
def health():
    import os
    firebase_status = "connected" if is_firebase_available() else "unavailable (local mode)"
    gemini_mode = "Google AI Studio (gemini-2.5-flash)" if os.environ.get("GEMINI_API_KEY") else (
        "Vertex AI (gemini-3.1-pro)" if is_firebase_available() else "Local Heuristic Mock"
    )
    return {
        "status": "running",
        "firebase": firebase_status,
        "analysis_engine": gemini_mode,
        "version": "2.0.0"
    }


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    temp_video_path = None
    temp_landmarks_path = None
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

        # Perform Gemini analysis locally (or mock fallback)
        from deepfake_detection import generate
        print("Running deepfake analysis...")
        analysis_raw = generate(temp_video_path, temp_landmarks_path)
        
        # Parse the JSON response
        clean_result = analysis_raw
        if clean_result.strip().startswith("```json"):
            clean_result = clean_result.strip().strip("```json\n").strip("```").strip()
        try:
            import json
            structured_data = json.loads(clean_result)
        except Exception:
            structured_data = {"raw_output": analysis_raw}
            
        # Get frame count
        total_frames = 0
        try:
            with open(temp_landmarks_path) as f:
                landmarks = json.load(f)
                total_frames = len(landmarks)
        except Exception:
            pass

        score = float(structured_data.get("authenticity_score", 100))
        prob_fake = 100.0 - score
        status = "Fake" if prob_fake >= 50.0 else "Real"
        
        result_payload = {
            "probability": prob_fake,
            "confidence": max(score, prob_fake),
            "framesAnalyzed": total_frames,
            "status": status,
            "report": structured_data.get("forensic_explanation", "") or structured_data.get("raw_output", "")
        }
        
        video_uri = None
        # If Firebase is available, upload files and store analysis
        if is_firebase_available():
            try:
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
                
                bucket_name = "deepfake-detector-494710.firebasestorage.app"
                video_uri = f"gs://{bucket_name}/{video_blob}"
                
                # Save analysis directly to Firestore (bypassing Cloud Function)
                save_analysis_to_firestore(video_uri, analysis_raw)
            except Exception as fe:
                print(f"Warning: Failed to save to Firebase: {fe}")
        
        # Clean up temporary files
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if temp_landmarks_path and os.path.exists(temp_landmarks_path):
            os.remove(temp_landmarks_path)
            
        return JSONResponse({
            "message": "Analysis completed successfully",
            "videoUri": video_uri,
            "result": result_payload
        })
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        # Clean up temporary files if they exist
        try:
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if temp_landmarks_path and os.path.exists(temp_landmarks_path):
                os.remove(temp_landmarks_path)
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
