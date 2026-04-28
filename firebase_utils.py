import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
from datetime import datetime

def initialize_firebase(project_id: str):
    """
    Initializes Firebase Admin SDK using Application Default Credentials.
    """
    if not firebase_admin._apps:
        # Since gcloud auth application-default login is run, 
        # we can use the default credentials.
        cred = credentials.ApplicationDefault()
        
        # Specific bucket for this project
        bucket_name = "deepfake-detector-494710.firebasestorage.app"
        
        firebase_admin.initialize_app(cred, {
            'projectId': project_id,
            'storageBucket': bucket_name
        })
        print(f"Firebase initialized for project: {project_id}")

def upload_video_to_storage(video_path: str, bucket_folder: str = "videos") -> str:
    """
    Uploads a video to Firebase Cloud Storage.
    
    Args:
        video_path: path to the local video file.
        bucket_folder: Folder name in the storage bucket.
        
    Returns:
        The public or signed URL of the uploaded video, or the blob path.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    filename = os.path.basename(video_path)
    timestamp = datetime.now().strftime("%Y%md_%H%M%S")
    destination_blob_name = f"{bucket_folder}/{timestamp}_{filename}"
    
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    
    print(f"Uploading {video_path} to Firebase Storage as {destination_blob_name}...")
    
    # Specify the correct content type for mp4
    blob.upload_from_filename(video_path, content_type='video/mp4')
    
    print("Upload complete.")
    
    # You can also make it publicly accessible if needed:
    # blob.make_public()
    # return blob.public_url
    
    return destination_blob_name

def save_analysis_to_firestore(video_storage_path: str, analysis_result_text: str) -> str:
    """
    Saves the deepfake analysis result to a Firestore database collection.
    
    Args:
        video_storage_path: The cloud storage path of the analyzed video.
        analysis_result_text: The markdown and JSON result text returned by Gemini.
        
    Returns:
        The Document ID of the newly created Firestore record.
    """
    print(f"Saving analysis results to Firestore...")
    
    # Initialize Firestore client
    db = firestore.client()
    
    # Clean and parse the JSON from Gemini's markdown output if possible
    # Gemini often returns ```json\n{ ... }\n```
    clean_result = analysis_result_text
    if clean_result.startswith("```json"):
        clean_result = clean_result.strip("```json\n").strip("```").strip()
        
    try:
        import json
        structured_data = json.loads(clean_result)
    except Exception:
        # Fallback if the model didn't return perfect JSON
        structured_data = {"raw_output": analysis_result_text}
        
    # Append metadata
    document_data = {
        "video_reference": video_storage_path,
        "analysis": structured_data,
        "analyzed_at": firestore.SERVER_TIMESTAMP
    }
    
    # Add to an 'analyses' collection
    doc_ref = db.collection("analyses").document()
    doc_ref.set(document_data)
    
    print(f"Successfully stored in Firestore with Document ID: {doc_ref.id}")
    return doc_ref.id

