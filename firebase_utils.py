import firebase_admin
from firebase_admin import credentials, storage
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
