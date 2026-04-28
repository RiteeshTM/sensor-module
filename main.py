from deepfake_detection import generate
from firebase_utils import initialize_firebase, upload_video_to_storage, save_analysis_to_firestore
import json
import sys
import os


def main():
    """
    Example usage of the deepfake detection module.
    
    Files available in workspace:
    - video_0001_landmarks.json
    - video_0003_landmarks.txt
    - white_man_speaking_landmarks.txt
    """
    
    # Example 1: Using JSON landmarks file
    # We will use the download.mp4 and download_landmarks.json we downloaded
    video_path = "download.mp4"  # Replace with actual video path if needed
    landmarks_path = "download_landmarks.json"
    
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "deepfake-detector-494710")
    
    try:
        # Initialize Firebase and Upload Video to Cloud Storage Backend
        print(f"Connecting to Firebase backend for project: {project_id}")
        initialize_firebase(project_id)
        
        # Perform Video Upload to Firebase Storage
        storage_path = upload_video_to_storage(video_path, bucket_folder="analyzed_videos")
        
        # After upload finishes, call Gemini to analyze the deepfake locally
        print(f"Video backed up to Firebase at: {storage_path}")
        print("Starting deepfake analysis...")
        results = generate(video_path, landmarks_path)
        
        # Save results to Firebase Firestore Database automatically
        doc_id = save_analysis_to_firestore(storage_path, results)
        
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        print(json.dumps(results, indent=2))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print(f"  1. Video file: {video_path}")
        print(f"  2. Landmarks file: {landmarks_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
