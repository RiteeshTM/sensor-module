from deepfake_detection import analyze_deepfake
import json
import sys


def main():
    """
    Example usage of the deepfake detection module.
    
    Files available in workspace:
    - video_0001_landmarks.json
    - video_0003_landmarks.txt
    - white_man_speaking_landmarks.txt
    """
    
    # Example 1: Using JSON landmarks file
    video_path = "video_0001.mp4"  # Replace with actual video path
    landmarks_path = "video_0001_landmarks.json"
    
    try:
        print("Starting deepfake analysis...")
        results = analyze_deepfake(video_path, landmarks_path)
        
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
