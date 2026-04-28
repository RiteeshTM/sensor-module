from google import genai
from google.genai import types
import base64
import os
import sys


def load_video_file(video_path: str) -> bytes:
  """
  Dynamically loads video file from disk.
  
  Args:
      video_path: Path to video file
  
  Returns:
      Binary video data
  """
  if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")
  
  with open(video_path, 'rb') as f:
    return f.read()


def load_landmarks_file(landmarks_path: str) -> str:
  """
  Dynamically loads landmarks data from JSON or text file.
  
  Args:
      landmarks_path: Path to landmarks file (JSON or TXT format)
  
  Returns:
      Landmarks data as text string
  """
  if not os.path.exists(landmarks_path):
    raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")
  
  with open(landmarks_path, 'r') as f:
    return f.read()


def generate(video_path=None, landmarks_path=None):
  
  if not os.environ.get("GOOGLE_CLOUD_PROJECT_ID"):
    raise ValueError("GOOGLE_CLOUD_PROJECT_ID environment variable must be set.")
  
  client = genai.Client(
      vertexai=True,
      project=os.environ.get("GOOGLE_CLOUD_PROJECT_ID"),
      location="global",
  )

  # Load files dynamically if provided
  video_bytes = None
  landmarks_text = ""
  
  if video_path and landmarks_path:
    print(f"Loading video from: {video_path}")
    video_bytes = load_video_file(video_path)
    
    print(f"Loading landmarks from: {landmarks_path}")
    landmarks_text = load_landmarks_file(landmarks_path)
    print("Files loaded successfully. Starting analysis...\n")

  text1 = types.Part.from_text(text=f"""I am uploading a video file and a JSON file containing the 3D coordinates (x,y,z) of facial landmarks extracted via MediaPipe.Task:
Cross-reference the visual frames of the video with the provided coordinate data.
Analyze the Acceleration (a=Δv/Δt) of the chin (Landmarks 152, 175). Is there a "snap-to-grid" effect or unnatural smoothness?
Check the Gaze Vector consistency. Does the eye movement follow the "Main Sequence" velocity curve?
Identify any "Micro-Expression Spikes"—sudden frame-level changes in landmark positions that don't match the surrounding frames (common in diffusion-based flickering).

Output Format: Return your findings in this EXACT JSON format: 
{{ 
  "authenticity_score": [0-100], 
  "flagged_anomalies": ["list specific timestamps and physical reasons"], 
  "forensic_explanation": "A concise, technical summary of why this is human or AI." 
}}

LANDMARK DATA:
{landmarks_text}
""")
  
  si_text1 = """System Persona:
You are an expert Forensic Video Analyst specializing in Behavioral Biometrics and Biological Physics. Your goal is to detect deepfakes by identifying \"Kinetic Dissonance\"—where the visual movement in a video contradicts the laws of human physiology.

Analysis Framework:

    The Law of Inertia: Human head and jaw movements have mass. Look for \"weightless\" transitions in the provided data.

    Saccadic Eye Movement: Human eyes move in discrete, high-velocity jumps. Flag \"linear sliding\" eye movements as AI-generated.

    Biological Noise: Real humans have micro-tremors (3-7 Hz). If the motion data shows \"perfect\" curves or zero jitter, it is a synthetic interpolation.

    Temporal Lag: Look for delays between the eyes and mouth that exceed 50ms, as AI often desyncs micro-expressions."""

  model = "gemini-3.1-pro-preview"
  
  # Build parts dynamically
  parts = [text1]
  if video_bytes:
    parts.append(types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"))
  
  contents = [
    types.Content(
      role="user",
      parts=parts
    )
  ]
  tools = [
    types.Tool(google_search=types.GoogleSearch()),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 0.1,
    top_p = 0.95,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    tools = tools,
    system_instruction=[types.Part.from_text(text=si_text1)],
    thinking_config=types.ThinkingConfig(
      thinking_level="HIGH",
    ),
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
        continue
    print(chunk.text, end="")


if __name__ == "__main__":
  if len(sys.argv) == 3:
    # Run with dynamic files: python sample.py <video_path> <landmarks_path>
    video_path = sys.argv[1]
    landmarks_path = sys.argv[2]
    try:
      generate(video_path, landmarks_path)
    except FileNotFoundError as e:
      print(f"Error: {e}", file=sys.stderr)
      sys.exit(1)
  else:
    # Run without files (original behavior)
    generate()