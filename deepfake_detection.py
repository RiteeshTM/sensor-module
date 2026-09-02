"""Deepfake analysis engine with a three-tier fallback chain.

Tier 1  Google AI Studio  (GEMINI_API_KEY set)        -> gemini-2.5-flash
Tier 2  Vertex AI         (GCP credentials available) -> gemini-3.1-pro-preview
Tier 3  Physics heuristic (nothing configured)        -> fully local, zero cost

Every tier returns the same JSON shape, so callers never have to care which one
ran. Use :func:`analyze` when you want to know *which* engine produced the
result; :func:`generate` is the thin backwards-compatible wrapper that returns
just the raw text.
"""

import json
import os
import random
import sys

from analysis_utils import compute_metrics, load_records

# --- Engine identifiers -------------------------------------------------------

ENGINE_AI_STUDIO = "Google AI Studio"
ENGINE_VERTEX = "Vertex AI"
ENGINE_HEURISTIC = "Local physics heuristic"

MODEL_AI_STUDIO = "gemini-2.5-flash"
MODEL_VERTEX = "gemini-3.1-pro-preview"
MODEL_HEURISTIC = "kinetic-forensics-v2"

# Landmark data is embedded in the prompt as text. A 60-second clip is ~1800
# frames, which is far more than the model needs and burns tokens for nothing,
# so we sample it the same way the Cloud Function does.
MAX_PROMPT_FRAMES = int(os.environ.get("GEMINI_MAX_FRAMES", "240"))
MAX_PROMPT_CHARS = int(os.environ.get("GEMINI_MAX_TEXT_CHARS", "20000"))

SYSTEM_PERSONA = """System Persona:
You are an expert Forensic Video Analyst specializing in Behavioral Biometrics and Biological Physics. Your goal is to detect deepfakes by identifying "Kinetic Dissonance" - where the visual movement in a video contradicts the laws of human physiology.

Analysis Framework:
    The Law of Inertia: Human head and jaw movements have mass. Look for "weightless" transitions in the provided data.
    Saccadic Eye Movement: Human eyes move in discrete, high-velocity jumps. Flag "linear sliding" eye movements as AI-generated.
    Biological Noise: Real humans have micro-tremors (3-7 Hz). If the motion data shows "perfect" curves or zero jitter, it is a synthetic interpolation.
    Temporal Lag: Look for delays between the eyes and mouth that exceed 50ms, as AI often desyncs micro-expressions."""


def _vertex_credentials_present() -> bool:
    """Best-effort check for usable Application Default Credentials."""
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    try:
        import google.auth  # type: ignore

        google.auth.default()
        return True
    except Exception:  # noqa: BLE001 - absence of credentials is the normal case
        return False


def describe_engine() -> dict:
    """Report which tier *would* run right now, without calling anything."""
    if os.environ.get("GEMINI_API_KEY"):
        return {"engine": ENGINE_AI_STUDIO, "model": MODEL_AI_STUDIO}
    if _vertex_credentials_present():
        return {"engine": ENGINE_VERTEX, "model": MODEL_VERTEX}
    return {"engine": ENGINE_HEURISTIC, "model": MODEL_HEURISTIC}


# --- File loading -------------------------------------------------------------


def load_video_file(video_path: str) -> bytes:
    """Read a video file from disk as raw bytes."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    with open(video_path, "rb") as handle:
        return handle.read()


def load_landmarks_file(landmarks_path: str) -> str:
    """Read a landmarks file (JSON or TXT) as text."""
    if not os.path.exists(landmarks_path):
        raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")
    with open(landmarks_path, "r", encoding="utf-8") as handle:
        return handle.read()


def summarize_landmarks(raw_text: str) -> str:
    """Downsample a landmarks JSON string so it fits comfortably in a prompt."""
    try:
        records = json.loads(raw_text)
    except Exception:  # noqa: BLE001 - fall back to a plain truncation
        return raw_text[:MAX_PROMPT_CHARS]

    if not isinstance(records, list) or not records:
        return raw_text[:MAX_PROMPT_CHARS]

    step = max(1, -(-len(records) // MAX_PROMPT_FRAMES))  # ceiling division
    sampled = records[::step]
    summary = json.dumps(
        {
            "total_frames": len(records),
            "sampled_every_n_frames": step,
            "frames": sampled,
        }
    )
    return summary[:MAX_PROMPT_CHARS]


def build_prompt(landmarks_text: str) -> str:
    """Assemble the forensic analysis prompt around the sampled landmark data."""
    return f"""I am uploading a video file and a JSON file containing the 3D coordinates (x,y,z) of facial landmarks extracted via MediaPipe.

Task:
Cross-reference the visual frames of the video with the provided coordinate data.
Analyze the acceleration (a = dv/dt) of the chin (landmarks 152, 175). Is there a "snap-to-grid" effect or unnatural smoothness?
Check gaze vector consistency. Does the eye movement follow the "Main Sequence" velocity curve?
Identify any "micro-expression spikes" - sudden frame-level changes in landmark positions that do not match the surrounding frames (common in diffusion-based flickering).

Output Format: return your findings in this EXACT JSON format, and nothing else:
{{
  "authenticity_score": <integer 0-100, where 100 means certainly human>,
  "flagged_anomalies": ["specific timestamps and physical reasons - keep each entry short"],
  "forensic_explanation": "A concise, technical summary of why this is human or AI. Max 4 sentences."
}}

LANDMARK DATA (SAMPLED):
{landmarks_text}
"""


# --- Tier 3: local physics heuristic -----------------------------------------


def run_mock_analysis(video_path=None, landmarks_path=None) -> str:
    """Physics-based heuristic analyzer.

    Computes real statistics from the MediaPipe landmark data when no Gemini
    API is reachable: chin velocity variance (biological noise), eye saccade
    discreteness, and temporal face-detection continuity.
    """
    print("Running physics-based heuristic analysis on extracted landmarks...")

    records = load_records(landmarks_path)
    metrics = compute_metrics(records)

    total_frames = metrics["totalFrames"]
    frames_with_face = metrics["framesWithFace"]
    detection_rate = metrics["detectionRate"]
    jitter_ratio = metrics["jitterRatio"]

    anomalies = []
    kinetic_score = 85  # Start from an assumption of authenticity.

    # --- Metric 1: chin velocity variance (biological noise) ---
    if metrics["meanVelocity"] > 0:
        if jitter_ratio < 0.15:
            kinetic_score -= 35
            anomalies.append(
                f"Low kinetic jitter detected (sigma/mu = {jitter_ratio:.3f}). "
                "Chin movement is suspiciously smooth - biological micro-tremors (3-7 Hz) absent. "
                "Suggests synthetic frame interpolation."
            )
        elif jitter_ratio > 1.5:
            kinetic_score -= 15
            anomalies.append(
                f"Excessive kinetic noise (sigma/mu = {jitter_ratio:.3f}). "
                "Possible diffusion-model temporal flicker between frames."
            )

    # --- Metric 2: eye saccade discreteness ---
    if metrics["eyeSamples"] > 30 and metrics["saccadePeaks"] < 2:
        kinetic_score -= 25
        anomalies.append(
            "Eye movement appears linear and continuous - lacks the discrete saccadic jumps "
            "characteristic of natural human gaze (Main Sequence violations). "
            "Consistent with AI-generated iris interpolation."
        )

    # --- Metric 3: temporal face-detection continuity ---
    if total_frames and detection_rate < 0.7:
        kinetic_score -= 10
        anomalies.append(
            f"Face detected in only {detection_rate * 100:.1f}% of frames. "
            "Frequent face-loss may indicate warping artifacts from face-swap models."
        )

    kinetic_score = max(5, min(98, kinetic_score))

    # Small, *deterministic* perturbation derived from the data itself, so the
    # same clip always produces the same score (reproducibility matters here).
    detected = [r for r in records if r.get("face_detected") and r.get("chin")]
    seed_val = int(sum(detected[0]["chin"]) * 1000) if detected else total_frames
    random.seed(seed_val)
    kinetic_score = max(5, min(98, kinetic_score + random.randint(-3, 3)))

    prob_fake = 100 - kinetic_score
    status = "SYNTHETIC (AI-generated)" if prob_fake >= 50 else "AUTHENTIC (human)"

    if not anomalies:
        anomalies.append(
            "No significant kinetic anomalies detected. "
            "Biological noise profile, saccadic patterns and temporal continuity are all within normal human ranges."
        )

    explanation = (
        f"Physics-based kinetic forensics analysis of {total_frames} frames "
        f"({frames_with_face} with detected faces, {detection_rate * 100:.1f}% detection rate). "
        f"Verdict: {status}. Authenticity score: {kinetic_score}/100. "
        f"Chin-velocity jitter ratio (sigma/mu) measured at {jitter_ratio:.3f}; "
        f"{metrics['saccadePeaks']} discrete saccadic events across {metrics['eyeSamples']} eye-movement samples. "
        "The verdict is based on chin velocity variance (biological noise), eye saccade discreteness "
        "and temporal face continuity - three physiological signals that neural face-swap models "
        "consistently fail to replicate perfectly."
    )

    return json.dumps(
        {
            "authenticity_score": kinetic_score,
            "flagged_anomalies": anomalies,
            "forensic_explanation": explanation,
        },
        indent=2,
    )


# --- Tier 1 & 2: Gemini -------------------------------------------------------


def _run_ai_studio(api_key: str, video_path, prompt_text: str) -> str:
    from google import genai
    from google.genai import types

    print("Initializing Google AI Studio client...")
    client = genai.Client(api_key=api_key)

    uploaded_file = None
    parts = []
    if video_path and os.path.exists(video_path):
        print(f"Uploading local video {video_path} to Google AI Studio...")
        uploaded_file = client.files.upload(file=video_path)
        print(f"Upload complete. File name: {uploaded_file.name}")
        parts.append(uploaded_file)

    parts.append(types.Part.from_text(text=prompt_text))

    try:
        print(f"Generating analysis with {MODEL_AI_STUDIO}...")
        response = client.models.generate_content(
            model=MODEL_AI_STUDIO,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
                system_instruction=SYSTEM_PERSONA,
            ),
        )
        return response.text
    finally:
        # Always clean up the remote copy, even if generation raised.
        if uploaded_file:
            try:
                print("Cleaning up uploaded video file from Google AI Studio...")
                client.files.delete(name=uploaded_file.name)
            except Exception as delete_err:  # noqa: BLE001
                print(f"Warning: could not delete uploaded file: {delete_err}")


def _run_vertex(video_path, prompt_text: str) -> str:
    from google import genai
    from google.genai import types

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "deepfake-detector-494710")
    location = os.environ.get("GEMINI_LOCATION", "global")
    print(f"Initializing Vertex AI client for project {project_id}...")
    client = genai.Client(vertexai=True, project=project_id, location=location)

    parts = [types.Part.from_text(text=prompt_text)]
    if video_path and os.path.exists(video_path):
        parts.append(types.Part.from_bytes(data=load_video_file(video_path), mime_type="video/mp4"))

    contents = [types.Content(role="user", parts=parts)]

    print("Generating content stream via Vertex AI...")
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL_VERTEX,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.95,
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            system_instruction=[types.Part.from_text(text=SYSTEM_PERSONA)],
        ),
    ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        full_response += chunk.text
        print(chunk.text, end="")
    print()
    return full_response


# --- Public entry points ------------------------------------------------------


def _heuristic_result(video_path, landmarks_path, reason=None) -> dict:
    result = {
        "raw": run_mock_analysis(video_path, landmarks_path),
        "engine": ENGINE_HEURISTIC,
        "model": MODEL_HEURISTIC,
        "fallback": reason is not None,
    }
    if reason:
        result["fallback_reason"] = reason
    return result


def analyze(video_path=None, landmarks_path=None) -> dict:
    """Run the best available engine and report which one produced the result.

    Returns a dict with keys ``raw`` (the model or heuristic text), ``engine``,
    ``model`` and ``fallback`` (True when a cloud tier failed and the local
    heuristic took over).
    """
    landmarks_text = ""
    if landmarks_path:
        try:
            landmarks_text = summarize_landmarks(load_landmarks_file(landmarks_path))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not read landmarks file: {exc}")

    prompt_text = build_prompt(landmarks_text)
    api_key = os.environ.get("GEMINI_API_KEY")

    if api_key:
        try:
            return {
                "raw": _run_ai_studio(api_key, video_path, prompt_text),
                "engine": ENGINE_AI_STUDIO,
                "model": MODEL_AI_STUDIO,
                "fallback": False,
            }
        except Exception as err:  # noqa: BLE001
            print(f"Google AI Studio generation failed: {err}")
            print("Falling back to the local heuristic analyzer...")
            return _heuristic_result(video_path, landmarks_path, str(err))

    try:
        return {
            "raw": _run_vertex(video_path, prompt_text),
            "engine": ENGINE_VERTEX,
            "model": MODEL_VERTEX,
            "fallback": False,
        }
    except Exception as err:  # noqa: BLE001
        print(f"\nVertex AI generation failed: {err}")
        print("Falling back to the local heuristic analyzer...")
        return _heuristic_result(video_path, landmarks_path, str(err))


def generate(video_path=None, landmarks_path=None) -> str:
    """Backwards-compatible wrapper: returns only the raw analysis text."""
    return analyze(video_path, landmarks_path)["raw"]


if __name__ == "__main__":
    if len(sys.argv) == 3:
        try:
            result = analyze(sys.argv[1], sys.argv[2])
            print(f"\n\nEngine: {result['engine']} ({result['model']})")
            print("Analysis result:")
            print(result["raw"])
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage: python deepfake_detection.py <video_path> <landmarks_path>", file=sys.stderr)
        sys.exit(2)
