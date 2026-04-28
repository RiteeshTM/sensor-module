const { onObjectFinalized } = require("firebase-functions/v2/storage");
const { setGlobalOptions } = require("firebase-functions/v2");
const admin = require("firebase-admin");
const { GoogleGenAI } = require("@google/genai");

setGlobalOptions({ region: "asia-southeast1" });

admin.initializeApp();

exports.processLandmarks = onObjectFinalized({ timeoutSeconds: 540, memory: "512MiB" }, async (event) => {
    const object = event.data;
    const filePath = object.name;
    const bucketName = object.bucket;
    
    console.log("File uploaded:", filePath);

    if (!filePath.endsWith("landmarks.json")) {
      console.log("Not a landmarks file. Skipping.");
      return null;
    }

    console.log("Landmarks file detected. Initiating Gemini Analysis...");

    const db = admin.firestore();
    const bucket = admin.storage().bucket(bucketName);

    const baseName = filePath.replace("_landmarks.json", "");
    const videoPath = `${baseName}.mp4`; 
    
    const landmarksFile = bucket.file(filePath);
    
    const [landmarksBuffer] = await landmarksFile.download();
    const landmarksText = landmarksBuffer.toString("utf-8");

    const projectId = process.env.GOOGLE_CLOUD_PROJECT || "deepfake-detector-494710";
    const vertexLocation = process.env.GEMINI_LOCATION || "global";

    const maxFrames = Number(process.env.GEMINI_MAX_FRAMES || "240");
    const maxTextChars = Number(process.env.GEMINI_MAX_TEXT_CHARS || "20000");

    const summarizeLandmarks = (rawText) => {
      try {
        const records = JSON.parse(rawText);
        if (!Array.isArray(records) || records.length === 0) {
          return rawText.slice(0, maxTextChars);
        }
        const step = Math.max(1, Math.ceil(records.length / maxFrames));
        const sampled = records.filter((_, idx) => idx % step === 0);
        const summary = {
          total_frames: records.length,
          sampled_every_n_frames: step,
          frames: sampled
        };
        const summaryText = JSON.stringify(summary);
        return summaryText.length > maxTextChars
          ? summaryText.slice(0, maxTextChars)
          : summaryText;
      } catch (err) {
        return rawText.slice(0, maxTextChars);
      }
    };
    
    const summarizedLandmarksText = summarizeLandmarks(landmarksText);

    const promptText = `System Persona:
You are an expert Forensic Video Analyst specializing in Behavioral Biometrics and Biological Physics. Your goal is to detect deepfakes by identifying "Kinetic Dissonance"—where the visual movement in a video contradicts the laws of human physiology.

Analysis Framework:
    The Law of Inertia: Human head and jaw movements have mass. Look for "weightless" transitions in the provided data.
    Saccadic Eye Movement: Human eyes move in discrete, high-velocity jumps. Flag "linear sliding" eye movements as AI-generated.
    Biological Noise: Real humans have micro-tremors (3-7 Hz). If the motion data shows "perfect" curves or zero jitter, it is a synthetic interpolation.
    Temporal Lag: Look for delays between the eyes and mouth that exceed 50ms, as AI often desyncs micro-expressions.

I am uploading a video file and a JSON file containing the 3D coordinates (x,y,z) of facial landmarks extracted via MediaPipe.
Task:
Cross-reference the visual frames of the video with the provided coordinate data.
Analyze the Acceleration of the chin (Landmarks 152, 175). Is there a "snap-to-grid" effect or unnatural smoothness?
Check the Gaze Vector consistency. Does the eye movement follow the "Main Sequence" velocity curve?
Identify any "Micro-Expression Spikes"—sudden frame-level changes in landmark positions that do not match the surrounding frames.

Output Format: Return your findings in this EXACT JSON format: 
{ 
  "authenticity_score": [0-100], 
  "flagged_anomalies": ["list specific timestamps and physical reasons - KEEP SHORT!"], 
  "forensic_explanation": "A short, technical summary of why this is human or AI. Max 3 sentences." 
}

LANDMARK DATA (SAMPLED):
${summarizedLandmarksText}`;

    const videoUri = `gs://${bucketName}/${videoPath}`;

    try {
      console.log(`Calling Gemini API with video ${videoUri}...`);
      const ai = new GoogleGenAI({ vertexai: true, project: projectId, location: vertexLocation });

      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: [
            {
              role: "user",
              parts: [
                {
                  fileData: {
                    mimeType: "video/mp4",
                    fileUri: videoUri
                  }
                },
                { text: promptText }
              ]
            }
        ],
        config: {
          temperature: 0.1,
          topP: 0.95,
          maxOutputTokens: 8192,
          responseMimeType: "application/json"
        }
      });
      
      const responseText = response.text;
      
      let cleanResult = responseText;
      if (cleanResult && cleanResult.startsWith("```json")) {
        cleanResult = cleanResult.replace(/```json\n?/g, "").replace(/```/g, "").trim();
      }

      let parsedResult;
      try {
        parsedResult = JSON.parse(cleanResult);
      } catch (err) {
        parsedResult = { raw_output: responseText };
      }

      const docRef = await db.collection("analyses").add({
        video_reference: videoUri,
        landmarks_reference: `gs://${bucketName}/${filePath}`,
        analysis: parsedResult,
        analyzed_at: admin.firestore.FieldValue.serverTimestamp()
      });

      console.log(`Processing successful! Result saved to Firestore: ${docRef.id}`);

    } catch (error) {
      console.error("Error during Gemini deepfake analysis:");
      console.error(error);
    }
    
    return null;
});
