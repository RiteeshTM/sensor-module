const { onObjectFinalized } = require("firebase-functions/v2/storage");
const { setGlobalOptions } = require("firebase-functions/v2");
const admin = require("firebase-admin");
const { VertexAI } = require("@google-cloud/vertexai");

setGlobalOptions({ region: "asia-southeast1" });

admin.initializeApp();

exports.processLandmarks = onObjectFinalized(async (event) => {

    const object = event.data;
    const filePath = object.name;
    const bucketName = object.bucket;
    
    console.log("File uploaded:", filePath);

    // Trigger when a landmarks.json is uploaded
    if (!filePath.endsWith("landmarks.json")) {
      console.log("Not a landmarks file. Skipping.");
      return null;
    }

    console.log("Landmarks file detected. Initiating Gemini Analysis...");

    const db = admin.firestore();
    const bucket = admin.storage().bucket(bucketName);

    // Assume the video has the same base name, e.g., "download" from "download_landmarks.json"
    const baseName = filePath.replace("_landmarks.json", "");
    const videoPath = `${baseName}.mp4`; 
    
    const landmarksFile = bucket.file(filePath);
    
    // Download landmarks content to string
    const [landmarksBuffer] = await landmarksFile.download();
    const landmarksText = landmarksBuffer.toString("utf-8");

    const projectId = process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT || "deepfake-detector-494710";
    
    // Initialize Vertex AI
    const vertexAI = new VertexAI({ project: projectId, location: "global" });
    const generativeModel = vertexAI.getGenerativeModel({
      model: "gemini-3.1-pro-preview",
      generationConfig: {
        temperature: 0.1,
        topP: 0.95,
        maxOutputTokens: 65535,
      },
      systemInstruction: {
        parts: [{
          text: `System Persona:\nYou are an expert Forensic Video Analyst specializing in Behavioral Biometrics and Biological Physics. Your goal is to detect deepfakes by identifying "Kinetic Dissonance"—where the visual movement in a video contradicts the laws of human physiology.\n\nAnalysis Framework:\n\n    The Law of Inertia: Human head and jaw movements have mass. Look for "weightless" transitions in the provided data.\n\n    Saccadic Eye Movement: Human eyes move in discrete, high-velocity jumps. Flag "linear sliding" eye movements as AI-generated.\n\n    Biological Noise: Real humans have micro-tremors (3-7 Hz). If the motion data shows "perfect" curves or zero jitter, it is a synthetic interpolation.\n\n    Temporal Lag: Look for delays between the eyes and mouth that exceed 50ms, as AI often desyncs micro-expressions.`
        }]
      }
    });

    const promptText = `I am uploading a video file and a JSON file containing the 3D coordinates (x,y,z) of facial landmarks extracted via MediaPipe.Task:\nCross-reference the visual frames of the video with the provided coordinate data.\nAnalyze the Acceleration (a=Δv/Δt) of the chin (Landmarks 152, 175). Is there a "snap-to-grid" effect or unnatural smoothness?\nCheck the Gaze Vector consistency. Does the eye movement follow the "Main Sequence" velocity curve?\nIdentify any "Micro-Expression Spikes"—sudden frame-level changes in landmark positions that don't match the surrounding frames (common in diffusion-based flickering).\n\nOutput Format: Return your findings in this EXACT JSON format: \n{ \n  "authenticity_score": [0-100], \n  "flagged_anomalies": ["list specific timestamps and physical reasons"], \n  "forensic_explanation": "A concise, technical summary of why this is human or AI." \n}\n\nLANDMARK DATA:\n${landmarksText}`;

    // Provide the video to Vertex AI directly using its Google Cloud Storage URI
    const videoUri = `gs://${bucketName}/${videoPath}`;
    
    const request = {
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
      ]
    };

    try {
      console.log(`Calling Gemini Vertex AI with video ${videoUri}...`);
      const response = await generativeModel.generateContent(request);
      
      const responseText = response.response.candidates[0].content.parts[0].text;
      
      // Clean up markdown block if present
      let cleanResult = responseText;
      if (cleanResult.startsWith("```json")) {
        cleanResult = cleanResult.replace(/```json\n?/g, "").replace(/```/g, "").trim();
      }

      let parsedResult;
      try {
        parsedResult = JSON.parse(cleanResult);
      } catch (err) {
        parsedResult = { raw_output: responseText };
      }

      // Save to Firestore
      const docRef = await db.collection("analyses").add({
        video_reference: videoUri,
        landmarks_reference: `gs://${bucketName}/${filePath}`,
        analysis: parsedResult,
        analyzed_at: admin.firestore.FieldValue.serverTimestamp()
      });

      console.log(`Processing successful! Result saved to Firestore: ${docRef.id}`);

    } catch (error) {
      console.error("Error during Gemini deepfake analysis:", error);
    }
    
    return null;
  });