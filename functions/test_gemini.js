const { GoogleGenAI } = require("@google/genai");

async function test() {
  const projectId = "deepfake-detector-494710";
  const vertexLocation = "global";
  const ai = new GoogleGenAI({ vertexai: true, project: projectId, location: vertexLocation });

  const promptText = 'Return a giant JSON response to test if it gets cut off.';

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.1-pro-preview",
      contents: promptText,
      config: {
        temperature: 0.1,
        topP: 0.95,
        maxOutputTokens: 2048,
        responseMimeType: "application/json"
      }
    });
    console.log("Raw text:", response.text);
    console.log("Candidates:", JSON.stringify(response.candidates, null, 2));
  } catch(e) {
    console.error(e);
  }
}
test();
