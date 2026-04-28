const { GoogleGenAI } = require('@google/genai');

async function test() {
  const ai = new GoogleGenAI({
    vertexai: { project: 'deepfake-detector-494710', location: 'us-central1' }
  });

  const response = await ai.models.generateContent({
    model: 'gemini-1.5-flash-002',
    contents: 'Hello',
  });
  
  console.log(response.text);
}

test().catch(console.error);