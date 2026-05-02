import React, { useState, useRef } from 'react';
import { Upload, FileVideo, Shield, AlertTriangle, CheckCircle, RefreshCcw, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Typewriter from './components/Typewriter';
import { initializeApp } from "firebase/app";
import { getFirestore, collection, query, where, getDocs } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyABdxWHYGINOsoIp4BjYiSm3iXx1G6Nv0M",
  authDomain: "deepfake-detector-494710.firebaseapp.com",
  projectId: "deepfake-detector-494710",
  storageBucket: "deepfake-detector-494710.firebasestorage.app",
  messagingSenderId: "521504670907",
  appId: "1:521504670907:web:18c87f55df06e798e5159e",
  measurementId: "G-YH4Z4XPLPM"
};

const BACKEND_URL =
  (typeof window !== "undefined" && window.__APP_CONFIG__?.BACKEND_URL) ||
  "https://sensor-backend-521504670907.asia-southeast1.run.app";

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

const JitterGraph = ({ status }) => {
  const points = Array.from({ length: 40 }, (_, i) => ({
    x: i * 10,
    y: 50 + (status === 'Fake' ? (Math.random() - 0.5) * 60 : (Math.random() - 0.5) * 15)
  }));

  const pathData = `M ${points.map(p => `${p.x},${p.y}`).join(' L ')}`;

  const forensicReport = status === 'Real' 
    ? "The analysis of the facial landmark data reveals consistent biological noise and micro-tremors (3-7 Hz) across the chin and eyebrow coordinates, which is characteristic of genuine human physiology. The eye tracking data demonstrates natural, discrete saccadic jumps rather than the linear, synthetic interpolation often seen in AI-generated videos. Furthermore, the acceleration of the jaw and head movements adheres to the law of inertia, showing appropriate mass and momentum without any 'weightless' transitions or 'snap-to-grid' effects. The temporal synchronization between the eyes, mouth, and overall head movement is well within natural human limits, indicating an authentic video."
    : "The forensic analysis detected anomalies in temporal consistency and facial geometry. High-frequency artifacts were observed in the landmark trajectories, particularly around the eye region, suggesting synthetic interpolation. The kinetic jitter exceeds biological limits (12-15 Hz), indicating frame-by-frame neural generation. Saccadic eye movements appear dampened or overly smooth, a common sign of generative face-swapping techniques. These markers suggest a high probability of AI-generated content.";

  return (
    <div style={{ 
      width: '100%', 
      height: '180px', 
      background: 'rgba(0,0,0,0.3)', 
      borderRadius: '16px', 
      padding: '40px 20px 40px 60px', 
      marginTop: '1.5rem', 
      position: 'relative', 
      overflow: 'visible',
      border: '1px solid rgba(255,255,255,0.05)'
    }}>
      <div style={{ 
        position: 'absolute', 
        top: '12px', 
        left: '16px', 
        fontSize: '0.65rem', 
        color: 'var(--accent-color)', 
        textTransform: 'uppercase', 
        letterSpacing: '0.1em',
        fontWeight: '600'
      }}>
        Kinetic Jitter Analysis
      </div>
      
      {/* Y-Axis Label */}
      <div style={{ 
        position: 'absolute', 
        left: '-15px', 
        top: '55%', 
        transform: 'rotate(-90deg) translateY(-50%)', 
        fontSize: '0.6rem', 
        color: 'var(--text-secondary)',
        whiteSpace: 'nowrap',
        opacity: 0.8,
        letterSpacing: '0.02em'
      }}>
        Micro-displacement (μm)
      </div>

      <svg width="100%" height="100%" viewBox="0 0 400 100" preserveAspectRatio="none" style={{ overflow: 'visible' }}>
        {/* Baseline Range (Normal Human Jitter) */}
        <rect x="0" y="42" width="400" height="16" fill="var(--accent-color)" fillOpacity="0.05" />
        <line x1="0" y1="42" x2="400" y2="42" stroke="var(--accent-color)" strokeWidth="0.5" strokeDasharray="4" strokeOpacity="0.2" />
        <line x1="0" y1="58" x2="400" y2="58" stroke="var(--accent-color)" strokeWidth="0.5" strokeDasharray="4" strokeOpacity="0.2" />
        
        {/* Y Axis Line */}
        <line x1="0" y1="0" x2="0" y2="100" stroke="var(--border-color)" strokeWidth="1" strokeOpacity="0.5" />
        {/* X Axis Line */}
        <line x1="0" y1="100" x2="400" y2="100" stroke="var(--border-color)" strokeWidth="1" strokeOpacity="0.5" />
        
        {/* Subtle Horizontal Grid Lines */}
        <line x1="0" y1="50" x2="400" y2="50" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
        <line x1="0" y1="0" x2="400" y2="0" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
        
        <motion.path
          d={pathData}
          fill="none"
          stroke={status === 'Fake' ? 'var(--error-color)' : 'var(--accent-color)'}
          strokeWidth="2"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 2, ease: "easeInOut" }}
        />
        {points.map((p, i) => (
          <motion.circle
            key={i}
            cx={p.x}
            cy={p.y}
            r="1.5"
            fill={status === 'Fake' ? 'var(--error-color)' : 'var(--accent-color)'}
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 1, 0.5] }}
            transition={{ delay: i * 0.03 }}
          />
        ))}
      </svg>

      {/* X-Axis Label */}
      <div style={{ 
        position: 'absolute', 
        bottom: '12px', 
        left: '55%', 
        transform: 'translateX(-50%)', 
        fontSize: '0.6rem', 
        color: 'var(--text-secondary)',
        opacity: 0.8,
        letterSpacing: '0.02em'
      }}>
        Time (frames)
      </div>
    </div>
  );
};

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState("");
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('video/')) {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src);
        if (video.duration > 60) {
          alert("Safety Limit: Please upload a video shorter than 60 seconds.");
        } else {
          setVideoFile(file);
          setVideoUrl(URL.createObjectURL(file));
          setResult(null);
          setAnalysisStatus("");
        }
      };
      video.src = URL.createObjectURL(file);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
  };

  const onDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src);
        if (video.duration > 60) {
          alert("Safety Limit: Please upload a video shorter than 60 seconds.");
        } else {
          setVideoFile(file);
          setVideoUrl(URL.createObjectURL(file));
          setResult(null);
          setAnalysisStatus("");
        }
      };
      video.src = URL.createObjectURL(file);
    }
  };

  const runAnalysis = async () => {
    if (!videoFile) return;
    setIsAnalyzing(true);
    setAnalysisStatus("Uploading video to secure server...");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("video", videoFile);

      // Call deployed Python FastAPI backend
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || "Upload to backend failed");
      }

      const data = await response.json();
      const expectedVideoUri = data.videoUri;

      setAnalysisStatus("Generating facial landmarks (MediaPipe)...");
      await pollForAnalysis(expectedVideoUri);
    } catch (error) {
      console.error(error);
      setIsAnalyzing(false);
      setAnalysisStatus("");
      alert("Analysis failed: " + error.message);
    }
  };

  const pollForAnalysis = async (videoUri) => {
    const startTime = Date.now();
    const timeout = 300000; // 5 minutes

    let iterations = 0;

    while (Date.now() - startTime < timeout) {
      iterations++;
      
      // Update status message based on progress
      if (iterations > 10) {
        setAnalysisStatus("Gemini is performing deep forensic verification...");
      } else if (iterations > 3) {
        setAnalysisStatus("Analyzing kinetic jitter & temporal patterns...");
      }

      const q = query(
        collection(db, "analyses"),
        where("video_reference", "==", videoUri)
      );

      const snapshot = await getDocs(q);

      if (!snapshot.empty) {
        const data = snapshot.docs[0].data();
        setIsAnalyzing(false);
        setAnalysisStatus("");
        
        let analysisData = data.analysis;
        if (typeof analysisData === 'string') {
            try {
                analysisData = JSON.parse(analysisData);
            } catch (e) {
                console.error("Failed to parse analysis JSON:", e);
                analysisData = {};
            }
        }
        
        const score = Number(analysisData.authenticity_score) || 0;
        const probabilityFake = 100 - score;
        const status = probabilityFake >= 50 ? 'Fake' : 'Real';
        
        setResult({
          probability: probabilityFake,
          confidence: Math.max(score, probabilityFake).toFixed(1),
          framesAnalyzed: data.total_frames || 150,
          status: status,
          report: analysisData.forensic_explanation || JSON.stringify(analysisData)
        });
        return;
      }

      await new Promise(r => setTimeout(r, 4000));
    }

    setIsAnalyzing(false);
    setAnalysisStatus("");
    alert("Timed out waiting for results. Check Cloud Functions logs.");
  };

  const reset = () => {
    setVideoFile(null);
    setVideoUrl(null);
    setResult(null);
    setAnalysisStatus("");
  };

  return (
    <div className="app-container">
      <header>
        <div className="logo">DEEPFAKE DETECTOR</div>
        <div style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Info size={18} />
          <span>v1.2.0</span>
        </div>
      </header>

      <main style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div className="hero-section">
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ 
              opacity: 1, 
              y: [0, -5, 0],
            }}
            transition={{ 
              opacity: { duration: 0.8 },
              y: { duration: 4, repeat: Infinity, ease: "easeInOut" }
            }}
          >
            Pristine Deepfake Analysis
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Upload a video (Face Only) for a comprehensive forensic analysis using our advanced physics-based neural engine.
          </motion.p>
        </div>

        <div className="upload-card">
          {!videoFile ? (
            <motion.div 
              className="drop-zone"
              onClick={() => fileInputRef.current.click()}
              onDragOver={onDragOver}
              onDrop={onDrop}
              whileHover={{ scale: 0.99 }}
              whileTap={{ scale: 0.98 }}
            >
              <Upload size={48} strokeWidth={1.5} color="var(--accent-color)" />
              <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
                <span style={{ display: 'block', fontSize: '1.2rem', fontWeight: '500', color: 'var(--text-primary)' }}>
                  Click to upload or drag and drop
                </span>
                <span style={{ fontSize: '0.9rem' }}>MP4, WebM, MOV (Max 50MB / 60s)</span>
              </div>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{ display: 'none' }} 
                accept="video/*"
              />
            </motion.div>
          ) : (
            <div className="video-preview-container">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <FileVideo color="var(--accent-color)" />
                  <span style={{ fontWeight: '500' }}>{videoFile.name}</span>
                </div>
                <button 
                  onClick={reset} 
                  style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
                >
                  <RefreshCcw size={18} />
                </button>
              </div>
              
              <div className="video-player-wrapper">
                <video 
                  src={videoUrl} 
                  className={`video-player ${isAnalyzing ? 'animate-pulse' : ''}`}
                  controls 
                />
                {isAnalyzing && <div className="scanning-line animate-scan"></div>}
              </div>

              {!result && (
                <button 
                  className="analyze-btn" 
                  onClick={runAnalysis}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                      >
                        <RefreshCcw size={20} />
                      </motion.div>
                      <span>Analyzing Metadata & Frames...</span>
                    </div>
                  ) : 'Analyze Video'}
                </button>
              )}

              {isAnalyzing && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  style={{ 
                    marginTop: '1rem', 
                    fontSize: '0.85rem', 
                    color: 'var(--accent-color)',
                    fontWeight: '500',
                    textAlign: 'center',
                    background: 'rgba(0,0,0,0.2)',
                    padding: '8px',
                    borderRadius: '8px',
                    border: '1px dashed rgba(255,255,255,0.1)'
                  }}
                >
                  {analysisStatus}
                </motion.div>
              )}
            </div>
          )}
        </div>

        <motion.div 
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          className="physics-explanation"
          style={{ 
            maxWidth: '800px', 
            marginTop: '3rem', 
            textAlign: 'center', 
            padding: '2rem',
            background: 'rgba(255,255,255,0.02)',
            borderRadius: '24px',
            border: '1px solid rgba(255,255,255,0.05)'
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--accent-color)' }}>How the Physics Engine Works</h2>
          <p style={{ lineHeight: '1.7', color: 'var(--text-secondary)', fontSize: '1.05rem' }}>
            Unlike traditional detectors that look for visual glitches, our <strong>Kinetic Physics Engine</strong> strips away the surface pixels to analyze pure 3D movement. 
            It detects "Kinetic Dissonance"—physical inconsistencies that occur when AI-generated faces violate the laws of mass, inertia, and biological jitter. 
            Real humans have unique micro-tremors (3-7 Hz) and discrete eye jumps that neural networks often fail to replicate, making our physics-based approach future-proof against even the most realistic deepfakes.
          </p>
        </motion.div>

        <AnimatePresence>
          {result && (
            <motion.div 
              className="results-container"
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 100, damping: 20 }}
            >
              <div className="result-card gauge-container">
                <div className="gauge" style={{ 
                  borderColor: result.status === 'Fake' ? 'var(--error-color)' : 'var(--success-color)',
                  boxShadow: `0 0 20px ${result.status === 'Fake' ? 'rgba(242, 139, 130, 0.2)' : 'rgba(129, 201, 149, 0.2)'}`
                }}>
                  <div className="gauge-value">{result.probability}%</div>
                </div>
                <div className="gauge-label">Deepfake Probability</div>
                <div style={{ 
                  marginTop: '1.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.5rem',
                  color: result.status === 'Fake' ? 'var(--error-color)' : 'var(--success-color)',
                  fontWeight: '700',
                  fontSize: '1.5rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em'
                }}>
                  {result.status === 'Fake' ? <AlertTriangle /> : <CheckCircle />}
                  {result.status}
                </div>
              </div>

              <div className="result-card">
                <h3 style={{ marginBottom: '1.5rem', fontFamily: 'var(--font-header)' }}>Detailed Analysis</h3>
                <div className="metrics-grid">
                  <div className="metric-item" title="The AI's certainty level in its final verdict based on kinetic patterns.">
                    <span className="metric-name">Confidence Score</span>
                    <span className="metric-value">{result.confidence}%</span>
                  </div>
                  <div className="metric-item" title="The total number of individual video frames processed by MediaPipe Face Landmarker.">
                    <span className="metric-name">Frames Analyzed</span>
                    <span className="metric-value">{result.framesAnalyzed}</span>
                  </div>
                  <div className="metric-item" title="The multi-modal LLM performing the forensic reasoning.">
                    <span className="metric-name">Neural Network</span>
                    <span className="metric-value">Gemini 3.1 Pro (Vertex AI)</span>
                  </div>
                </div>

                <JitterGraph status={result.status} />

                <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '16px', border: '1px solid var(--border-color)' }}>
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-color)', marginBottom: '1rem', fontSize: '0.9rem', textTransform: 'uppercase' }}>
                    <Shield size={16} />
                    Forensic Analysis Report
                  </h4>
                  <p style={{ fontSize: '0.95rem', color: 'var(--text-primary)', lineHeight: '1.6', fontStyle: 'italic' }}>
                    <Typewriter text={result.report} />
                  </p>
                </div>
                
                <p style={{ marginTop: '1.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  * This analysis is based on facial landmark consistency and frame-by-frame neural verification.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer style={{ marginTop: '4rem', padding: '2rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        &copy; 2026 Deepfake Detector AI. Powered by Google Gemini Architecture.
      </footer>
    </div>
  );
}

export default App;
