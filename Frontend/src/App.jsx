import React, { useState, useRef, useMemo } from 'react';
import { Upload, FileVideo, Shield, AlertTriangle, CheckCircle, RefreshCcw, Info, HelpCircle, XCircle } from 'lucide-react';
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

// Local development always wins: when the page is served from localhost we talk
// to the local FastAPI server, regardless of what config.js says. Anywhere else
// we use the runtime override from public/config.js, falling back to Cloud Run.
const IS_LOCAL =
  typeof window !== "undefined" &&
  (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");

const BACKEND_URL = IS_LOCAL
  ? "http://localhost:8000"
  : ((typeof window !== "undefined" && window.__APP_CONFIG__?.BACKEND_URL) ||
     "https://sensor-backend-521504670907.asia-southeast1.run.app");

const MAX_DURATION_SECONDS = 60;
const MAX_FILE_BYTES = 50 * 1024 * 1024;

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// --- Formatting helpers -------------------------------------------------------

const fmtPct = (value, digits = 1) =>
  Number.isFinite(value) ? `${Number(value).toFixed(digits)}%` : "--";

const fmtNum = (value, digits = 3) =>
  Number.isFinite(value) ? Number(value).toFixed(digits) : "--";

const statusColor = (status) => {
  if (status === 'Fake') return 'var(--error-color)';
  if (status === 'Real') return 'var(--success-color)';
  return 'var(--text-secondary)';
};

const StatusIcon = ({ status, ...props }) => {
  if (status === 'Fake') return <AlertTriangle {...props} />;
  if (status === 'Real') return <CheckCircle {...props} />;
  return <HelpCircle {...props} />;
};

// --- Kinetic jitter chart -----------------------------------------------------
// Plots the real per-frame chin velocity series returned by the backend.
// No synthetic data: if the series is absent, the chart says so.

const JitterGraph = ({ series, meanVelocity, stdVelocity, status }) => {
  const data = useMemo(
    () => (Array.isArray(series) ? series.filter((p) => p && Number.isFinite(p.v)) : []),
    [series]
  );

  const chart = useMemo(() => {
    if (data.length < 2) return null;

    const W = 400;
    const H = 100;
    const values = data.map((p) => p.v);

    let vMin = Math.min(...values);
    let vMax = Math.max(...values);
    if (vMax - vMin < 1e-9) {
      // Perfectly flat signal - pad the range so the line renders mid-frame.
      const pad = Math.max(Math.abs(vMax) * 0.1, 1e-6);
      vMin -= pad;
      vMax += pad;
    }
    const span = vMax - vMin;

    const x = (i) => (i / (data.length - 1)) * W;
    const y = (v) => H - ((v - vMin) / span) * H;
    const clampY = (v) => Math.max(0, Math.min(H, y(v)));

    const path = data
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(2)},${y(p.v).toFixed(2)}`)
      .join(' ');

    const mean = Number.isFinite(meanVelocity)
      ? meanVelocity
      : values.reduce((a, b) => a + b, 0) / values.length;
    const sd = Number.isFinite(stdVelocity) ? stdVelocity : 0;

    return {
      W,
      H,
      path,
      points: data.map((p, i) => ({ cx: x(i), cy: y(p.v) })),
      meanY: clampY(mean),
      bandTopY: clampY(mean + sd),
      bandBottomY: clampY(mean - sd),
      vMax,
      tStart: data[0].t,
      tEnd: data[data.length - 1].t
    };
  }, [data, meanVelocity, stdVelocity]);

  const frameStyle = {
    width: '100%',
    background: 'rgba(0,0,0,0.3)',
    borderRadius: '16px',
    marginTop: '1.5rem',
    border: '1px solid rgba(255,255,255,0.05)'
  };

  if (!chart) {
    return (
      <div style={{ ...frameStyle, padding: '2rem', textAlign: 'center' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--accent-color)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600, marginBottom: '0.75rem' }}>
          Kinetic Jitter Analysis
        </div>
        <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
          Per-frame velocity data is not available for this run.
        </div>
      </div>
    );
  }

  const lineColor = status === 'Fake' ? 'var(--error-color)' : 'var(--accent-color)';
  const showPoints = chart.points.length <= 80;

  return (
    <div style={{ ...frameStyle, height: '200px', padding: '40px 20px 40px 62px', position: 'relative', overflow: 'visible' }}>
      <div style={{ position: 'absolute', top: '12px', left: '16px', fontSize: '0.65rem', color: 'var(--accent-color)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600 }}>
        Kinetic Jitter Analysis
      </div>
      <div style={{ position: 'absolute', top: '12px', right: '16px', fontSize: '0.6rem', color: 'var(--text-secondary)' }}>
        {data.length} sampled points
      </div>

      {/* Y-axis label */}
      <div style={{ position: 'absolute', left: '-24px', top: '55%', transform: 'rotate(-90deg) translateY(-50%)', fontSize: '0.6rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap', opacity: 0.8 }}>
        Chin velocity (units/s)
      </div>
      {/* Y-axis max tick */}
      <div style={{ position: 'absolute', left: '14px', top: '34px', fontSize: '0.55rem', color: 'var(--text-secondary)', opacity: 0.7 }}>
        {chart.vMax.toFixed(3)}
      </div>
      <div style={{ position: 'absolute', left: '14px', bottom: '34px', fontSize: '0.55rem', color: 'var(--text-secondary)', opacity: 0.7 }}>
        0
      </div>

      <svg width="100%" height="100%" viewBox="0 0 400 100" preserveAspectRatio="none" style={{ overflow: 'visible' }}>
        {/* +/- 1 sigma band around the measured mean velocity */}
        <rect
          x="0"
          y={Math.min(chart.bandTopY, chart.bandBottomY)}
          width="400"
          height={Math.max(1, Math.abs(chart.bandBottomY - chart.bandTopY))}
          fill="var(--accent-color)"
          fillOpacity="0.07"
        />
        <line x1="0" y1={chart.meanY} x2="400" y2={chart.meanY} stroke="var(--accent-color)" strokeWidth="0.6" strokeDasharray="4" strokeOpacity="0.45" />

        {/* Axes */}
        <line x1="0" y1="0" x2="0" y2="100" stroke="var(--border-color)" strokeWidth="1" strokeOpacity="0.5" />
        <line x1="0" y1="100" x2="400" y2="100" stroke="var(--border-color)" strokeWidth="1" strokeOpacity="0.5" />

        <motion.path
          d={chart.path}
          fill="none"
          stroke={lineColor}
          strokeWidth="1.6"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.6, ease: "easeInOut" }}
        />

        {showPoints && chart.points.map((p, i) => (
          <motion.circle
            key={i}
            cx={p.cx}
            cy={p.cy}
            r="1.4"
            fill={lineColor}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.85 }}
            transition={{ delay: i * 0.01 }}
          />
        ))}
      </svg>

      <div style={{ position: 'absolute', bottom: '12px', left: '62px', fontSize: '0.55rem', color: 'var(--text-secondary)', opacity: 0.7 }}>
        {chart.tStart.toFixed(1)}s
      </div>
      <div style={{ position: 'absolute', bottom: '12px', left: '50%', transform: 'translateX(-50%)', fontSize: '0.6rem', color: 'var(--text-secondary)', opacity: 0.8 }}>
        Time - dashed line = mean, shaded band = +/- 1 sigma
      </div>
      <div style={{ position: 'absolute', bottom: '12px', right: '20px', fontSize: '0.55rem', color: 'var(--text-secondary)', opacity: 0.7 }}>
        {chart.tEnd.toFixed(1)}s
      </div>
    </div>
  );
};

// --- App ----------------------------------------------------------------------

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const acceptFile = (file) => {
    setError("");

    if (!file) return;
    if (!file.type.startsWith('video/')) {
      setError("That file is not a video. Supported formats: MP4, WebM, MOV.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setError(`That file is ${(file.size / 1e6).toFixed(1)} MB. The limit is ${MAX_FILE_BYTES / 1e6} MB.`);
      return;
    }

    const probe = document.createElement('video');
    probe.preload = 'metadata';
    probe.onloadedmetadata = () => {
      window.URL.revokeObjectURL(probe.src);
      if (probe.duration > MAX_DURATION_SECONDS) {
        setError(`That clip is ${probe.duration.toFixed(0)}s long. Please use one under ${MAX_DURATION_SECONDS}s.`);
        return;
      }
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setResult(null);
      setAnalysisStatus("");
    };
    probe.onerror = () => {
      window.URL.revokeObjectURL(probe.src);
      setError("This video could not be read by the browser. Try re-encoding it as MP4 (H.264).");
    };
    probe.src = URL.createObjectURL(file);
  };

  const handleFileChange = (e) => acceptFile(e.target.files[0]);

  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    acceptFile(e.dataTransfer.files[0]);
  };

  const runAnalysis = async () => {
    if (!videoFile) return;
    setIsAnalyzing(true);
    setError("");
    setAnalysisStatus("Uploading video and extracting landmarks...");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("video", videoFile);

      const response = await fetch(`${BACKEND_URL}/analyze`, { method: "POST", body: formData });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Backend returned ${response.status}`);
      }

      const data = await response.json();

      // Direct mode: the backend returns the verdict in the response body.
      if (data.result) {
        setResult(data.result);
        setIsAnalyzing(false);
        setAnalysisStatus("");
        return;
      }

      // Cloud mode: a Cloud Function writes the verdict to Firestore.
      setAnalysisStatus("Generating facial landmarks (MediaPipe)...");
      await pollForAnalysis(data.videoUri);
    } catch (err) {
      console.error(err);
      setIsAnalyzing(false);
      setAnalysisStatus("");
      setError(
        `${err.message}. ` +
        (IS_LOCAL ? "Is the backend running at " + BACKEND_URL + "?" : "The analysis backend may be unavailable.")
      );
    }
  };

  const pollForAnalysis = async (videoUri) => {
    const startTime = Date.now();
    const timeout = 300000; // 5 minutes
    let iterations = 0;

    while (Date.now() - startTime < timeout) {
      iterations++;
      if (iterations > 10) setAnalysisStatus("Gemini is performing deep forensic verification...");
      else if (iterations > 3) setAnalysisStatus("Analyzing kinetic jitter and temporal patterns...");

      const snapshot = await getDocs(
        query(collection(db, "analyses"), where("video_reference", "==", videoUri))
      );

      if (!snapshot.empty) {
        const doc = snapshot.docs[0].data();
        setIsAnalyzing(false);
        setAnalysisStatus("");

        let analysisData = doc.analysis;
        if (typeof analysisData === 'string') {
          try { analysisData = JSON.parse(analysisData); }
          catch (e) { console.error("Failed to parse analysis JSON:", e); analysisData = {}; }
        }
        analysisData = analysisData || {};

        const rawScore = Number(analysisData.authenticity_score);
        const hasScore = Number.isFinite(rawScore);
        const probabilityFake = hasScore ? 100 - rawScore : null;

        setResult({
          status: hasScore ? (probabilityFake >= 50 ? 'Fake' : 'Real') : 'Inconclusive',
          probability: probabilityFake,
          confidence: hasScore ? Math.max(rawScore, probabilityFake) : null,
          framesAnalyzed: doc.total_frames ?? null,
          engine: "Vertex AI",
          model: "gemini-3.1-pro-preview",
          anomalies: Array.isArray(analysisData.flagged_anomalies) ? analysisData.flagged_anomalies : [],
          // The Cloud Function pipeline does not return a per-frame series.
          jitterSeries: [],
          report: analysisData.forensic_explanation
            || analysisData.raw_output
            || (hasScore ? "" : "The analysis engine did not return a usable authenticity score."),
          warning: hasScore ? undefined : "No authenticity score was returned by the analysis engine."
        });
        return;
      }

      await new Promise(r => setTimeout(r, 4000));
    }

    setIsAnalyzing(false);
    setAnalysisStatus("");
    setError("Timed out waiting for results. Check the Cloud Functions logs.");
  };

  const reset = () => {
    setVideoFile(null);
    setVideoUrl(null);
    setResult(null);
    setAnalysisStatus("");
    setError("");
  };

  return (
    <div className="app-container">
      <header>
        <div className="logo">DEEPFAKE DETECTOR</div>
        <div style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Info size={18} />
          <span>v2.1.0</span>
        </div>
      </header>

      <main style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div className="hero-section">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: [0, -5, 0] }}
            transition={{ opacity: { duration: 0.8 }, y: { duration: 4, repeat: Infinity, ease: "easeInOut" } }}
          >
            Pristine Deepfake Analysis
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Upload a video (face only) for a forensic analysis of how the face <em>moves</em>,
            not how it looks.
          </motion.p>
        </div>

        <div className="upload-card">
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              style={{
                display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
                marginBottom: '1.25rem', padding: '0.9rem 1rem',
                background: 'rgba(242, 139, 130, 0.08)',
                border: '1px solid rgba(242, 139, 130, 0.35)',
                borderRadius: '12px', color: 'var(--error-color)', fontSize: '0.9rem'
              }}
            >
              <XCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
              <span>{error}</span>
            </motion.div>
          )}

          {!videoFile ? (
            <motion.div
              className={`drop-zone${isDragging ? ' active' : ''}`}
              onClick={() => fileInputRef.current.click()}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              whileHover={{ scale: 0.99 }}
              whileTap={{ scale: 0.98 }}
            >
              <Upload size={48} strokeWidth={1.5} color="var(--accent-color)" />
              <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
                <span style={{ display: 'block', fontSize: '1.2rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                  Click to upload or drag and drop
                </span>
                <span style={{ fontSize: '0.9rem' }}>MP4, WebM, MOV (max 50 MB / 60 s)</span>
              </div>
              <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} accept="video/*" />
            </motion.div>
          ) : (
            <div className="video-preview-container">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: 0 }}>
                  <FileVideo color="var(--accent-color)" style={{ flexShrink: 0 }} />
                  <span style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {videoFile.name}
                  </span>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', flexShrink: 0 }}>
                    {(videoFile.size / 1e6).toFixed(1)} MB
                  </span>
                </div>
                <button onClick={reset} title="Choose a different video"
                  style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  <RefreshCcw size={18} />
                </button>
              </div>

              <div className="video-player-wrapper">
                <video src={videoUrl} className="video-player" controls />
                {isAnalyzing && <div className="scanning-line animate-scan"></div>}
              </div>

              {!result && (
                <button className="analyze-btn" onClick={runAnalysis} disabled={isAnalyzing}>
                  {isAnalyzing ? (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                      <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }}>
                        <RefreshCcw size={20} />
                      </motion.div>
                      <span>Analyzing frames...</span>
                    </div>
                  ) : 'Analyze Video'}
                </button>
              )}

              {isAnalyzing && (
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  style={{
                    marginTop: '1rem', fontSize: '0.85rem', color: 'var(--accent-color)',
                    fontWeight: 500, textAlign: 'center', background: 'rgba(0,0,0,0.2)',
                    padding: '8px', borderRadius: '8px', border: '1px dashed rgba(255,255,255,0.1)'
                  }}
                >
                  {analysisStatus}
                </motion.div>
              )}
            </div>
          )}
        </div>

        <motion.div
          initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
          className="physics-explanation"
          style={{
            maxWidth: '800px', marginTop: '3rem', textAlign: 'center', padding: '2rem',
            background: 'rgba(255,255,255,0.02)', borderRadius: '24px',
            border: '1px solid rgba(255,255,255,0.05)'
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--accent-color)' }}>How the Physics Engine Works</h2>
          <p style={{ lineHeight: 1.7, color: 'var(--text-secondary)', fontSize: '1.05rem' }}>
            Unlike traditional detectors that look for visual glitches, the <strong>Kinetic Physics Engine</strong> strips
            away surface pixels and analyses pure 3D movement. It looks for "kinetic dissonance" - physical
            inconsistencies that appear when generated faces violate the laws of mass, inertia and biological jitter.
            Real humans have micro-tremors (3-7 Hz) and discrete saccadic eye jumps that generative models
            still struggle to reproduce.
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
                  borderColor: statusColor(result.status),
                  boxShadow: result.status === 'Inconclusive'
                    ? 'none'
                    : `0 0 20px ${result.status === 'Fake' ? 'rgba(242, 139, 130, 0.2)' : 'rgba(129, 201, 149, 0.2)'}`
                }}>
                  <div className="gauge-value" style={{ fontSize: Number.isFinite(result.probability) ? '3rem' : '2rem' }}>
                    {Number.isFinite(result.probability) ? `${Number(result.probability).toFixed(1)}%` : 'N/A'}
                  </div>
                </div>
                <div className="gauge-label">Deepfake Probability</div>
                <div style={{
                  marginTop: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem',
                  color: statusColor(result.status), fontWeight: 700, fontSize: '1.4rem',
                  textTransform: 'uppercase', letterSpacing: '0.08em', textAlign: 'center'
                }}>
                  <StatusIcon status={result.status} />
                  {result.status}
                </div>

                {result.warning && (
                  <div style={{
                    marginTop: '1.25rem', fontSize: '0.8rem', color: 'var(--text-secondary)',
                    textAlign: 'center', lineHeight: 1.5, maxWidth: '260px'
                  }}>
                    {result.warning}
                  </div>
                )}
              </div>

              <div className="result-card">
                <h3 style={{ marginBottom: '1.5rem', fontFamily: 'var(--font-header)' }}>Detailed Analysis</h3>

                <div className="metrics-grid">
                  <div className="metric-item" title="How certain the engine is in its verdict.">
                    <span className="metric-name">Confidence</span>
                    <span className="metric-value">{fmtPct(result.confidence)}</span>
                  </div>
                  <div className="metric-item" title="Total frames decoded and passed through MediaPipe.">
                    <span className="metric-name">Frames Analyzed</span>
                    <span className="metric-value">
                      {Number.isFinite(result.framesAnalyzed) ? result.framesAnalyzed : '--'}
                    </span>
                  </div>
                  {Number.isFinite(result.detectionRate) && (
                    <div className="metric-item" title="Share of frames where a face was located. Low values suggest face-swap warping.">
                      <span className="metric-name">Face Detection Rate</span>
                      <span className="metric-value">{fmtPct(result.detectionRate * 100)}</span>
                    </div>
                  )}
                  {Number.isFinite(result.jitterRatio) && (
                    <div className="metric-item" title="Standard deviation over mean of chin velocity. Below 0.15 is suspiciously smooth.">
                      <span className="metric-name">Kinetic Jitter (sigma/mu)</span>
                      <span className="metric-value">{fmtNum(result.jitterRatio)}</span>
                    </div>
                  )}
                  {Number.isFinite(result.saccadePeaks) && (
                    <div className="metric-item" title="Discrete eye jumps detected. Natural gaze produces many; interpolated gaze produces almost none.">
                      <span className="metric-name">Saccadic Events</span>
                      <span className="metric-value">{result.saccadePeaks}</span>
                    </div>
                  )}
                  <div className="metric-item" title="The engine that actually produced this verdict.">
                    <span className="metric-name">Analysis Engine</span>
                    <span className="metric-value" style={{ textAlign: 'right' }}>
                      {result.engine || 'Unknown'}
                      {result.model && (
                        <span style={{ display: 'block', fontWeight: 400, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                          {result.model}
                        </span>
                      )}
                    </span>
                  </div>
                </div>

                {result.usedFallback && (
                  <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    The cloud model was unreachable, so the local physics engine produced this verdict.
                  </div>
                )}

                <JitterGraph
                  series={result.jitterSeries}
                  meanVelocity={result.meanVelocity}
                  stdVelocity={result.stdVelocity}
                  status={result.status}
                />

                {Array.isArray(result.anomalies) && result.anomalies.length > 0 && (
                  <div style={{ marginTop: '1.5rem' }}>
                    <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-color)', marginBottom: '0.75rem', fontSize: '0.9rem', textTransform: 'uppercase' }}>
                      <AlertTriangle size={16} />
                      Flagged Anomalies
                    </h4>
                    <ul style={{ paddingLeft: '1.1rem', color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: 1.6 }}>
                      {result.anomalies.map((a, i) => <li key={i} style={{ marginBottom: '0.4rem' }}>{a}</li>)}
                    </ul>
                  </div>
                )}

                {result.report && (
                  <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '16px', border: '1px solid var(--border-color)' }}>
                    <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-color)', marginBottom: '1rem', fontSize: '0.9rem', textTransform: 'uppercase' }}>
                      <Shield size={16} />
                      Forensic Analysis Report
                    </h4>
                    <p style={{ fontSize: '0.95rem', color: 'var(--text-primary)', lineHeight: 1.6, fontStyle: 'italic' }}>
                      <Typewriter text={result.report} />
                    </p>
                  </div>
                )}

                <p style={{ marginTop: '1.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  * Every number above is computed from the landmark data extracted from this clip.
                  This is a research prototype, not a legal authentication tool.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer style={{ marginTop: '4rem', padding: '2rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        Kinetic Forensics - physics-based deepfake analysis. Powered by MediaPipe and Google Gemini.
      </footer>
    </div>
  );
}

export default App;
