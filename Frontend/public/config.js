// Runtime backend override for DEPLOYED builds (Firebase Hosting, any static host).
// Edit this file after `npm run build` without rebuilding the bundle.
//
// This is ignored when the page is served from localhost/127.0.0.1 - local
// development always targets http://localhost:8000 so you cannot accidentally
// point a dev session at production.
window.__APP_CONFIG__ = {
  BACKEND_URL: "https://sensor-backend-521504670907.asia-southeast1.run.app"
};
