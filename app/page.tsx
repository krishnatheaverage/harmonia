"use client";

import { ChangeEvent, CSSProperties, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  analyzeFace,
  detectLandmarks,
  drawLandmarkOverlay,
  getPreset,
  morphImage,
  PRESETS,
  type FaceAnalysis,
  type Point,
} from "../lib/morph";

type Status = "empty" | "camera" | "loading" | "ready" | "error";

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("This image could not be opened."));
    };
    image.src = url;
  });
}

function fitDimensions(width: number, height: number, maxSide = 1800) {
  const ratio = Math.min(1, maxSide / Math.max(width, height));
  return { width: Math.round(width * ratio), height: Math.round(height * ratio) };
}

export default function Home() {
  const originalRef = useRef<HTMLCanvasElement>(null);
  const resultRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sourceRef = useRef<HTMLCanvasElement | null>(null);
  const landmarksRef = useRef<Point[] | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanRunRef = useRef(0);
  const [status, setStatus] = useState<Status>("empty");
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const [presetId, setPresetId] = useState("harmony");
  const [strength, setStrength] = useState(55);
  const [showOriginal, setShowOriginal] = useState(false);
  const [showMesh, setShowMesh] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [regions, setRegions] = useState<string[]>([]);
  const [movement, setMovement] = useState(0);
  const [analysis, setAnalysis] = useState<FaceAnalysis | null>(null);
  const [scanProgress, setScanProgress] = useState(0);

  const stopCamera = useCallback(() => {
    scanRunRef.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const paint = useCallback(() => {
    const source = sourceRef.current;
    const landmarks = landmarksRef.current;
    const resultCanvas = resultRef.current;
    const originalCanvas = originalRef.current;
    if (!source || !landmarks || !resultCanvas || !originalCanvas) return;

    const transformed = morphImage(source, landmarks, getPreset(presetId), strength);
    for (const [destination, image] of [
      [resultCanvas, transformed.canvas],
      [originalCanvas, source],
    ] as const) {
      destination.width = image.width;
      destination.height = image.height;
      const context = destination.getContext("2d", { alpha: false });
      context?.drawImage(image, 0, 0);
    }
    if (showMesh) drawLandmarkOverlay(resultCanvas, landmarks);
    setRegions(transformed.movedRegions);
    setMovement(transformed.maxMovementPx);
  }, [presetId, strength, showMesh]);

  useEffect(() => {
    if (status !== "ready") return;
    const timer = window.setTimeout(paint, 25);
    return () => window.clearTimeout(timer);
  }, [paint, status]);

  const processFile = async (file?: File) => {
    if (!file) return;
    stopCamera();
    if (!file.type.startsWith("image/")) {
      setError("Choose a JPG, PNG, or WebP portrait.");
      setStatus("error");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("Choose an image smaller than 20 MB.");
      setStatus("error");
      return;
    }
    setStatus("loading");
    setError("");
    setFileName(file.name);
    try {
      const image = await loadImage(file);
      const dimensions = fitDimensions(image.naturalWidth, image.naturalHeight);
      const source = document.createElement("canvas");
      source.width = dimensions.width;
      source.height = dimensions.height;
      const context = source.getContext("2d", { alpha: false });
      if (!context) throw new Error("Canvas is unavailable in this browser.");
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(image, 0, 0, dimensions.width, dimensions.height);

      const landmarks = await detectLandmarks(source);
      sourceRef.current = source;
      landmarksRef.current = landmarks;
      setAnalysis(analyzeFace(landmarks));
      setStatus("ready");
      window.setTimeout(paint, 0);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "The portrait could not be analyzed.";
      setError(message);
      setStatus("error");
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void processFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    void processFile(event.dataTransfer.files?.[0]);
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera scanning is not supported in this browser. Choose a photo instead.");
      setStatus("error");
      return;
    }
    setError("");
    setScanProgress(8);
    setStatus("camera");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 1280 } },
      });
      streamRef.current = stream;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const video = videoRef.current;
      if (!video) throw new Error("The camera preview could not start.");
      video.srcObject = stream;
      await video.play();
      const run = ++scanRunRef.current;
      let stableFrames = 0;
      setScanProgress(20);

      const scan = async () => {
        if (run !== scanRunRef.current || !streamRef.current) return;
        try {
          const landmarks = await detectLandmarks(video);
          const faceWidth = Math.abs(landmarks[454].x - landmarks[234].x);
          const centerX = (landmarks[1].x + landmarks[152].x) / 2;
          const centerY = (landmarks[10].y + landmarks[152].y) / 2;
          const wellFramed =
            faceWidth > 0.24 && centerX > 0.28 && centerX < 0.72 && centerY > 0.3 && centerY < 0.72;
          stableFrames = wellFramed ? stableFrames + 1 : 0;
          setScanProgress(wellFramed ? 42 + stableFrames * 19 : 32);

          if (stableFrames >= 3) {
            const dimensions = fitDimensions(video.videoWidth, video.videoHeight, 1600);
            const source = document.createElement("canvas");
            source.width = dimensions.width;
            source.height = dimensions.height;
            const context = source.getContext("2d", { alpha: false });
            if (!context) throw new Error("Canvas is unavailable in this browser.");
            context.translate(source.width, 0);
            context.scale(-1, 1);
            context.drawImage(video, 0, 0, source.width, source.height);
            const mirrored = landmarks.map((point) => ({ ...point, x: 1 - point.x }));
            sourceRef.current = source;
            landmarksRef.current = mirrored;
            setAnalysis(analyzeFace(mirrored));
            setFileName("camera-scan.jpg");
            setShowOriginal(false);
            setScanProgress(100);
            stopCamera();
            setStatus("ready");
            window.setTimeout(paint, 0);
            return;
          }
        } catch (cause) {
          const message = cause instanceof Error ? cause.message : "";
          if (!message.includes("clear face")) {
            stopCamera();
            setError("The face scanner could not start. Refresh the page or choose a photo.");
            setStatus("error");
            return;
          }
          stableFrames = 0;
          setScanProgress(24);
        }
        window.setTimeout(scan, 240);
      };
      void scan();
    } catch (cause) {
      stopCamera();
      const denied = cause instanceof DOMException && (cause.name === "NotAllowedError" || cause.name === "SecurityError");
      setError(denied ? "Camera access was blocked. Allow camera access or choose a photo." : "The camera could not start. Choose a photo instead.");
      setStatus("error");
    }
  };

  const download = () => {
    const source = sourceRef.current;
    const landmarks = landmarksRef.current;
    if (!source || !landmarks) return;
    const canvas = morphImage(source, landmarks, getPreset(presetId), strength).canvas;
    const link = document.createElement("a");
    const base = fileName.replace(/\.[^.]+$/, "") || "portrait";
    link.download = `${base}-${presetId}.png`;
    link.href = canvas.toDataURL("image/png", 1);
    link.click();
  };

  const reset = () => {
    setStatus("empty");
    setError("");
    setFileName("");
    setAnalysis(null);
    setScanProgress(0);
    stopCamera();
    sourceRef.current = null;
    landmarksRef.current = null;
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Harmonia home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>HARMONIA</span>
        </a>
        <div className="privacy-chip"><span /> On-device processing</div>
        <nav aria-label="Application links">
          <button className="nav-button" onClick={() => document.getElementById("how")?.scrollIntoView({ behavior: "smooth" })}>How it works</button>
          <button className="nav-button" onClick={() => fileInputRef.current?.click()}>New edit</button>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow"><span>PIXEL-ONLY MORPHING</span><span className="eyebrow-line" /></div>
        <h1>Scan. Measure.<br /><em>Harmonize.</em></h1>
        <p className="hero-copy">Your face is mapped against a reference blueprint for harmony, symmetry and dimorphism—then reshaped using only your original pixels.</p>
      </section>

      <section className={`studio ${status === "ready" ? "studio-active" : ""}`} aria-label="Portrait editor">
        <div className="editor-stage">
          {status === "camera" ? (
            <div className="camera-stage">
              <video ref={videoRef} className="camera-video" muted playsInline aria-label="Live camera face scan" />
              <div className="camera-frame" aria-hidden="true"><span /><span /><span /><span /></div>
              <div className="camera-status">
                <strong>{scanProgress < 25 ? "Starting private scanner" : scanProgress < 42 ? "Center your face" : "Hold still—face found"}</strong>
                <div className="scan-meter"><i style={{ width: `${scanProgress}%` }} /></div>
                <small>Capture happens automatically when your face is centered.</small>
              </div>
              <button className="camera-cancel" onClick={() => { stopCamera(); setStatus("empty"); }}>Cancel</button>
            </div>
          ) : status === "empty" || status === "error" ? (
            <div
              className={`dropzone ${dragging ? "is-dragging" : ""}`}
              onDragEnter={() => setDragging(true)}
              onDragLeave={() => setDragging(false)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
            >
              <div className="face-guide" aria-hidden="true">
                <span className="guide-corner tl" /><span className="guide-corner tr" />
                <span className="guide-corner bl" /><span className="guide-corner br" />
                <div className="guide-face"><i className="guide-eye left" /><i className="guide-eye right" /><i className="guide-nose" /><i className="guide-mouth" /></div>
              </div>
              <h2>{status === "error" ? "Try again" : "Scan your face"}</h2>
              <p>{error || "The camera maps and captures your face automatically. No upload required."}</p>
              <div className="capture-actions">
                <button className="primary-button" onClick={() => void startCamera()}><span aria-hidden="true">◎</span> Start camera</button>
                <button className="secondary-button" onClick={() => fileInputRef.current?.click()}>Choose photo instead</button>
              </div>
              <small>JPG, PNG or WEBP · Up to 20 MB · Never uploaded</small>
            </div>
          ) : (
            <div className="canvas-wrap">
              <canvas ref={resultRef} className={showOriginal ? "hidden-canvas" : ""} aria-label="Morphed portrait preview" />
              <canvas ref={originalRef} className={!showOriginal ? "hidden-canvas" : ""} aria-label="Original portrait preview" />
              {status === "loading" && (
                <div className="analyzing"><span className="scanner" /><strong>Mapping facial structure</strong><small>Finding safe reshape regions…</small></div>
              )}
              <div className="compare-control" aria-label="Before and after preview">
                <button className={showOriginal ? "active" : ""} onClick={() => setShowOriginal(true)}>Before</button>
                <button className={!showOriginal ? "active" : ""} onClick={() => setShowOriginal(false)}>After</button>
              </div>
            </div>
          )}
          {status === "loading" && (
            <div className="loading-panel"><span className="loading-orbit" /><strong>Analyzing your portrait</strong><small>The first run may take a few seconds.</small></div>
          )}
        </div>

        <aside className="control-panel">
          <div className="panel-heading">
            <div><span className="step-number">01</span><h2>Choose a direction</h2></div>
            {status === "ready" && <span className="ready-badge">Face mapped</span>}
          </div>
          <p className="mode-intro">Each direction uses your own measurements to create an adaptive morph plan.</p>
          <div className="preset-grid">
            {PRESETS.map((option) => (
              <button
                key={option.id}
                className={`preset-card ${presetId === option.id ? "selected" : ""}`}
                onClick={() => setPresetId(option.id)}
                aria-pressed={presetId === option.id}
              >
                <span className={`preset-glyph glyph-${option.id}`} aria-hidden="true"><i /><i /></span>
                <span><strong>{option.label}</strong><small>{option.description}</small></span>
                <b aria-hidden="true">{presetId === option.id ? "✓" : ""}</b>
              </button>
            ))}
          </div>

          <div className="divider" />
          <div className="strength-row">
            <div><span className="step-number">02</span><h3>Edit strength</h3></div>
            <output>{strength}%</output>
          </div>
          <input className="range" style={{ "--range": `${strength}%` } as CSSProperties} type="range" min="0" max="100" value={strength} onChange={(event) => setStrength(Number(event.target.value))} aria-label="Edit strength" />
          <div className="range-labels"><span>Subtle</span><span>Noticeable</span></div>

          {status === "ready" && (
            <div className="safety-card">
              <div><span className="safety-icon">◇</span><strong>{getPreset(presetId).label} applied</strong><small>{regions.join(" · ")}</small></div>
              <span>{movement.toFixed(1)} px max</span>
            </div>
          )}

          {status === "ready" && analysis && (
            <div className="blueprint-card">
              <div className="blueprint-heading"><strong>Blueprint comparison</strong><small>geometry, not a beauty score</small></div>
              {[
                ["Harmony", analysis.harmony],
                ["Symmetry", analysis.symmetry],
                ["Structure", analysis.structure],
              ].map(([label, value]) => (
                <div className="metric-row" key={String(label)}><span>{label}</span><i><b style={{ width: `${value}%` }} /></i><output>{value}</output></div>
              ))}
            </div>
          )}

          <div className="panel-actions">
            <button className="mesh-button" disabled={status !== "ready"} onClick={() => setShowMesh((value) => !value)} aria-pressed={showMesh}><span>⌘</span>{showMesh ? "Hide map" : "Show map"}</button>
            <button className="export-button" disabled={status !== "ready"} onClick={download}>Export PNG <span>↓</span></button>
          </div>
          {status === "ready" && <button className="start-over" onClick={reset}>Start over with a different photo</button>}
        </aside>
      </section>

      <section className="principles" id="how">
        <div className="principle-title"><span>BUILT DIFFERENT</span><h2>No generation.<br />Just careful geometry.</h2></div>
        <div className="principle-list">
          <article><span>01</span><div><h3>Your pixels stay yours</h3><p>The image is processed inside your browser. Nothing is sent to a server or retained.</p></div></article>
          <article><span>02</span><div><h3>Identity stays intact</h3><p>A bounded mesh moves existing pixels only. No new skin, hair, shadows or background.</p></div></article>
          <article><span>03</span><div><h3>Guardrails by design</h3><p>Movement is tapered at the face boundary and capped to keep the result believable.</p></div></article>
        </div>
      </section>

      <footer><span>HARMONIA / V1</span><p>A creative photo-editing tool. Results are subjective—not a measure of your worth.</p><span>PRIVATE BY DEFAULT</span></footer>
      <input ref={fileInputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
    </main>
  );
}
