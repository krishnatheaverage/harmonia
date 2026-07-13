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

type QualityIssue =
  | "none"
  | "framing"
  | "pixels"
  | "blur"
  | "dark"
  | "bright"
  | "yaw"
  | "roll"
  | "mouth";

type FrameAssessment = {
  ok: boolean;
  issue: QualityIssue;
  title: string;
  hint: string;
  sharpness: number;
};

const STABILITY_LANDMARKS = [10, 152, 234, 454, 33, 263, 1, 61, 291, 172, 397];

function freezeMirroredFrame(video: HTMLVideoElement) {
  const dimensions = fitDimensions(video.videoWidth, video.videoHeight, 1600);
  const canvas = document.createElement("canvas");
  canvas.width = dimensions.width;
  canvas.height = dimensions.height;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("Canvas is unavailable in this browser.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.translate(canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function faceCropMetrics(canvas: HTMLCanvasElement, landmarks: Point[]) {
  const xs = landmarks.map((point) => point.x * canvas.width);
  const ys = landmarks.map((point) => point.y * canvas.height);
  const minX = Math.max(0, Math.min(...xs));
  const maxX = Math.min(canvas.width, Math.max(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxY = Math.min(canvas.height, Math.max(...ys));
  const cropWidth = Math.max(2, maxX - minX);
  const cropHeight = Math.max(2, maxY - minY);
  const sampleWidth = 224;
  const sampleHeight = Math.max(160, Math.min(300, Math.round((cropHeight / cropWidth) * sampleWidth)));
  const sample = document.createElement("canvas");
  sample.width = sampleWidth;
  sample.height = sampleHeight;
  const context = sample.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!context) throw new Error("Canvas is unavailable in this browser.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(canvas, minX, minY, cropWidth, cropHeight, 0, 0, sampleWidth, sampleHeight);
  const pixels = context.getImageData(0, 0, sampleWidth, sampleHeight).data;
  const luminance = new Float32Array(sampleWidth * sampleHeight);
  let brightness = 0;
  let darkPixels = 0;
  let brightPixels = 0;

  for (let index = 0; index < luminance.length; index += 1) {
    const offset = index * 4;
    const value = pixels[offset] * 0.299 + pixels[offset + 1] * 0.587 + pixels[offset + 2] * 0.114;
    luminance[index] = value;
    brightness += value;
    if (value <= 20) darkPixels += 1;
    if (value >= 245) brightPixels += 1;
  }

  let laplacianTotal = 0;
  let laplacianSquaredTotal = 0;
  let laplacianCount = 0;
  for (let y = 1; y < sampleHeight - 1; y += 1) {
    for (let x = 1; x < sampleWidth - 1; x += 1) {
      const index = y * sampleWidth + x;
      const laplacian =
        luminance[index - 1] +
        luminance[index + 1] +
        luminance[index - sampleWidth] +
        luminance[index + sampleWidth] -
        4 * luminance[index];
      laplacianTotal += laplacian;
      laplacianSquaredTotal += laplacian * laplacian;
      laplacianCount += 1;
    }
  }
  const laplacianMean = laplacianTotal / Math.max(1, laplacianCount);
  const sharpness = laplacianSquaredTotal / Math.max(1, laplacianCount) - laplacianMean * laplacianMean;

  return {
    sharpness,
    brightness: brightness / luminance.length,
    darkFraction: darkPixels / luminance.length,
    brightFraction: brightPixels / luminance.length,
  };
}

function assessFrame(canvas: HTMLCanvasElement, landmarks: Point[], mode: "camera" | "file"): FrameAssessment {
  const faceWidth = Math.abs(landmarks[454].x - landmarks[234].x);
  const faceHeight = Math.abs(landmarks[152].y - landmarks[10].y);
  const faceWidthPx = faceWidth * canvas.width;
  const faceHeightPx = faceHeight * canvas.height;
  const minFaceX = Math.min(...landmarks.map((point) => point.x));
  const maxFaceX = Math.max(...landmarks.map((point) => point.x));
  const minFaceY = Math.min(...landmarks.map((point) => point.y));
  const maxFaceY = Math.max(...landmarks.map((point) => point.y));
  const centerX = (landmarks[234].x + landmarks[454].x) / 2;
  const centerY = (landmarks[10].y + landmarks[152].y) / 2;
  const eyeLeft = {
    x: (landmarks[33].x + landmarks[133].x) / 2,
    y: (landmarks[33].y + landmarks[133].y) / 2,
  };
  const eyeRight = {
    x: (landmarks[263].x + landmarks[362].x) / 2,
    y: (landmarks[263].y + landmarks[362].y) / 2,
  };
  const roll = Math.abs(Math.atan2((eyeRight.y - eyeLeft.y) * canvas.height, (eyeRight.x - eyeLeft.x) * canvas.width) * (180 / Math.PI));
  const noseX = landmarks[1].x;
  const leftNoseSpan = Math.abs(noseX - landmarks[234].x);
  const rightNoseSpan = Math.abs(landmarks[454].x - noseX);
  const yaw = Math.abs(leftNoseSpan - rightNoseSpan) / Math.max(0.001, leftNoseSpan + rightNoseSpan);
  const pixelDistance = (a: Point, b: Point) => Math.hypot((a.x - b.x) * canvas.width, (a.y - b.y) * canvas.height);
  const mouthWidthPx = pixelDistance(landmarks[61], landmarks[291]);
  const mouthOpen = pixelDistance(landmarks[13], landmarks[14]) / Math.max(1, mouthWidthPx);
  const mouthToFace = mouthWidthPx / Math.max(1, faceWidthPx);
  const crop = faceCropMetrics(canvas, landmarks);
  const cameraMode = mode === "camera";

  if (cameraMode && (faceWidth < 0.23 || faceHeight < 0.34)) {
    return { ok: false, issue: "framing", title: "Move a little closer", hint: "Keep your full face inside the guide.", sharpness: crop.sharpness };
  }
  if (cameraMode && (faceWidth > 0.72 || faceHeight > 0.84)) {
    return { ok: false, issue: "framing", title: "Move a little farther back", hint: "Leave a small border around your face.", sharpness: crop.sharpness };
  }
  if (cameraMode && (centerX < 0.29 || centerX > 0.71 || centerY < 0.3 || centerY > 0.7)) {
    return { ok: false, issue: "framing", title: "Center your face", hint: "Align your eyes and chin inside the guide.", sharpness: crop.sharpness };
  }
  if (!cameraMode && (minFaceX < 0.015 || maxFaceX > 0.985 || minFaceY < 0.015 || maxFaceY > 0.985)) {
    return { ok: false, issue: "framing", title: "Face is too close to the edge", hint: "Choose a portrait with a small clear border around the face.", sharpness: crop.sharpness };
  }
  const minimumFaceWidth = cameraMode ? Math.min(240, canvas.width * 0.28) : 140;
  const minimumFaceHeight = cameraMode ? Math.min(300, canvas.height * 0.38) : 180;
  if (faceWidthPx < minimumFaceWidth || faceHeightPx < minimumFaceHeight) {
    return { ok: false, issue: "pixels", title: "Face is too small", hint: cameraMode ? "Move closer so the scan has enough detail." : "Choose a higher-resolution portrait where the face is larger.", sharpness: crop.sharpness };
  }
  if (yaw > (cameraMode ? 0.16 : 0.22)) {
    return { ok: false, issue: "yaw", title: "Look straight at the camera", hint: "Turn until both sides of your face are evenly visible.", sharpness: crop.sharpness };
  }
  if (roll > (cameraMode ? 8 : 15)) {
    return { ok: false, issue: "roll", title: "Level your head", hint: "Keep the line between your eyes horizontal.", sharpness: crop.sharpness };
  }
  if (mouthOpen > (cameraMode ? 0.17 : 0.26) || mouthToFace < (cameraMode ? 0.25 : 0.22)) {
    return { ok: false, issue: "mouth", title: "Relax your mouth", hint: "Rest your lips naturally—do not open or purse them.", sharpness: crop.sharpness };
  }
  if (crop.brightness < (cameraMode ? 48 : 36) || crop.darkFraction > (cameraMode ? 0.22 : 0.34)) {
    return { ok: false, issue: "dark", title: "Add light in front of you", hint: "Avoid a bright window behind your head.", sharpness: crop.sharpness };
  }
  if (crop.brightness > (cameraMode ? 210 : 224) || crop.brightFraction > (cameraMode ? 0.2 : 0.34)) {
    return { ok: false, issue: "bright", title: "Lighting is too bright", hint: "Step away from direct light so facial detail is visible.", sharpness: crop.sharpness };
  }
  if (crop.sharpness < (cameraMode ? 32 : 24)) {
    return { ok: false, issue: "blur", title: "Image is too blurry", hint: cameraMode ? "Wipe the lens and hold the phone still." : "Choose a sharper, in-focus portrait.", sharpness: crop.sharpness };
  }
  return { ok: true, issue: "none", title: "Face quality looks good", hint: "Hold still while we confirm seven clear frames.", sharpness: crop.sharpness };
}

function landmarkMotion(previous: Point[], current: Point[], width: number, height: number) {
  const previousFaceWidth = Math.abs(previous[454].x - previous[234].x) * width;
  const currentFaceWidth = Math.abs(current[454].x - current[234].x) * width;
  const faceWidth = Math.max(1, (previousFaceWidth + currentFaceWidth) / 2);
  const squaredMotion = STABILITY_LANDMARKS.reduce((total, index) => {
    const distance = Math.hypot(
      (previous[index].x - current[index].x) * width,
      (previous[index].y - current[index].y) * height,
    ) / faceWidth;
    return total + distance * distance;
  }, 0);
  return Math.sqrt(squaredMotion / STABILITY_LANDMARKS.length);
}

function fileQualityError(assessment: FrameAssessment) {
  switch (assessment.issue) {
    case "framing": return "Choose a portrait with the full face centered and a small clear border around it.";
    case "blur": return "This portrait is too blurry for a safe reshape. Choose a sharper, in-focus photo.";
    case "pixels": return "The face is too small in this portrait. Choose a higher-resolution, closer photo.";
    case "dark": return "This portrait is too dark to map safely. Choose one with soft light on the face.";
    case "bright": return "This portrait is overexposed. Choose one where facial detail is visible.";
    case "yaw": return "Choose a frontal or moderate three-quarter portrait for a reliable reshape.";
    case "roll": return "Choose a portrait with the head more level.";
    case "mouth": return "Choose a portrait with a neutral expression or mild closed-mouth smile.";
    default: return "Choose a clear, well-lit portrait with one unobstructed face.";
  }
}

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
  const [strength, setStrength] = useState(35);
  const [showOriginal, setShowOriginal] = useState(false);
  const [showMesh, setShowMesh] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [regions, setRegions] = useState<string[]>([]);
  const [movement, setMovement] = useState(0);
  const [safetyScale, setSafetyScale] = useState(1);
  const [safetyStatus, setSafetyStatus] = useState<"passed" | "weakened" | "identity">("passed");
  const [analysis, setAnalysis] = useState<FaceAnalysis | null>(null);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanMessage, setScanMessage] = useState("Starting private scanner");
  const [scanHint, setScanHint] = useState("Keep one clear face inside the guide.");

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
    setSafetyScale(transformed.safetyScale);
    setSafetyStatus(transformed.safetyStatus);
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
      const assessment = assessFrame(source, landmarks, "file");
      if (!assessment.ok) throw new Error(fileQualityError(assessment));
      sourceRef.current = source;
      landmarksRef.current = landmarks;
      setAnalysis(analyzeFace(landmarks, source.width, source.height));
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
    setScanMessage("Starting private scanner");
    setScanHint("Keep one clear face inside the guide.");
    setStatus("camera");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30, max: 30 },
        },
      });
      streamRef.current = stream;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const video = videoRef.current;
      if (!video) throw new Error("The camera preview could not start.");
      video.srcObject = stream;
      await video.play();
      const run = ++scanRunRef.current;
      let stableFrames = 0;
      let previousLandmarks: Point[] | null = null;
      let bestFrame: { canvas: HTMLCanvasElement; landmarks: Point[]; sharpness: number } | null = null;
      setScanProgress(20);
      setScanMessage("Finding your face");
      setScanHint("Center your face and look straight ahead.");

      const scan = async () => {
        if (run !== scanRunRef.current || !streamRef.current) return;
        try {
          if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
            setScanMessage("Warming up the camera");
            setScanHint("The first clear frame can take a moment.");
            setScanProgress(18);
            window.setTimeout(scan, 180);
            return;
          }

          // Detection runs on this frozen, mirrored canvas so the chosen pixels
          // and landmarks are guaranteed to come from the exact same frame.
          const frame = freezeMirroredFrame(video);
          const landmarks = await detectLandmarks(frame);
          const assessment = assessFrame(frame, landmarks, "camera");

          if (!assessment.ok) {
            stableFrames = 0;
            bestFrame = null;
            previousLandmarks = landmarks;
            setScanMessage(assessment.title);
            setScanHint(assessment.hint);
            setScanProgress(28);
          } else {
            const motion = previousLandmarks ? landmarkMotion(previousLandmarks, landmarks, frame.width, frame.height) : Number.POSITIVE_INFINITY;
            previousLandmarks = landmarks;
            if (motion > 0.012) {
              stableFrames = 0;
              bestFrame = null;
              setScanMessage("Hold still");
              setScanHint("Keep your head and phone steady while focus locks.");
              setScanProgress(38);
            } else {
              stableFrames += 1;
              if (!bestFrame || assessment.sharpness > bestFrame.sharpness) {
                bestFrame = { canvas: frame, landmarks, sharpness: assessment.sharpness };
              }
              setScanMessage(stableFrames < 7 ? `Checking clear frame ${stableFrames} of 7` : "Clear scan captured");
              setScanHint(stableFrames < 7 ? "Stay still and keep a relaxed, closed-mouth expression." : "Using the sharpest frame from this scan.");
              setScanProgress(Math.min(100, 38 + stableFrames * 9));
            }
          }

          if (stableFrames >= 7 && bestFrame) {
            const chosen = bestFrame;
            sourceRef.current = chosen.canvas;
            landmarksRef.current = chosen.landmarks;
            setAnalysis(analyzeFace(chosen.landmarks, chosen.canvas.width, chosen.canvas.height));
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
          bestFrame = null;
          previousLandmarks = null;
          setScanMessage("No clear face yet");
          setScanHint("Remove obstructions and keep one face inside the guide.");
          setScanProgress(22);
        }
        window.setTimeout(scan, 180);
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
    setSafetyScale(1);
    setSafetyStatus("passed");
    setScanProgress(0);
    setScanMessage("Starting private scanner");
    setScanHint("Keep one clear face inside the guide.");
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
        <p className="hero-copy">Your face is mapped as dense geometry. Harmony, symmetry and dimorphism edits adapt to your existing structure, then move only original pixels.</p>
      </section>

      <section className={`studio ${status === "ready" ? "studio-active" : ""}`} aria-label="Portrait editor">
        <div className="editor-stage">
          {status === "camera" ? (
            <div className="camera-stage">
              <video ref={videoRef} className="camera-video" muted playsInline aria-label="Live camera face scan" />
              <div className="camera-frame" aria-hidden="true"><span /><span /><span /><span /></div>
              <div className="camera-status">
                <strong>{scanMessage}</strong>
                <div className="scan-meter"><i style={{ width: `${scanProgress}%` }} /></div>
                <small>{scanHint}</small>
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
          <p className="mode-intro">Each direction uses pose-normalized measurements to create a conservative morph plan around your own structure.</p>
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
          <input className="range" style={{ "--range": `${(strength / 70) * 100}%` } as CSSProperties} type="range" min="0" max="70" value={strength} onChange={(event) => setStrength(Number(event.target.value))} aria-label="Edit strength" />
          <div className="range-labels"><span>Conservative</span><span>Safe maximum</span></div>

          {status === "ready" && (
            <div className="safety-card">
              <div>
                <span className="safety-icon">◇</span>
                <strong>{safetyStatus === "weakened" ? "Guardrail reduced this edit" : safetyStatus === "identity" ? "Original geometry retained" : `${getPreset(presetId).label} passed`}</strong>
                <small>{regions.length ? regions.join(" · ") : "No safe region needed movement"}</small>
              </div>
              <span>{Math.round(safetyScale * 100)}% plan · {movement.toFixed(1)} px max</span>
            </div>
          )}

          {status === "ready" && analysis && (
            <div className="blueprint-card">
              <div className="blueprint-heading"><strong>Geometric readout</strong><small>measured ratios, not a beauty score</small></div>
              {[
                ["Lower jaw", analysis.metrics.jawToFace * 100],
                ["Nose width", analysis.metrics.noseToFace * 100],
                ["Mouth width", analysis.metrics.mouthToFace * 100],
                ["Lower third", analysis.metrics.lowerThird * 100],
                ["Paired drift", analysis.metrics.pairedDeviation],
              ].map(([label, value]) => (
                <div className="metric-row" key={String(label)}><span>{label}</span><i><b style={{ width: `${Math.min(100, Number(value))}%` }} /></i><output>{Number(value).toFixed(1)}%</output></div>
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
          <article><span>03</span><div><h3>Guardrails by design</h3><p>A fixed face-boundary ring and triangle checks weaken any plan that would pinch, fold or disturb the background.</p></div></article>
        </div>
      </section>

      <footer><span>HARMONIA / V1</span><p>A creative photo-editing tool. Results are subjective—not a measure of your worth.</p><span>PRIVATE BY DEFAULT</span></footer>
      <input ref={fileInputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
    </main>
  );
}
