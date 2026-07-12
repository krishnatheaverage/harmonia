"use client";

import { ChangeEvent, CSSProperties, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  detectLandmarks,
  drawLandmarkOverlay,
  getPreset,
  morphImage,
  PRESETS,
  type Point,
} from "../lib/morph";

type Status = "empty" | "loading" | "ready" | "error";

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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sourceRef = useRef<HTMLCanvasElement | null>(null);
  const landmarksRef = useRef<Point[] | null>(null);
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
        <h1>Harmony or<br /><em>angularity.</em></h1>
        <p className="hero-copy">Two focused edits. One balances facial relationships; the other adds definition. Both reshape only your original pixels.</p>
      </section>

      <section className={`studio ${status === "ready" ? "studio-active" : ""}`} aria-label="Portrait editor">
        <div className="editor-stage">
          {status === "empty" || status === "error" ? (
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
              <h2>{status === "error" ? "Try another portrait" : "Drop in a portrait"}</h2>
              <p>{error || "Frontal, three-quarter and clean profile photos work best."}</p>
              <button className="primary-button" onClick={() => fileInputRef.current?.click()}>
                <span aria-hidden="true">＋</span> Choose photo
              </button>
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
            <div><span className="step-number">01</span><h2>Choose the edit</h2></div>
            {status === "ready" && <span className="ready-badge">Face mapped</span>}
          </div>
          <p className="mode-intro">Select a single morph plan. You can switch modes instantly after your face is mapped.</p>
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
