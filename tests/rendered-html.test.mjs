import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the complete Harmonia studio", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Harmonia — Pixel-only portrait morphing<\/title>/i);
  assert.match(html, /Scan your face/);
  assert.match(html, /Harmony/);
  assert.match(html, /Symmetry/);
  assert.match(html, /Dimorphism/);
  assert.match(html, /Start camera/);
  assert.doesNotMatch(html, /Chadlite|Refined|Angularity/);
  assert.match(html, /Export PNG/);
  assert.match(html, /Never uploaded/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("ships the local landmark model and pixel warp engine", async () => {
  const [engine, page] = await Promise.all([
    readFile(new URL("../lib/morph.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    access(new URL("../public/models/face_landmarker.task", import.meta.url)),
    access(new URL("../public/mediapipe/wasm/vision_wasm_internal.wasm", import.meta.url)),
  ]);
  assert.match(engine, /FaceLandmarker/);
  assert.match(engine, /FilesetResolver\.forVisionTasks\(\s*publicAsset\("mediapipe\/wasm"\)/);
  assert.match(engine, /id: "harmony"/);
  assert.match(engine, /id: "symmetry"/);
  assert.match(engine, /id: "dimorphism"/);
  assert.doesNotMatch(engine, /id: "chadlite"|id: "refined"|id: "angularity"/);
  assert.match(engine, /analyzeFace/);
  assert.match(engine, /drawTriangle/);
  assert.match(engine, /buildFaceMesh/);
  assert.match(engine, /FACE_LANDMARKS_TESSELATION/);
  assert.match(engine, /regularizeDisplacements/);
  assert.match(engine, /fixed ring follows the actual face oval/i);
  assert.match(engine, /planIsSafe/);
  assert.match(engine, /sourceArea \* targetArea <= 0/);
  assert.match(engine, /maximumStretch \/ Math\.max\(minimumStretch/);
  assert.match(engine, /candidateScale/);
  assert.match(page, /morphImage/);
  assert.match(page, /getUserMedia/);
  assert.match(page, /freezeMirroredFrame/);
  assert.match(page, /assessFrame/);
  assert.match(page, /laplacianSquaredTotal/);
  assert.match(page, /mouthWidth/);
  assert.match(page, /stableFrames >= 7/);
  assert.match(page, /sourceRef\.current = chosen\.canvas/);
  assert.doesNotMatch(page, /detectLandmarks\(video\)/);
  assert.match(page, /toDataURL\("image\/png"/);
  assert.doesNotMatch(page, /fetch\(|XMLHttpRequest|FormData/);
});
