import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";

import { createCanvas } from "@napi-rs/canvas";
import { analyzeFace, createMorphPlan, DEFAULT_DIRECTION_MIX } from "../lib/face-intelligence.ts";
import { CANONICAL_FACE_LANDMARKS } from "./fixtures/canonical-face-landmarks.mjs";

const moduleHooks = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "./face-intelligence") return nextResolve("./face-intelligence.ts", context);
    return nextResolve(specifier, context);
  },
});
const { initializeFaceTopology, morphImage } = await import("../lib/morph.ts");
moduleHooks.deregister();
const { FaceLandmarker } = await import("@mediapipe/tasks-vision");
initializeFaceTopology(FaceLandmarker.FACE_LANDMARKS_TESSELATION);

const WIDTH = 800;
const HEIGHT = 960;
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
  127, 162, 21, 54, 103, 67, 109,
];

function observation() {
  return {
    landmarks: CANONICAL_FACE_LANDMARKS.map((point) => ({ ...point })),
    // A naturally closed mouth can produce this coefficient. Geometry should
    // corroborate it before the lower face is locked.
    blendshapes: { mouthPucker: 0.54 },
    transformationMatrix: {
      rows: 4,
      columns: 4,
      data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    },
  };
}

function texturedPortrait() {
  const canvas = createCanvas(WIDTH, HEIGHT);
  const context = canvas.getContext("2d");
  context.fillStyle = "#17201b";
  context.fillRect(0, 0, WIDTH, HEIGHT);
  for (let y = 0; y < HEIGHT; y += 8) {
    for (let x = 0; x < WIDTH; x += 8) {
      const red = (x * 3 + y) % 210 + 25;
      const green = (x + y * 2) % 190 + 35;
      const blue = (x * 2 + y * 3) % 180 + 45;
      context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      context.fillRect(x, y, 8, 8);
    }
  }
  return canvas;
}

function outerBoundaryBounds(landmarks) {
  const oval = FACE_OVAL.map((index) => ({
    x: landmarks[index].x * WIDTH,
    y: landmarks[index].y * HEIGHT,
  }));
  const minX = Math.min(...oval.map((point) => point.x));
  const maxX = Math.max(...oval.map((point) => point.x));
  const minY = Math.min(...oval.map((point) => point.y));
  const maxY = Math.max(...oval.map((point) => point.y));
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const boundary = oval.map((point) => ({
    x: centerX + (point.x - centerX) * 1.18,
    y: centerY + (point.y - centerY) * 1.16,
  }));
  return {
    minX: Math.floor(Math.min(...boundary.map((point) => point.x))) - 2,
    maxX: Math.ceil(Math.max(...boundary.map((point) => point.x))) + 2,
    minY: Math.floor(Math.min(...boundary.map((point) => point.y))) - 2,
    maxY: Math.ceil(Math.max(...boundary.map((point) => point.y))) + 2,
  };
}

test("a supported frontal plan changes real raster pixels monotonically without materially altering distant background", () => {
  const face = observation();
  const analysis = analyzeFace(face, WIDTH, HEIGHT, {
    qualityConfidence: 1,
    temporalStability: 1,
    resolutionSupport: 1,
    exposureSupport: 1,
  });
  const plan = createMorphPlan(analysis, DEFAULT_DIRECTION_MIX);
  const source = texturedPortrait();
  const priorDocument = globalThis.document;
  globalThis.document = {
    createElement(tag) {
      assert.equal(tag, "canvas");
      return createCanvas(1, 1);
    },
  };
  let rendered;
  let mediumRendered;
  try {
    mediumRendered = morphImage(source, face.landmarks, plan, 80);
    rendered = morphImage(source, face.landmarks, plan, 100);
  } finally {
    globalThis.document = priorDocument;
  }

  assert.notEqual(rendered.safetyStatus, "identity");
  assert.ok(rendered.movedRegions.includes("Jaw"));
  assert.ok(rendered.movedRegions.includes("Chin"));
  assert.ok(rendered.maxMovementPx >= 8, `expected an obvious lower-face morph, received ${rendered.maxMovementPx.toFixed(2)}px`);

  const baseline = createCanvas(WIDTH, HEIGHT);
  const baselineContext = baseline.getContext("2d", { alpha: false });
  baselineContext.imageSmoothingEnabled = true;
  baselineContext.imageSmoothingQuality = "high";
  baselineContext.drawImage(source, 0, 0);
  const before = baselineContext.getImageData(0, 0, WIDTH, HEIGHT).data;
  const after = rendered.canvas.getContext("2d").getImageData(0, 0, WIDTH, HEIGHT).data;
  const mediumAfter = mediumRendered.canvas.getContext("2d").getImageData(0, 0, WIDTH, HEIGHT).data;
  const bounds = outerBoundaryBounds(face.landmarks);
  let facePixels = 0;
  let changedFacePixels = 0;
  let changedMediumFacePixels = 0;
  let changedDistantPixels = 0;
  let firstDistantDiff = null;
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const offset = (y * WIDTH + x) * 4;
      const delta = Math.abs(before[offset] - after[offset]) +
        Math.abs(before[offset + 1] - after[offset + 1]) +
        Math.abs(before[offset + 2] - after[offset + 2]);
      const insideBoundaryBox = x >= bounds.minX && x <= bounds.maxX && y >= bounds.minY && y <= bounds.maxY;
      if (insideBoundaryBox) {
        facePixels += 1;
        if (delta >= 18) changedFacePixels += 1;
        const mediumDelta = Math.abs(before[offset] - mediumAfter[offset]) +
          Math.abs(before[offset + 1] - mediumAfter[offset + 1]) +
          Math.abs(before[offset + 2] - mediumAfter[offset + 2]);
        if (mediumDelta >= 18) changedMediumFacePixels += 1;
      } else if (delta >= 18) {
        changedDistantPixels += 1;
        firstDistantDiff ??= { x, y, delta, before: [...before.slice(offset, offset + 4)], after: [...after.slice(offset, offset + 4)] };
      }
    }
  }
  assert.equal(changedDistantPixels, 0, `pixels outside the fixed face boundary must not be geometrically resampled (${JSON.stringify(firstDistantDiff)})`);
  assert.ok(
    changedFacePixels / facePixels >= 0.08,
    `real raster output changed only ${((changedFacePixels / facePixels) * 100).toFixed(2)}% of the bounded face area`,
  );
  assert.ok(
    changedFacePixels >= changedMediumFacePixels,
    `maximum strength must not change fewer face pixels than 80% (${changedFacePixels} < ${changedMediumFacePixels})`,
  );
});
