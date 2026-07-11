import type { FaceLandmarker } from "@mediapipe/tasks-vision";

export type Point = { x: number; y: number; z?: number };

export type MorphPreset = {
  id: string;
  label: string;
  description: string;
  jaw: number;
  chin: number;
  nose: number;
  lips: number;
  brows: number;
  symmetry: number;
};

export type MorphResult = {
  canvas: HTMLCanvasElement;
  movedRegions: string[];
  maxMovementPx: number;
};

export const PRESETS: MorphPreset[] = [
  {
    id: "harmony",
    label: "Natural harmony",
    description: "Balanced proportions with the lightest touch.",
    jaw: 0.3,
    chin: 0.2,
    nose: 0.25,
    lips: 0.15,
    brows: 0.18,
    symmetry: 0.3,
  },
  {
    id: "chadlite",
    label: "Chadlite",
    description: "A squarer jaw, firmer chin and straighter brow line.",
    jaw: 0.95,
    chin: 0.7,
    nose: 0.25,
    lips: 0.05,
    brows: 0.45,
    symmetry: 0.3,
  },
  {
    id: "refined",
    label: "Refined",
    description: "Softer taper, subtle lip balance and a narrower nose.",
    jaw: -0.38,
    chin: 0.28,
    nose: 0.65,
    lips: 0.5,
    brows: 0.36,
    symmetry: 0.42,
  },
  {
    id: "symmetry",
    label: "Symmetry",
    description: "Gently evens paired features without changing character.",
    jaw: 0,
    chin: 0,
    nose: 0.1,
    lips: 0.08,
    brows: 0.08,
    symmetry: 0.88,
  },
];

const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
  127, 162, 21, 54, 103, 67, 109,
];

let landmarkerPromise: Promise<FaceLandmarker> | null = null;

export function getPreset(id: string) {
  return PRESETS.find((preset) => preset.id === id) ?? PRESETS[0];
}

export function presetFromPrompt(prompt: string): MorphPreset {
  const text = prompt.toLowerCase();
  if (/chad|masculine|square|strong jaw|model man/.test(text)) {
    return getPreset("chadlite");
  }
  if (/pretty|prettier|refin|feminine|soft|delicate|slim/.test(text)) {
    return getPreset("refined");
  }
  if (/symmetr|even|balanced sides/.test(text)) {
    return getPreset("symmetry");
  }
  return getPreset("harmony");
}

export async function createFaceLandmarker(): Promise<FaceLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import(
        "@mediapipe/tasks-vision"
      );
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
      );
      const common = {
        baseOptions: {
          modelAssetPath: "/models/face_landmarker.task",
        },
        runningMode: "IMAGE" as const,
        numFaces: 1,
        minFaceDetectionConfidence: 0.55,
        minFacePresenceConfidence: 0.55,
        minTrackingConfidence: 0.55,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: true,
      };
      try {
        return await FaceLandmarker.createFromOptions(vision, {
          ...common,
          baseOptions: { ...common.baseOptions, delegate: "GPU" },
        });
      } catch {
        return FaceLandmarker.createFromOptions(vision, common);
      }
    })();
  }
  return landmarkerPromise;
}

export async function detectLandmarks(image: HTMLImageElement | HTMLCanvasElement): Promise<Point[]> {
  const landmarker = await createFaceLandmarker();
  const result = landmarker.detect(image);
  const face = result.faceLandmarks?.[0];
  if (!face || face.length < 468) {
    throw new Error("We couldn’t find one clear face in this photo.");
  }
  return face.map((point) => ({ x: point.x, y: point.y, z: point.z }));
}

type Control = { source: Point; dx: number; dy: number; radius: number; region: string };

const pair = (left: number, right: number): [number, number] => [left, right];
const SYMMETRY_PAIRS = [
  pair(33, 263),
  pair(133, 362),
  pair(70, 300),
  pair(105, 334),
  pair(61, 291),
  pair(78, 308),
  pair(234, 454),
  pair(172, 397),
  pair(136, 365),
];

function buildControls(
  points: Point[],
  width: number,
  height: number,
  preset: MorphPreset,
  strength: number,
): Control[] {
  const controls: Control[] = [];
  const p = (index: number) => ({
    x: points[index].x * width,
    y: points[index].y * height,
  });
  const faceWidth = Math.abs(p(454).x - p(234).x);
  const faceHeight = Math.abs(p(152).y - p(10).y);
  const scale = Math.min(faceWidth, faceHeight) * 0.022 * strength;
  const add = (
    index: number,
    dx: number,
    dy: number,
    radius: number,
    region: string,
  ) => controls.push({ source: p(index), dx, dy, radius, region });

  // A positive jaw value creates a firmer lower third. A negative value tapers it.
  const jaw = preset.jaw * scale;
  [234, 172, 136, 150].forEach((index, order) =>
    add(index, -jaw * (0.45 + order * 0.15), -Math.abs(jaw) * 0.08, faceWidth * 0.16, "Jaw"),
  );
  [454, 397, 365, 379].forEach((index, order) =>
    add(index, jaw * (0.45 + order * 0.15), -Math.abs(jaw) * 0.08, faceWidth * 0.16, "Jaw"),
  );

  const chin = preset.chin * scale;
  add(152, 0, -chin * 0.65, faceWidth * 0.18, "Chin");
  add(148, -chin * 0.16, -chin * 0.32, faceWidth * 0.14, "Chin");
  add(377, chin * 0.16, -chin * 0.32, faceWidth * 0.14, "Chin");

  const nose = preset.nose * scale;
  add(49, nose * 0.48, 0, faceWidth * 0.1, "Nose");
  add(279, -nose * 0.48, 0, faceWidth * 0.1, "Nose");
  add(1, 0, -nose * 0.1, faceWidth * 0.09, "Nose");

  const lips = preset.lips * scale;
  add(61, -lips * 0.28, 0, faceWidth * 0.1, "Lips");
  add(291, lips * 0.28, 0, faceWidth * 0.1, "Lips");
  add(13, 0, -lips * 0.14, faceWidth * 0.08, "Lips");
  add(14, 0, lips * 0.12, faceWidth * 0.08, "Lips");

  const brow = preset.brows * scale;
  add(70, 0, -brow * 0.36, faceWidth * 0.11, "Brows");
  add(300, 0, -brow * 0.36, faceWidth * 0.11, "Brows");
  add(105, 0, -brow * 0.2, faceWidth * 0.1, "Brows");
  add(334, 0, -brow * 0.2, faceWidth * 0.1, "Brows");

  const symmetry = preset.symmetry * strength * 0.38;
  const centerX = (p(10).x + p(152).x) / 2;
  for (const [li, ri] of SYMMETRY_PAIRS) {
    const left = p(li);
    const right = p(ri);
    const targetY = (left.y + right.y) / 2;
    const halfSpan = (Math.abs(centerX - left.x) + Math.abs(right.x - centerX)) / 2;
    controls.push({
      source: left,
      dx: (centerX - halfSpan - left.x) * symmetry,
      dy: (targetY - left.y) * symmetry,
      radius: faceWidth * 0.09,
      region: "Symmetry",
    });
    controls.push({
      source: right,
      dx: (centerX + halfSpan - right.x) * symmetry,
      dy: (targetY - right.y) * symmetry,
      radius: faceWidth * 0.09,
      region: "Symmetry",
    });
  }

  return controls.filter((control) => Math.hypot(control.dx, control.dy) > 0.08);
}

function drawTriangle(
  ctx: CanvasRenderingContext2D,
  image: CanvasImageSource,
  source: Point[],
  target: Point[],
) {
  const [s0, s1, s2] = source;
  const [d0, d1, d2] = target;
  const determinant =
    s0.x * (s1.y - s2.y) + s1.x * (s2.y - s0.y) + s2.x * (s0.y - s1.y);
  if (Math.abs(determinant) < 0.001) return;

  const a =
    (d0.x * (s1.y - s2.y) + d1.x * (s2.y - s0.y) + d2.x * (s0.y - s1.y)) /
    determinant;
  const c =
    (d0.x * (s2.x - s1.x) + d1.x * (s0.x - s2.x) + d2.x * (s1.x - s0.x)) /
    determinant;
  const e =
    (d0.x * (s1.x * s2.y - s2.x * s1.y) +
      d1.x * (s2.x * s0.y - s0.x * s2.y) +
      d2.x * (s0.x * s1.y - s1.x * s0.y)) /
    determinant;
  const b =
    (d0.y * (s1.y - s2.y) + d1.y * (s2.y - s0.y) + d2.y * (s0.y - s1.y)) /
    determinant;
  const d =
    (d0.y * (s2.x - s1.x) + d1.y * (s0.x - s2.x) + d2.y * (s1.x - s0.x)) /
    determinant;
  const f =
    (d0.y * (s1.x * s2.y - s2.x * s1.y) +
      d1.y * (s2.x * s0.y - s0.x * s2.y) +
      d2.y * (s0.x * s1.y - s1.x * s0.y)) /
    determinant;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(d0.x, d0.y);
  ctx.lineTo(d1.x, d1.y);
  ctx.lineTo(d2.x, d2.y);
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(a, b, c, d, e, f);
  ctx.drawImage(image, 0, 0);
  ctx.restore();
}

export function morphImage(
  sourceCanvas: HTMLCanvasElement,
  landmarks: Point[],
  preset: MorphPreset,
  strengthPercent: number,
): MorphResult {
  const { width, height } = sourceCanvas;
  const output = document.createElement("canvas");
  output.width = width;
  output.height = height;
  const ctx = output.getContext("2d", { alpha: false });
  if (!ctx) throw new Error("Canvas rendering is unavailable in this browser.");

  const strength = Math.max(0, Math.min(1, strengthPercent / 100));
  if (strength === 0) {
    ctx.drawImage(sourceCanvas, 0, 0);
    return { canvas: output, movedRegions: [], maxMovementPx: 0 };
  }

  const controls = buildControls(landmarks, width, height, preset, strength);
  const oval = FACE_OVAL.map((index) => ({
    x: landmarks[index].x * width,
    y: landmarks[index].y * height,
  }));
  const minX = Math.min(...oval.map((point) => point.x));
  const maxX = Math.max(...oval.map((point) => point.x));
  const minY = Math.min(...oval.map((point) => point.y));
  const maxY = Math.max(...oval.map((point) => point.y));
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const radiusX = (maxX - minX) * 0.59;
  const radiusY = (maxY - minY) * 0.57;

  const columns = 34;
  const rows = Math.max(24, Math.round(columns * (height / width)));
  const sourcePoints: Point[][] = [];
  const targetPoints: Point[][] = [];

  for (let row = 0; row <= rows; row += 1) {
    const sourceRow: Point[] = [];
    const targetRow: Point[] = [];
    for (let column = 0; column <= columns; column += 1) {
      const source = { x: (column / columns) * width, y: (row / rows) * height };
      const ellipse =
        ((source.x - centerX) / radiusX) ** 2 + ((source.y - centerY) / radiusY) ** 2;
      const faceMask = Math.max(0, Math.min(1, (1.12 - ellipse) / 0.22));
      let sum = 0;
      let dx = 0;
      let dy = 0;
      for (const control of controls) {
        const distanceSquared =
          (source.x - control.source.x) ** 2 + (source.y - control.source.y) ** 2;
        const weight = Math.exp(-distanceSquared / (2 * control.radius ** 2));
        sum += weight;
        dx += control.dx * weight;
        dy += control.dy * weight;
      }
      const normalized = sum > 0 ? Math.min(1, sum) / Math.max(1, sum * 0.72) : 0;
      sourceRow.push(source);
      targetRow.push({
        x: source.x + dx * normalized * faceMask,
        y: source.y + dy * normalized * faceMask,
      });
    }
    sourcePoints.push(sourceRow);
    targetPoints.push(targetRow);
  }

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(sourceCanvas, 0, 0);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const s00 = sourcePoints[row][column];
      const s10 = sourcePoints[row][column + 1];
      const s01 = sourcePoints[row + 1][column];
      const s11 = sourcePoints[row + 1][column + 1];
      const d00 = targetPoints[row][column];
      const d10 = targetPoints[row][column + 1];
      const d01 = targetPoints[row + 1][column];
      const d11 = targetPoints[row + 1][column + 1];
      const displacement = Math.max(
        Math.hypot(d00.x - s00.x, d00.y - s00.y),
        Math.hypot(d10.x - s10.x, d10.y - s10.y),
        Math.hypot(d01.x - s01.x, d01.y - s01.y),
        Math.hypot(d11.x - s11.x, d11.y - s11.y),
      );
      if (displacement < 0.01) continue;
      drawTriangle(ctx, sourceCanvas, [s00, s10, s11], [d00, d10, d11]);
      drawTriangle(ctx, sourceCanvas, [s00, s11, s01], [d00, d11, d01]);
    }
  }

  return {
    canvas: output,
    movedRegions: [...new Set(controls.map((control) => control.region))],
    maxMovementPx: Math.max(...controls.map((control) => Math.hypot(control.dx, control.dy)), 0),
  };
}

export function drawLandmarkOverlay(
  canvas: HTMLCanvasElement,
  landmarks: Point[],
  color = "#c8ff45",
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.78;
  for (const index of [...FACE_OVAL, 33, 133, 263, 362, 61, 291, 1, 152]) {
    const point = landmarks[index];
    ctx.beginPath();
    ctx.arc(point.x * canvas.width, point.y * canvas.height, Math.max(1.4, canvas.width / 650), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
