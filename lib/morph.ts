import type { FaceLandmarker } from "@mediapipe/tasks-vision";

export type Point = { x: number; y: number; z?: number };
type Triangle = [number, number, number];

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
  safetyScale: number;
  safetyStatus: "passed" | "weakened" | "identity";
};

export type FaceAnalysis = {
  metrics: {
    jawToFace: number;
    noseToFace: number;
    mouthToFace: number;
    lowerThird: number;
    pairedDeviation: number;
  };
};

export const PRESETS: MorphPreset[] = [
  {
    id: "harmony",
    label: "Harmony",
    description: "Gently balances lower-face proportions around your existing structure.",
    jaw: 0.22,
    chin: 0.16,
    nose: 0.12,
    lips: 0,
    brows: 0.08,
    symmetry: 0.34,
  },
  {
    id: "symmetry",
    label: "Symmetry",
    description: "Softens only measurable paired-feature drift, with a natural deadband.",
    jaw: 0,
    chin: 0,
    nose: 0,
    lips: 0,
    brows: 0,
    symmetry: 0.72,
  },
  {
    id: "dimorphism",
    label: "Dimorphism",
    description: "Adds conservative jaw and brow definition without changing identity.",
    jaw: 0.72,
    chin: 0.34,
    nose: 0.06,
    lips: 0,
    brows: 0.2,
    symmetry: 0.18,
  },
];

const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
  127, 162, 21, 54, 103, 67, 109,
];

let landmarkerPromise: Promise<FaceLandmarker> | null = null;
let canonicalFaceTriangles: Triangle[] | null = null;

const publicAsset = (path: string) => {
  const base =
    (import.meta as ImportMeta & { env?: { BASE_URL?: string } }).env?.BASE_URL ?? "/";
  return `${base.endsWith("/") ? base : `${base}/`}${path.replace(/^\//, "")}`;
};

export function getPreset(id: string) {
  return PRESETS.find((preset) => preset.id === id) ?? PRESETS[0];
}

export async function createFaceLandmarker(): Promise<FaceLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import(
        "@mediapipe/tasks-vision"
      );
      canonicalFaceTriangles ??= trianglesFromConnections(
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      );
      const vision = await FilesetResolver.forVisionTasks(
        publicAsset("mediapipe/wasm")
      );
      const common = {
        baseOptions: {
          modelAssetPath: publicAsset("models/face_landmarker.task"),
        },
        runningMode: "IMAGE" as const,
        numFaces: 1,
        minFaceDetectionConfidence: 0.65,
        minFacePresenceConfidence: 0.65,
        minTrackingConfidence: 0.65,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      };
      try {
        return await FaceLandmarker.createFromOptions(vision, {
          ...common,
          baseOptions: { ...common.baseOptions, delegate: "GPU" },
        });
      } catch {
        return FaceLandmarker.createFromOptions(vision, common);
      }
    })().catch((error) => {
      landmarkerPromise = null;
      throw error;
    });
  }
  return landmarkerPromise;
}

export async function detectLandmarks(image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): Promise<Point[]> {
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

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

export function analyzeFace(points: Point[], width = 1, height = 1): FaceAnalysis {
  const distance = (a: number, b: number) =>
    Math.hypot(
      (points[a].x - points[b].x) * width,
      (points[a].y - points[b].y) * height,
    );
  const faceWidth = Math.max(distance(234, 454), 0.001);
  const faceHeight = Math.max(distance(10, 152), 0.001);
  const jawToFace = distance(172, 397) / faceWidth;
  const noseToFace = distance(49, 279) / faceWidth;
  const mouthToFace = distance(61, 291) / faceWidth;
  const lowerThird = distance(2, 152) / faceHeight;
  const frame = createPoseFrame(points, width, height);
  const symmetryError =
    SYMMETRY_PAIRS.reduce((total, [left, right]) => {
      return total + Math.abs(frame.local(left).v - frame.local(right).v) / faceHeight;
    }, 0) / SYMMETRY_PAIRS.length;
  return {
    metrics: {
      jawToFace,
      noseToFace,
      mouthToFace,
      lowerThird,
      pairedDeviation: symmetryError * 100,
    },
  };
}

function createPoseFrame(points: Point[], width: number, height: number) {
  const point = (index: number) => ({
    x: points[index].x * width,
    y: points[index].y * height,
  });
  const leftEye = point(33);
  const rightEye = point(263);
  const eyeSpan = Math.max(Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y), 0.001);
  const ux = (rightEye.x - leftEye.x) / eyeSpan;
  const uy = (rightEye.y - leftEye.y) / eyeSpan;
  const vx = -uy;
  const vy = ux;
  const forehead = point(10);
  const chin = point(152);
  const origin = {
    x: (forehead.x + chin.x) / 2,
    y: (forehead.y + chin.y) / 2,
  };
  const local = (index: number) => {
    const current = point(index);
    const x = current.x - origin.x;
    const y = current.y - origin.y;
    return { u: x * ux + y * uy, v: x * vx + y * vy };
  };
  const delta = (du: number, dv: number) => ({
    dx: du * ux + dv * vx,
    dy: du * uy + dv * vy,
  });
  return { point, local, delta };
}

function buildControls(
  points: Point[],
  width: number,
  height: number,
  preset: MorphPreset,
  strength: number,
): Control[] {
  const controls: Control[] = [];
  const frame = createPoseFrame(points, width, height);
  const p = frame.point;
  const faceWidth = Math.max(
    Math.abs(frame.local(454).u - frame.local(234).u),
    1,
  );
  const faceHeight = Math.max(
    Math.abs(frame.local(152).v - frame.local(10).v),
    1,
  );
  const scale = Math.min(faceWidth, faceHeight) * 0.035 * strength;
  const analysis = analyzeFace(points, width, height);
  let jawFactor = preset.jaw;
  let chinFactor = preset.chin;
  let noseFactor = preset.nose;
  let lipFactor = preset.lips;
  let browFactor = preset.brows;
  if (preset.id === "harmony") {
    // These centers are conservative engineering references, not an attractiveness score.
    // Only a small fraction of a measured deviation is applied.
    jawFactor = clamp((0.67 - analysis.metrics.jawToFace) * 2.2, -0.25, 0.25);
    chinFactor = clamp((0.36 - analysis.metrics.lowerThird) * 1.5, -0.16, 0.16);
    noseFactor = clamp((analysis.metrics.noseToFace - 0.285) * 1.5, -0.12, 0.16);
    lipFactor = 0;
    browFactor = 0.08;
  } else if (preset.id === "dimorphism") {
    // Add definition along the existing lower contour; do not infer gender or replace shape.
    jawFactor = preset.jaw;
    chinFactor = preset.chin;
    noseFactor = preset.nose;
    lipFactor = 0;
    browFactor = preset.brows;
  }
  const movementLimit = faceWidth * 0.015;
  const add = (
    index: number,
    dx: number,
    dy: number,
    radius: number,
    region: string,
  ) => controls.push({
    source: p(index),
    dx: clamp(dx, -movementLimit, movementLimit),
    dy: clamp(dy, -movementLimit, movementLimit),
    radius,
    region,
  });
  const addLocal = (
    index: number,
    du: number,
    dv: number,
    radius: number,
    region: string,
  ) => {
    const movement = frame.delta(du, dv);
    add(index, movement.dx, movement.dy, radius, region);
  };

  // Move the lower contour as a tapered group. The cheek extremes (234/454) stay fixed.
  const jaw = jawFactor * scale;
  const jawWeights = [1, 0.88, 0.66, 0.42, 0.2, 0.06];
  [172, 136, 150, 149, 176, 148].forEach((index, order) =>
    addLocal(index, -jaw * jawWeights[order], 0, faceWidth * 0.075, "Jaw"),
  );
  [397, 365, 379, 378, 400, 377].forEach((index, order) =>
    addLocal(index, jaw * jawWeights[order], 0, faceWidth * 0.075, "Jaw"),
  );

  const chin = chinFactor * scale;
  addLocal(152, 0, chin * 0.55, faceWidth * 0.07, "Chin");
  addLocal(148, 0, chin * 0.2, faceWidth * 0.065, "Chin");
  addLocal(377, 0, chin * 0.2, faceWidth * 0.065, "Chin");

  const nose = noseFactor * scale;
  addLocal(49, nose * 0.22, 0, faceWidth * 0.05, "Nose");
  addLocal(279, -nose * 0.22, 0, faceWidth * 0.05, "Nose");

  const lipDelta = lipFactor * scale;
  if (Math.abs(lipDelta) > 0.001) {
    addLocal(61, -lipDelta * 0.12, 0, faceWidth * 0.05, "Lips");
    addLocal(291, lipDelta * 0.12, 0, faceWidth * 0.05, "Lips");
  }

  const brow = browFactor * scale;
  const browWeights = [0.7, 0.9, 1, 0.78, 0.5];
  [70, 63, 105, 66, 107].forEach((index, order) =>
    addLocal(index, 0, -brow * 0.16 * browWeights[order], faceWidth * 0.045, "Brows"),
  );
  [300, 293, 334, 296, 336].forEach((index, order) =>
    addLocal(index, 0, -brow * 0.16 * browWeights[order], faceWidth * 0.045, "Brows"),
  );

  const noseU = frame.local(1).u;
  const leftSpan = Math.abs(noseU - frame.local(234).u);
  const rightSpan = Math.abs(frame.local(454).u - noseU);
  const yawSignal = Math.abs(leftSpan - rightSpan) / Math.max(leftSpan + rightSpan, 1);
  const poseFade = clamp(1 - yawSignal / 0.18, 0, 1);
  const symmetry = preset.symmetry * strength * 0.24 * poseFade;
  const centerU = (frame.local(10).u + frame.local(152).u + frame.local(1).u) / 3;
  const deadband = faceHeight * 0.0035;
  for (const [li, ri] of SYMMETRY_PAIRS.slice(0, 4)) {
    const left = frame.local(li);
    const right = frame.local(ri);
    const difference = right.v - left.v;
    if (Math.abs(difference) <= deadband) continue;
    const correction = (Math.abs(difference) - deadband) * Math.sign(difference) * symmetry * 0.5;
    addLocal(li, 0, correction, faceWidth * 0.052, "Symmetry");
    addLocal(ri, 0, -correction, faceWidth * 0.052, "Symmetry");
  }

  const applyCenteredGroup = (indices: number[], groupCenterU: number, amount: number, radius: number) => {
    const rawShift = centerU - groupCenterU;
    if (Math.abs(rawShift) <= deadband) return;
    const shift = (Math.abs(rawShift) - deadband) * Math.sign(rawShift) * amount;
    indices.forEach((index) => addLocal(index, shift, 0, radius, "Symmetry"));
  };
  const lipLandmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415];
  applyCenteredGroup(
    lipLandmarks,
    (frame.local(61).u + frame.local(291).u) / 2,
    symmetry * 0.6,
    faceWidth * 0.035,
  );
  applyCenteredGroup(
    [1, 2, 49, 279, 98, 327],
    frame.local(1).u,
    symmetry * 0.42,
    faceWidth * 0.04,
  );
  applyCenteredGroup(
    [152, 148, 377],
    frame.local(152).u,
    symmetry * 0.36,
    faceWidth * 0.055,
  );

  return controls.filter((control) => Math.hypot(control.dx, control.dy) > 0.05);
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

  const targetCenter = {
    x: (d0.x + d1.x + d2.x) / 3,
    y: (d0.y + d1.y + d2.y) / 3,
  };
  const clipPoints = target.map((point) => {
    const x = point.x - targetCenter.x;
    const y = point.y - targetCenter.y;
    const length = Math.max(Math.hypot(x, y), 0.001);
    const overdraw = 0.18;
    return {
      x: point.x + (x / length) * overdraw,
      y: point.y + (y / length) * overdraw,
    };
  });

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(clipPoints[0].x, clipPoints[0].y);
  ctx.lineTo(clipPoints[1].x, clipPoints[1].y);
  ctx.lineTo(clipPoints[2].x, clipPoints[2].y);
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(a, b, c, d, e, f);
  ctx.drawImage(image, 0, 0);
  ctx.restore();
}

const signedArea = (a: Point, b: Point, c: Point) =>
  (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);

function trianglesFromConnections(
  connections: Array<{ start: number; end: number }>,
): Triangle[] {
  const triangles: Triangle[] = [];
  for (let index = 0; index + 2 < connections.length; index += 3) {
    const vertices = [
      ...new Set(
        connections
          .slice(index, index + 3)
          .flatMap((connection) => [connection.start, connection.end]),
      ),
    ];
    if (vertices.length === 3 && vertices.every((vertex) => vertex < 468)) {
      triangles.push(vertices as Triangle);
    }
  }
  return triangles;
}

function circumcircleContains(points: Point[], triangle: Triangle, point: Point) {
  const [a, b, c] = triangle.map((index) => points[index]) as [Point, Point, Point];
  const denominator =
    2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
  if (Math.abs(denominator) < 1e-7) return false;
  const a2 = a.x * a.x + a.y * a.y;
  const b2 = b.x * b.x + b.y * b.y;
  const c2 = c.x * c.x + c.y * c.y;
  const ux =
    (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) /
    denominator;
  const uy =
    (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) /
    denominator;
  const radiusSquared = (ux - a.x) ** 2 + (uy - a.y) ** 2;
  const distanceSquared = (ux - point.x) ** 2 + (uy - point.y) ** 2;
  return distanceSquared <= radiusSquared + 0.01;
}

function delaunay(input: Point[]): Triangle[] {
  if (input.length < 3) return [];
  const minX = Math.min(...input.map((point) => point.x));
  const maxX = Math.max(...input.map((point) => point.x));
  const minY = Math.min(...input.map((point) => point.y));
  const maxY = Math.max(...input.map((point) => point.y));
  const size = Math.max(maxX - minX, maxY - minY, 1);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const points = [
    ...input,
    { x: centerX - size * 20, y: centerY - size * 16 },
    { x: centerX, y: centerY + size * 20 },
    { x: centerX + size * 20, y: centerY - size * 16 },
  ];
  const count = input.length;
  let triangles: Triangle[] = [[count, count + 1, count + 2]];

  for (let index = 0; index < count; index += 1) {
    const bad = triangles.filter((triangle) =>
      circumcircleContains(points, triangle, points[index]),
    );
    const edges = new Map<string, { edge: [number, number]; count: number }>();
    for (const [a, b, c] of bad) {
      for (const edge of [[a, b], [b, c], [c, a]] as [number, number][]) {
        const key = edge[0] < edge[1] ? `${edge[0]}:${edge[1]}` : `${edge[1]}:${edge[0]}`;
        const current = edges.get(key);
        if (current) current.count += 1;
        else edges.set(key, { edge, count: 1 });
      }
    }
    const badSet = new Set(bad);
    triangles = triangles.filter((triangle) => !badSet.has(triangle));
    for (const { edge, count: edgeCount } of edges.values()) {
      if (edgeCount !== 1) continue;
      const triangle: Triangle = [edge[0], edge[1], index];
      if (signedArea(points[triangle[0]], points[triangle[1]], points[triangle[2]]) < 0) {
        [triangle[0], triangle[1]] = [triangle[1], triangle[0]];
      }
      triangles.push(triangle);
    }
  }

  return triangles.filter(
    (triangle) =>
      triangle.every((index) => index < count) &&
      Math.abs(signedArea(input[triangle[0]], input[triangle[1]], input[triangle[2]])) > 0.08,
  );
}

function fieldDisplacement(source: Point, controls: Control[]) {
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
  if (sum === 0) return { x: 0, y: 0 };
  const influence = Math.min(1, sum);
  return {
    x: (dx / sum) * influence,
    y: (dy / sum) * influence,
  };
}

type MeshData = {
  points: Point[];
  facePointCount: number;
  triangles: Triangle[];
  boundary: Point[];
  signature: string;
};

const meshCache = new WeakMap<HTMLCanvasElement, MeshData>();

function buildFaceMesh(
  sourceCanvas: HTMLCanvasElement,
  landmarks: Point[],
): MeshData {
  const { width, height } = sourceCanvas;
  const signature = `${width}x${height}:${[10, 33, 152, 234, 263, 454]
    .map((index) => `${landmarks[index].x.toFixed(5)},${landmarks[index].y.toFixed(5)}`)
    .join(":")}`;
  const cached = meshCache.get(sourceCanvas);
  if (cached?.signature === signature) return cached;
  const facePoints = landmarks.slice(0, 468).map((point) => ({
    x: point.x * width,
    y: point.y * height,
  }));
  const points = [...facePoints];
  const facePointCount = points.length;
  const oval = FACE_OVAL.map((index) => facePoints[index]);
  const minX = Math.min(...oval.map((point) => point.x));
  const maxX = Math.max(...oval.map((point) => point.x));
  const minY = Math.min(...oval.map((point) => point.y));
  const maxY = Math.max(...oval.map((point) => point.y));
  const center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };

  // A fixed ring follows the actual face oval. Its zero-motion vertices keep the
  // hair, ears, clothing, and background unchanged outside the narrow blend band.
  const boundary: Point[] = [];
  for (const ovalPoint of oval) {
    const anchor = {
      x: clamp(center.x + (ovalPoint.x - center.x) * 1.065, 0, width),
      y: clamp(center.y + (ovalPoint.y - center.y) * 1.055, 0, height),
    };
    boundary.push(anchor);
    points.push(anchor);
  }

  const internalTriangles = canonicalFaceTriangles ?? delaunay(facePoints);
  const ringTriangles: Triangle[] = [];
  for (let index = 0; index < FACE_OVAL.length; index += 1) {
    const next = (index + 1) % FACE_OVAL.length;
    const inner = FACE_OVAL[index];
    const innerNext = FACE_OVAL[next];
    const outer = facePointCount + index;
    const outerNext = facePointCount + next;
    ringTriangles.push([inner, innerNext, outerNext], [inner, outerNext, outer]);
  }
  const triangles = [...internalTriangles, ...ringTriangles].filter(
    ([a, b, c]) => Math.abs(signedArea(points[a], points[b], points[c])) > 0.08,
  );

  const mesh = {
    points,
    facePointCount,
    triangles,
    boundary,
    signature,
  };
  meshCache.set(sourceCanvas, mesh);
  return mesh;
}

function regularizeDisplacements(displacements: Point[], mesh: MeshData) {
  const adjacency = Array.from({ length: mesh.points.length }, () => new Set<number>());
  for (const [a, b, c] of mesh.triangles) {
    adjacency[a].add(b); adjacency[a].add(c);
    adjacency[b].add(a); adjacency[b].add(c);
    adjacency[c].add(a); adjacency[c].add(b);
  }
  let current = displacements.map((point) => ({ x: point.x, y: point.y }));
  for (let iteration = 0; iteration < 2; iteration += 1) {
    current = current.map((point, index) => {
      if (index >= mesh.facePointCount || adjacency[index].size === 0) {
        return { x: 0, y: 0 };
      }
      let x = 0;
      let y = 0;
      for (const neighbor of adjacency[index]) {
        x += current[neighbor].x;
        y += current[neighbor].y;
      }
      const amount = 0.16;
      return {
        x: point.x * (1 - amount) + (x / adjacency[index].size) * amount,
        y: point.y * (1 - amount) + (y / adjacency[index].size) * amount,
      };
    });
  }
  return current;
}

function targetForScale(points: Point[], displacements: Point[], scale: number) {
  return points.map((point, index) => ({
    x: point.x + displacements[index].x * scale,
    y: point.y + displacements[index].y * scale,
  }));
}

function planIsSafe(
  source: Point[],
  target: Point[],
  triangles: Triangle[],
  width: number,
  height: number,
) {
  if (target.some((point) => point.x < 0 || point.y < 0 || point.x > width || point.y > height)) {
    return false;
  }
  const edgeRatio = (a: number, b: number) => {
    const sourceLength = Math.hypot(source[a].x - source[b].x, source[a].y - source[b].y);
    if (sourceLength < 1.5) return 1;
    const targetLength = Math.hypot(target[a].x - target[b].x, target[a].y - target[b].y);
    return targetLength / sourceLength;
  };
  return triangles.every(([a, b, c]) => {
    const sourceArea = signedArea(source[a], source[b], source[c]);
    const targetArea = signedArea(target[a], target[b], target[c]);
    const areaRatio = Math.abs(targetArea / sourceArea);
    if (sourceArea * targetArea <= 0 || areaRatio < 0.75 || areaRatio > 1.3) return false;
    const edgesSafe = [[a, b], [b, c], [c, a]].every(([start, end]) => {
      const ratio = edgeRatio(start, end);
      return ratio > 0.82 && ratio < 1.22;
    });
    if (!edgesSafe) return false;

    const sx1 = source[b].x - source[a].x;
    const sy1 = source[b].y - source[a].y;
    const sx2 = source[c].x - source[a].x;
    const sy2 = source[c].y - source[a].y;
    const tx1 = target[b].x - target[a].x;
    const ty1 = target[b].y - target[a].y;
    const tx2 = target[c].x - target[a].x;
    const ty2 = target[c].y - target[a].y;
    const determinant = sx1 * sy2 - sx2 * sy1;
    const f00 = (tx1 * sy2 - tx2 * sy1) / determinant;
    const f01 = (-tx1 * sx2 + tx2 * sx1) / determinant;
    const f10 = (ty1 * sy2 - ty2 * sy1) / determinant;
    const f11 = (-ty1 * sx2 + ty2 * sx1) / determinant;
    const frobeniusSquared = f00 ** 2 + f01 ** 2 + f10 ** 2 + f11 ** 2;
    const determinantSquared = (f00 * f11 - f01 * f10) ** 2;
    const discriminant = Math.sqrt(
      Math.max(0, frobeniusSquared ** 2 - 4 * determinantSquared),
    );
    const maximumStretch = Math.sqrt((frobeniusSquared + discriminant) / 2);
    const minimumStretch = Math.sqrt(
      Math.max(0, (frobeniusSquared - discriminant) / 2),
    );
    return (
      minimumStretch > 0.78 &&
      maximumStretch < 1.25 &&
      maximumStretch / Math.max(minimumStretch, 0.001) < 1.35
    );
  });
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
    return {
      canvas: output,
      movedRegions: [],
      maxMovementPx: 0,
      safetyScale: 0,
      safetyStatus: "identity",
    };
  }

  const controls = buildControls(landmarks, width, height, preset, strength);
  const mesh = buildFaceMesh(sourceCanvas, landmarks);
  const rawDisplacements = mesh.points.map((point, index) =>
    index < mesh.facePointCount
      ? fieldDisplacement(point, controls)
      : { x: 0, y: 0 },
  );
  const displacements = regularizeDisplacements(rawDisplacements, mesh);

  let planScale = 1;
  let targetPoints = targetForScale(mesh.points, displacements, planScale);
  if (!planIsSafe(mesh.points, targetPoints, mesh.triangles, width, height)) {
    let safeScale = 0;
    let unsafeScale = 1;
    for (let attempt = 0; attempt < 10; attempt += 1) {
      const candidateScale = (safeScale + unsafeScale) / 2;
      const candidate = targetForScale(mesh.points, displacements, candidateScale);
      if (planIsSafe(mesh.points, candidate, mesh.triangles, width, height)) {
        safeScale = candidateScale;
      }
      else unsafeScale = candidateScale;
    }
    planScale = safeScale;
    targetPoints = targetForScale(mesh.points, displacements, planScale);
  }

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(sourceCanvas, 0, 0);
  const identityOnly = controls.length === 0 || planScale < 0.08 || mesh.triangles.length === 0;
  if (!identityOnly) {
    const layer = document.createElement("canvas");
    layer.width = width;
    layer.height = height;
    const layerContext = layer.getContext("2d");
    if (!layerContext) throw new Error("Canvas rendering is unavailable in this browser.");
    layerContext.imageSmoothingEnabled = true;
    layerContext.imageSmoothingQuality = "high";
    // Render the complete anchored face region once. Drawing only moved triangles
    // over the original would leave duplicate contours when a feature contracts.
    layerContext.save();
    layerContext.beginPath();
    mesh.boundary.forEach((point, index) => {
      if (index === 0) layerContext.moveTo(point.x, point.y);
      else layerContext.lineTo(point.x, point.y);
    });
    layerContext.closePath();
    layerContext.clip();
    for (const [a, b, c] of mesh.triangles) {
      drawTriangle(
        layerContext,
        sourceCanvas,
        [mesh.points[a], mesh.points[b], mesh.points[c]],
        [targetPoints[a], targetPoints[b], targetPoints[c]],
      );
    }
    layerContext.restore();
    ctx.drawImage(layer, 0, 0);
  }

  const maxMovementPx = identityOnly
    ? 0
    : Math.max(
        ...displacements.map((point) => Math.hypot(point.x, point.y) * planScale),
        0,
      );
  return {
    canvas: output,
    movedRegions: identityOnly ? [] : [...new Set(controls.map((control) => control.region))],
    maxMovementPx,
    safetyScale: identityOnly ? 0 : planScale,
    safetyStatus: identityOnly ? "identity" : planScale < 0.995 ? "weakened" : "passed",
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
