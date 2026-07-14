import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";

import {
  analyzeFace,
  createMorphPlan,
} from "../lib/face-intelligence.ts";
import { CANONICAL_FACE_LANDMARKS } from "./fixtures/canonical-face-landmarks.mjs";

// The production bundler resolves TypeScript's extensionless local import.
// Node's strip-types loader needs the equivalent explicit suffix in tests.
const moduleHooks = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "./face-intelligence") {
      return nextResolve("./face-intelligence.ts", context);
    }
    return nextResolve(specifier, context);
  },
});
const { initializeFaceTopology, morphImage } = await import("../lib/morph.ts");
moduleHooks.deregister();
const { FaceLandmarker } = await import("@mediapipe/tasks-vision");
function trianglesFromConnections(connections) {
  const triangles = [];
  for (let index = 0; index + 2 < connections.length; index += 3) {
    const vertices = [
      ...new Set(
        connections
          .slice(index, index + 3)
          .flatMap((connection) => [connection.start, connection.end]),
      ),
    ];
    if (vertices.length === 3 && vertices.every((vertex) => vertex < 468)) {
      triangles.push(vertices);
    }
  }
  return triangles;
}
const topologyTriangles = trianglesFromConnections(
  FaceLandmarker.FACE_LANDMARKS_TESSELATION,
);
const topologyTriangleCount = initializeFaceTopology(
  FaceLandmarker.FACE_LANDMARKS_TESSELATION,
);
assert.ok(topologyTriangleCount > 800, "tests must exercise the production face topology");
assert.equal(topologyTriangleCount, topologyTriangles.length);

const WIDTH = 1000;
const HEIGHT = 1200;
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
  127, 162, 21, 54, 103, 67, 109,
];
const QUALITY = {
  qualityConfidence: 1,
  temporalStability: 1,
  resolutionSupport: 1,
  exposureSupport: 1,
};
const IDENTITY_MATRIX = {
  rows: 4,
  columns: 4,
  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
};

function faceFixture({ yaw = 0, asymmetric = false } = {}) {
  const yawRadians = yaw * Math.PI / 180;
  const cosine = Math.cos(yawRadians);
  const sine = Math.sin(yawRadians);
  const landmarks = CANONICAL_FACE_LANDMARKS.map((point) => {
    const localX = point.x - 0.5;
    return {
      x: 0.5 + localX * cosine + point.z * 1.4 * sine,
      y: point.y,
      z: point.z * cosine - (localX / 1.4) * sine,
    };
  });

  if (asymmetric) {
    for (const index of [300, 334, 291, 308, 397, 365]) {
      landmarks[index] = { ...landmarks[index], y: landmarks[index].y + 0.018 };
    }
  }

  return {
    landmarks,
    blendshapes: {},
    transformationMatrix: yaw === 0 ? IDENTITY_MATRIX : {
      rows: 4,
      columns: 4,
      data: [cosine, 0, -sine, 0, 0, 1, 0, 0, sine, 0, cosine, 0, 0, 0, 0, 1],
    },
  };
}

class RecordingContext {
  constructor() {
    this.drawImageCalls = [];
    this.transforms = [];
    this.clipPaths = [];
    this.currentPath = [];
  }

  drawImage(...args) { this.drawImageCalls.push(args); }
  save() {}
  restore() {}
  beginPath() { this.currentPath = []; }
  moveTo(x, y) { this.currentPath.push({ x, y }); }
  lineTo(x, y) { this.currentPath.push({ x, y }); }
  closePath() {}
  clip() { this.clipPaths.push(this.currentPath.map((point) => ({ ...point }))); }
  setTransform(a, b, c, d, e, f) { this.transforms.push({ a, b, c, d, e, f }); }
}

class RecordingCanvas {
  constructor(width = WIDTH, height = HEIGHT) {
    this.width = width;
    this.height = height;
    this.context = new RecordingContext();
  }

  getContext() { return this.context; }
}

function render(observation, plan, strength) {
  const priorDocument = globalThis.document;
  const createdCanvases = [];
  globalThis.document = {
    createElement(tag) {
      assert.equal(tag, "canvas");
      const canvas = new RecordingCanvas();
      createdCanvases.push(canvas);
      return canvas;
    },
  };
  const source = new RecordingCanvas();
  try {
    const result = morphImage(source, observation.landmarks, plan, strength);
    return {
      ...result,
      source,
      output: createdCanvases[0],
      layer: createdCanvases[1] ?? null,
    };
  } finally {
    globalThis.document = priorDocument;
  }
}

function faceWidthPx(observation) {
  const left = observation.landmarks[234];
  const right = observation.landmarks[454];
  return Math.hypot((right.x - left.x) * WIDTH, (right.y - left.y) * HEIGHT);
}

function poseStableFaceReferencePx(observation) {
  const forehead = observation.landmarks[10];
  const chin = observation.landmarks[152];
  const faceHeight = Math.hypot(
    (chin.x - forehead.x) * WIDTH,
    (chin.y - forehead.y) * HEIGHT,
  );
  // Projected cheek width collapses naturally in profile. Retain a fraction of
  // face height so the perceptibility contract does not get easier with yaw.
  return Math.max(faceWidthPx(observation), faceHeight * 0.65);
}

function expectedBoundary(observation) {
  const oval = FACE_OVAL.map((index) => ({
    x: observation.landmarks[index].x * WIDTH,
    y: observation.landmarks[index].y * HEIGHT,
  }));
  const minX = Math.min(...oval.map((point) => point.x));
  const maxX = Math.max(...oval.map((point) => point.x));
  const minY = Math.min(...oval.map((point) => point.y));
  const maxY = Math.max(...oval.map((point) => point.y));
  const center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  return oval.map((point) => ({
    x: clamp(center.x + (point.x - center.x) * 1.18, 0, WIDTH),
    y: clamp(center.y + (point.y - center.y) * 1.16, 0, HEIGHT),
  }));
}

function transformPoint(transform, point) {
  return {
    x: transform.a * point.x + transform.c * point.y + transform.e,
    y: transform.b * point.x + transform.d * point.y + transform.f,
  };
}

function assertPointClose(actual, expected, label) {
  assert.ok(
    Math.hypot(actual.x - expected.x, actual.y - expected.y) <= 0.002,
    `${label} moved (${expected.x.toFixed(3)}, ${expected.y.toFixed(3)}) to (${actual.x.toFixed(3)}, ${actual.y.toFixed(3)})`,
  );
}

function targetLandmarks(rendered, observation) {
  assert.ok(rendered.layer, "a visible morph should create a warped face layer");
  const transforms = rendered.layer.context.transforms;
  const targets = Array.from({ length: 468 }, () => null);
  for (let triangleIndex = 0; triangleIndex < topologyTriangles.length; triangleIndex += 1) {
    const transform = transforms[triangleIndex];
    assert.ok(transform, `missing affine map for production triangle ${triangleIndex}`);
    for (const landmarkIndex of topologyTriangles[triangleIndex]) {
      const source = {
        x: observation.landmarks[landmarkIndex].x * WIDTH,
        y: observation.landmarks[landmarkIndex].y * HEIGHT,
      };
      const target = transformPoint(transform, source);
      if (targets[landmarkIndex]) {
        assertPointClose(target, targets[landmarkIndex], `shared landmark ${landmarkIndex}`);
      } else {
        targets[landmarkIndex] = target;
      }
    }
  }
  assert.ok(
    targets.filter(Boolean).length >= 460,
    "production topology should expose target positions for nearly all face landmarks",
  );
  return targets;
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * fraction)];
}

function displacementStats(rendered, observation) {
  const targets = targetLandmarks(rendered, observation);
  const displacements = targets.flatMap((target, index) => {
    if (!target) return [];
    const source = {
      x: observation.landmarks[index].x * WIDTH,
      y: observation.landmarks[index].y * HEIGHT,
    };
    return [Math.hypot(target.x - source.x, target.y - source.y)];
  });
  return {
    targets,
    p75: percentile(displacements, 0.75),
    p95: percentile(displacements, 0.95),
    max: Math.max(...displacements),
  };
}

function singularValues(transform) {
  const frobeniusSquared = transform.a ** 2 + transform.b ** 2 + transform.c ** 2 + transform.d ** 2;
  const determinantSquared = (transform.a * transform.d - transform.b * transform.c) ** 2;
  const discriminant = Math.sqrt(Math.max(0, frobeniusSquared ** 2 - 4 * determinantSquared));
  return {
    maximum: Math.sqrt((frobeniusSquared + discriminant) / 2),
    minimum: Math.sqrt(Math.max(0, (frobeniusSquared - discriminant) / 2)),
  };
}

function assertPairedOrdering(stats, observation) {
  for (const [left, right, label] of [
    [33, 263, "eye corners"],
    [98, 327, "nostrils"],
    [61, 291, "mouth corners"],
  ]) {
    const sourceSpan = (observation.landmarks[right].x - observation.landmarks[left].x) * WIDTH;
    const targetSpan = stats.targets[right].x - stats.targets[left].x;
    assert.ok(sourceSpan * targetSpan > 0, `${label} reversed horizontal order`);
    assert.ok(
      Math.abs(targetSpan) >= Math.abs(sourceSpan) * 0.7,
      `${label} collapsed below 70% of its original span`,
    );
  }
}

function assertAnchoredSafeRender(rendered, observation) {
  assert.ok(rendered.layer, "a visible morph should create a warped face layer");
  assert.equal(
    rendered.output.context.drawImageCalls[0]?.[0],
    rendered.source,
    "the complete original frame must be drawn before the local face overlay",
  );

  const boundary = expectedBoundary(observation);
  const clip = rendered.layer.context.clipPaths[0];
  assert.equal(clip?.length, boundary.length, "the warped layer must be clipped to the fixed face ring");
  clip.forEach((point, index) => assertPointClose(point, boundary[index], `clip anchor ${index}`));

  const transforms = rendered.layer.context.transforms;
  assert.equal(
    transforms.length,
    topologyTriangleCount + FACE_OVAL.length * 2,
    "every production and boundary triangle should be rendered",
  );
  for (const [index, transform] of transforms.entries()) {
    const determinant = transform.a * transform.d - transform.b * transform.c;
    assert.ok(
      Number.isFinite(determinant) && determinant > 0,
      `triangle ${index} folded or produced an invalid affine map (determinant ${determinant})`,
    );
    const { minimum, maximum } = singularValues(transform);
    assert.ok(
      minimum >= 0.68 && maximum <= 1.48 && maximum / Math.max(minimum, 1e-6) <= 1.8,
      `triangle ${index} exceeded the local stretch contract (singular values ${minimum.toFixed(3)}, ${maximum.toFixed(3)})`,
    );
  }

  // Each pair of final triangles joins the moving face oval to this zero-motion
  // ring. Verify the affine maps leave every outer vertex exactly in place.
  for (let index = 0; index < boundary.length; index += 1) {
    const next = (index + 1) % boundary.length;
    const firstRingTriangle = transforms[topologyTriangleCount + index * 2];
    const secondRingTriangle = transforms[topologyTriangleCount + index * 2 + 1];
    assertPointClose(
      transformPoint(firstRingTriangle, boundary[next]),
      boundary[next],
      `outer ring ${next}`,
    );
    assertPointClose(
      transformPoint(secondRingTriangle, boundary[next]),
      boundary[next],
      `outer ring ${next}`,
    );
    assertPointClose(
      transformPoint(secondRingTriangle, boundary[index]),
      boundary[index],
      `outer ring ${index}`,
    );
  }
}

for (const { yaw, pose, minimumP95, minimumMax } of [
  { yaw: 0, pose: "frontal", minimumP95: 0.018, minimumMax: 0.018 },
  { yaw: 30, pose: "three-quarter", minimumP95: 0.012, minimumMax: 0.018 },
  { yaw: 58, pose: "profile", minimumP95: 0, minimumMax: 0.013 },
]) {
  test(`a clean ${pose} face receives a materially stronger safe edit`, () => {
    const observation = faceFixture({ yaw, asymmetric: yaw === 0 });
    const analysis = analyzeFace(observation, WIDTH, HEIGHT, QUALITY);
    const plan = createMorphPlan(analysis, {
      harmony: 100,
      symmetry: yaw === 0 ? 100 : 0,
      dimorphism: 100,
    });
    const medium = render(observation, plan, 55);
    const strong = render(observation, plan, 100);
    const faceReference = poseStableFaceReferencePx(observation);
    const mediumStats = displacementStats(medium, observation);
    const strongStats = displacementStats(strong, observation);

    assert.match(analysis.pose.class, new RegExp(pose));
    assert.ok(new Set(plan.actions.map((action) => action.region)).size >= 2, `${pose} fixture must edit at least two independently supported regions`);
    assert.notEqual(strong.safetyStatus, "identity", `${pose} strong morph must remain visible`);
    if (minimumP95 > 0) {
      assert.ok(
        strongStats.p95 >= faceReference * minimumP95,
        `${pose} strong morph P95 must reach ${(minimumP95 * 100).toFixed(1)}% of pose-stable face scale (P75 ${strongStats.p75.toFixed(2)}px, P95 ${strongStats.p95.toFixed(2)}px, max ${strongStats.max.toFixed(2)}px, face ${faceReference.toFixed(2)}px)`,
      );
    }
    assert.ok(
      strongStats.max >= faceReference * minimumMax,
      `${pose} strong morph must reach ${(minimumMax * 100).toFixed(1)}% of pose-stable face scale at its active contour (${strongStats.max.toFixed(2)}px of ${faceReference.toFixed(2)}px)`,
    );
    assert.ok(
      strongStats.p95 >= mediumStats.p95 * 1.5,
      `${pose} 100% P95 must materially exceed 55% (P75 ${mediumStats.p75.toFixed(2)}px -> ${strongStats.p75.toFixed(2)}px; P95 ${mediumStats.p95.toFixed(2)}px -> ${strongStats.p95.toFixed(2)}px)`,
    );
    assert.ok(
      strong.maxMovementPx <= faceReference * 0.04,
      `${pose} morph exceeded the conservative whole-face displacement ceiling`,
    );
    assertPairedOrdering(strongStats, observation);

    const regionalCaps = {
      Jaw: 0.05,
      Chin: 0.05,
      Nose: 0.025,
      Lips: 0.025,
      Brows: 0.018,
      Symmetry: 0.018,
    };
    for (const region of new Set(plan.actions.map((action) => action.region))) {
      const regionalPlan = {
        ...plan,
        actions: plan.actions.filter((action) => action.region === region),
      };
      const regionalRender = render(observation, regionalPlan, 100);
      if (regionalRender.safetyStatus === "identity") continue;
      const regionalStats = displacementStats(regionalRender, observation);
      assert.ok(
        regionalStats.max <= faceReference * regionalCaps[region],
        `${pose} ${region} exceeded its ${(regionalCaps[region] * 100).toFixed(1)}% regional cap`,
      );
    }
    assertAnchoredSafeRender(strong, observation);
  });
}

test("a locked region remains absent even at maximum visible strength", () => {
  const observation = faceFixture({ asymmetric: true });
  const analysis = analyzeFace(observation, WIDTH, HEIGHT, QUALITY);
  analysis.regionEditability = analysis.regionEditability.map((entry) => entry.region === "Jaw"
    ? { ...entry, score: 0.99, editable: false, reasons: ["Synthetic hard lock"] }
    : entry);
  const plan = createMorphPlan(analysis, { harmony: 100, symmetry: 100, dimorphism: 100 });
  const rendered = render(observation, plan, 100);

  assert.ok(!plan.actions.some((action) => action.region === "Jaw"));
  assert.ok(!rendered.movedRegions.includes("Jaw"));
  assert.equal(rendered.perRegionScale.Jaw, undefined);
  assert.match(plan.rejectedReasons.join(" "), /Jaw: Synthetic hard lock/i);
  assertAnchoredSafeRender(rendered, observation);
});

test("zero and negative jaw budgets remain hard no-op caps", () => {
  const observation = faceFixture();
  const analysis = analyzeFace(observation, WIDTH, HEIGHT, QUALITY);
  const fullPlan = createMorphPlan(analysis, { harmony: 100, symmetry: 0, dimorphism: 100 });
  const jawAction = fullPlan.actions.find((action) => action.region === "Jaw");
  assert.ok(jawAction, "fixture must produce a Jaw action");

  for (const maxDisplacement of [0, -0.02]) {
    const cappedPlan = {
      ...fullPlan,
      actions: [{ ...jawAction, maxDisplacement }],
    };
    const rendered = render(observation, cappedPlan, 100);
    assert.equal(rendered.maxMovementPx, 0);
    assert.ok(!rendered.movedRegions.includes("Jaw"));
    assert.equal(rendered.safetyStatus, "identity");
  }
});

test("jaw output stays visible from front to mild three-quarter poses", () => {
  const sourceObservation = faceFixture();
  const analysis = analyzeFace(sourceObservation, WIDTH, HEIGHT, QUALITY);
  const plan = createMorphPlan(analysis, { harmony: 100, symmetry: 0, dimorphism: 100 });
  const fixedJawPlan = {
    ...plan,
    actions: plan.actions.filter((action) => action.region === "Jaw"),
  };
  const movements = [];
  let priorMovement = 0;
  for (let yaw = 0; yaw <= 30; yaw += 1) {
    const rendered = render(faceFixture({ yaw }), fixedJawPlan, 100);
    movements.push(rendered.maxMovementPx);
    if (priorMovement > 0) {
      assert.ok(
        rendered.maxMovementPx >= priorMovement * 0.2 &&
          rendered.maxMovementPx <= priorMovement * 5,
        `one degree of yaw caused a jaw cliff at ${yaw}° (${priorMovement.toFixed(2)}px -> ${rendered.maxMovementPx.toFixed(2)}px)`,
      );
    }
    priorMovement = rendered.maxMovementPx;
  }
  assert.ok(
    Math.min(...movements) >= 1.8,
    `a supported mild yaw collapsed below a visible jaw edit (${movements.map((value) => value.toFixed(2)).join(", ")})`,
  );
});

test("visible strength does not reduce peak impact across supported poses", () => {
  for (const yaw of [0, 8, 18, 30, 58]) {
    const observation = faceFixture({ yaw, asymmetric: yaw === 0 });
    const analysis = analyzeFace(observation, WIDTH, HEIGHT, QUALITY);
    const plan = createMorphPlan(analysis, {
      harmony: 100,
      symmetry: yaw === 0 ? 100 : 0,
      dimorphism: 100,
    });
    let priorImpact = 0;
    let priorStrength = 0;
    for (const strength of [55, 65, 75, 80, 85, 90, 95, 100]) {
      const rendered = render(observation, plan, strength);
      const impact = rendered.maxMovementPx;
      assert.ok(
        impact >= priorImpact * 0.98,
        `yaw ${yaw} became weaker from ${priorStrength}% to ${strength}% (${priorImpact.toFixed(2)}px -> ${impact.toFixed(2)}px peak; ${JSON.stringify(rendered.perRegionScale)})`,
      );
      priorImpact = Math.max(priorImpact, impact);
      priorStrength = strength;
    }
  }
});

test("a pathological request backs off to a bounded non-collapsed safe morph", () => {
  const observation = faceFixture({ asymmetric: true });
  const plan = {
    schemaVersion: "3.0.0-beta.1",
    actions: [
      {
        primitive: "jaw-width",
        region: "Jaw",
        amount: 50,
        confidence: 1,
        editability: 1,
        maxDisplacement: 0.024,
        directions: ["harmony"],
        rationale: "Synthetic malformed jaw request",
      },
      {
        primitive: "nose-width",
        region: "Nose",
        amount: -50,
        confidence: 1,
        editability: 1,
        maxDisplacement: 0.01,
        directions: ["harmony"],
        rationale: "Synthetic malformed nose request",
      },
    ],
    evidenceFamilies: [],
    evidenceMeasurementCount: 0,
    preservedRegions: ["Hair", "Background"],
    rejectedReasons: [],
    candidateCount: 4,
    selectedCandidate: "full",
    directionContributions: { harmony: 100, symmetry: 0, dimorphism: 0 },
  };
  const rendered = render(observation, plan, 100);
  const faceReference = poseStableFaceReferencePx(observation);

  assert.equal(rendered.safetyStatus, "weakened");
  assert.ok(
    rendered.maxMovementPx >= faceReference * 0.001,
    `backoff should preserve a visible bounded morph, not collapse to identity (${rendered.maxMovementPx.toFixed(3)}px)`,
  );
  assert.ok(rendered.maxMovementPx <= faceReference * 0.04, "pathological geometry must remain bounded");
  assert.ok(Object.values(rendered.perRegionScale).every((scale) => scale > 0 && scale < 1));
  assert.deepEqual([...rendered.movedRegions].sort(), ["Jaw", "Nose"]);
  assertAnchoredSafeRender(rendered, observation);
});
