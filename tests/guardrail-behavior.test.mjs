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
assert.ok(
  initializeFaceTopology(FaceLandmarker.FACE_LANDMARKS_TESSELATION) > 800,
  "guardrail tests must exercise the production MediaPipe topology",
);

const IDENTITY_MATRIX = {
  rows: 4,
  columns: 4,
  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
};

/**
 * A deterministic, non-degenerate synthetic mesh with face-like placement for
 * every landmark used by the planner. The remaining samples fill the facial
 * oval so the renderer's geometric guardrail is exercised, not mocked away.
 */
function faceFixture({ asymmetric = false, yaw = 0 } = {}) {
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
    // Preserve the eye line used for roll normalization, while introducing
    // repeatable vertical residuals across brows, mouth, and lower jaw.
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

function actionMagnitude(plan) {
  return plan.actions.reduce((total, action) => total + Math.abs(action.amount), 0);
}

test("a clean frontal capture receives a visible balanced plan", () => {
  const analysis = analyzeFace(faceFixture({ asymmetric: true }), 1000, 1200, {
    qualityConfidence: 1,
    temporalStability: 1,
    resolutionSupport: 1,
    exposureSupport: 1,
  });
  const plan = createMorphPlan(analysis, {
    harmony: 60,
    symmetry: 35,
    dimorphism: 30,
  });

  assert.ok(analysis.overallConfidence >= 0.8, "fixture must represent a clean supported capture");
  assert.ok(plan.actions.length >= 3, "a supported, visibly deviant face should not collapse to one tiny edit");
  assert.ok(
    actionMagnitude(plan) >= 0.45,
    `default direction weights should retain a perceptible action budget (received ${actionMagnitude(plan).toFixed(3)})`,
  );
  assert.equal(plan.selectedCandidate, "balanced");
});

test("harmony, symmetry, and dimorphism contribute to the same plan", () => {
  const analysis = analyzeFace(faceFixture({ asymmetric: true }), 1000, 1200, {
    qualityConfidence: 1,
    temporalStability: 1,
    resolutionSupport: 1,
    exposureSupport: 1,
  });
  const plan = createMorphPlan(analysis, { harmony: 100, symmetry: 100, dimorphism: 100 });
  const contributed = new Set(plan.actions.flatMap((action) => action.directions));
  const blendedAction = plan.actions.find((action) =>
    action.directions.includes("harmony") && action.directions.includes("dimorphism"));

  assert.deepEqual([...contributed].sort(), ["dimorphism", "harmony", "symmetry"]);
  assert.ok(blendedAction, "compatible directions should accumulate on one primitive instead of replacing each other");
  assert.ok(actionMagnitude(plan) >= 0.8, "full blended directions should retain a useful edit budget");
});

test("low-confidence analysis still resolves to a true no-op", () => {
  const analysis = analyzeFace(faceFixture({ asymmetric: true }), 1000, 1200, {
    qualityConfidence: 0.42,
    temporalStability: 0.42,
    resolutionSupport: 0.42,
    exposureSupport: 0.42,
  });
  const plan = createMorphPlan(analysis, { harmony: 100, symmetry: 100, dimorphism: 100 });

  assert.ok(analysis.overallConfidence < 0.5);
  assert.equal(plan.selectedCandidate, "identity");
  assert.equal(plan.actions.length, 0);
  assert.match(plan.rejectedReasons.join(" "), /confidence/i);
});

test("the planner never edits a region the analysis marks as locked", () => {
  const analysis = analyzeFace(faceFixture(), 1000, 1200, {
    qualityConfidence: 1,
    temporalStability: 1,
    resolutionSupport: 1,
    exposureSupport: 1,
  });
  analysis.regionEditability = analysis.regionEditability.map((entry) =>
    entry.region === "Jaw"
      ? { ...entry, score: 0.54, editable: false, reasons: ["Synthetic locked region"] }
      : entry,
  );
  const plan = createMorphPlan(analysis, { harmony: 0, symmetry: 0, dimorphism: 100 });

  assert.ok(!plan.actions.some((action) => action.region === "Jaw"));
  assert.match(plan.rejectedReasons.join(" "), /Jaw: Synthetic locked region/i);
});

class CanvasContextStub {
  drawImage() {}
  save() {}
  restore() {}
  beginPath() {}
  moveTo() {}
  lineTo() {}
  closePath() {}
  clip() {}
  setTransform() {}
}

class CanvasStub {
  constructor(width = 1000, height = 1200) {
    this.width = width;
    this.height = height;
    this.context = new CanvasContextStub();
  }

  getContext() {
    return this.context;
  }
}

function renderPlan(actions, selectedCandidate = "balanced", strength = 85, observation = faceFixture()) {
  const previousDocument = globalThis.document;
  globalThis.document = {
    createElement(tag) {
      assert.equal(tag, "canvas");
      return new CanvasStub();
    },
  };
  try {
    const plan = {
      actions,
      preservedRegions: [],
      rejectedReasons: [],
      candidateCount: 4,
      selectedCandidate,
      directionContributions: { harmony: 100, symmetry: 0, dimorphism: 0 },
    };
    return morphImage(new CanvasStub(), observation.landmarks, plan, strength);
  } finally {
    globalThis.document = previousDocument;
  }
}

test("production topology preserves visible edits across supported poses", () => {
  for (const { yaw, expectedPose } of [
    { yaw: 0, expectedPose: /^frontal$/ },
    { yaw: 30, expectedPose: /three-quarter/ },
    { yaw: 58, expectedPose: /profile/ },
  ]) {
    const observation = faceFixture({ asymmetric: yaw === 0, yaw });
    const analysis = analyzeFace(observation, 1000, 1200, {
      qualityConfidence: 1,
      temporalStability: 1,
      resolutionSupport: 1,
      exposureSupport: 1,
    });
    const plan = createMorphPlan(analysis, { harmony: 70, symmetry: 35, dimorphism: 55 });
    const rendered = renderPlan(plan.actions, plan.selectedCandidate, 70, observation);

    assert.match(analysis.pose.class, expectedPose);
    assert.ok(plan.actions.length > 0, `${analysis.pose.label} should retain at least one supported action`);
    assert.notEqual(rendered.safetyStatus, "identity", `${analysis.pose.label} should retain a safe visible warp`);
    for (const region of new Set(plan.actions.map((action) => action.region))) {
      assert.ok(
        rendered.movedRegions.includes(region),
        `${analysis.pose.label} planned ${region}, so the renderer must not silently discard it`,
      );
    }
    if (analysis.pose.class.includes("three-quarter") && plan.actions.some((action) => action.region === "Nose")) {
      assert.ok(
        rendered.perRegionScale.Nose >= 0.8,
        `the visible-side three-quarter nose field should not trigger heavy global backoff (received ${rendered.perRegionScale.Nose})`,
      );
    }
    assert.ok(
      rendered.maxMovementPx >= 0.5,
      `${analysis.pose.label} should not collapse to sub-pixel identity (movement ${rendered.maxMovementPx.toFixed(3)}, scales ${JSON.stringify(rendered.perRegionScale)})`,
    );
  }
});

test("the visible strength control produces a material response before safe backoff", () => {
  const observation = faceFixture({ asymmetric: true });
  const analysis = analyzeFace(observation, 1000, 1200, {
    qualityConfidence: 1,
    temporalStability: 1,
    resolutionSupport: 1,
    exposureSupport: 1,
  });
  const plan = createMorphPlan(analysis, { harmony: 60, symmetry: 35, dimorphism: 30 });
  const defaultStrength = renderPlan(plan.actions, plan.selectedCandidate, 50, observation);
  const highStrength = renderPlan(plan.actions, plan.selectedCandidate, 90, observation);

  assert.ok(
    defaultStrength.maxMovementPx >= 1.75,
    `default strength should create a visible edit (received ${defaultStrength.maxMovementPx.toFixed(3)}px)`,
  );
  assert.ok(
    highStrength.maxMovementPx >= defaultStrength.maxMovementPx * 1.25,
    `raising strength from 50 to 90 must materially change the image (${defaultStrength.maxMovementPx.toFixed(3)}px → ${highStrength.maxMovementPx.toFixed(3)}px)`,
  );
});

test("the geometric guardrail weakens an unsafe request instead of erasing it", () => {
  const moderate = renderPlan([
    {
      primitive: "jaw-width",
      region: "Jaw",
      amount: 0.42,
      confidence: 0.9,
      editability: 0.9,
      directions: ["harmony"],
      rationale: "Synthetic moderate plan",
    },
  ]);
  const aggressive = renderPlan([
    {
      primitive: "jaw-width",
      region: "Jaw",
      amount: 4,
      confidence: 0.9,
      editability: 0.9,
      directions: ["harmony"],
      rationale: "Synthetic unsafe plan",
    },
  ], "full", 100);

  assert.notEqual(moderate.safetyStatus, "identity", "ordinary supported edits must survive the guardrail");
  assert.ok(moderate.maxMovementPx >= 1.5, "a moderate request should remain visible after safety checks");
  assert.equal(aggressive.safetyStatus, "weakened");
  assert.ok(aggressive.safetyScale >= 0.08, "backoff should preserve a bounded safe edit, not silently no-op");
  assert.ok(aggressive.safetyScale < 1, "the deliberately unsafe request must actually exercise backoff");
  assert.ok(aggressive.maxMovementPx <= moderate.maxMovementPx * 3, "backoff must cap runaway geometry");
});

test("a constrained region does not erase regions that already passed", () => {
  const combined = renderPlan([
    {
      primitive: "jaw-width",
      region: "Jaw",
      amount: 0.28,
      confidence: 0.9,
      editability: 0.9,
      directions: ["harmony"],
      rationale: "Synthetic ordinary jaw plan",
    },
    {
      primitive: "nose-width",
      region: "Nose",
      amount: 4,
      confidence: 0.9,
      editability: 0.9,
      directions: ["harmony"],
      rationale: "Synthetic oversized nose plan",
    },
  ], "full", 100);

  assert.ok(combined.perRegionScale.Jaw > 0, "the safe jaw region should survive");
  assert.ok(combined.movedRegions.includes("Jaw"));
  assert.ok(combined.perRegionScale.Nose < 1, "only the oversized nose request should be backed off");
  assert.equal(combined.safetyStatus, "weakened");
});
