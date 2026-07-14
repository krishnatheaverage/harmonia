import assert from "node:assert/strict";
import test from "node:test";

import {
  analyzeFace,
  createMorphPlan,
} from "../lib/face-intelligence.ts";
import { CANONICAL_FACE_LANDMARKS } from "./fixtures/canonical-face-landmarks.mjs";

const IDENTITY_MATRIX = {
  rows: 4,
  columns: 4,
  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
};

const QUALITY = {
  qualityConfidence: 1,
  temporalStability: 1,
  resolutionSupport: 1,
  exposureSupport: 1,
};

function observation({ yaw = 0, asymmetric = false, noiseSeed = null } = {}) {
  const yawRadians = yaw * Math.PI / 180;
  const cosine = Math.cos(yawRadians);
  const sine = Math.sin(yawRadians);
  const landmarks = CANONICAL_FACE_LANDMARKS.map((point, index) => {
    const localX = point.x - 0.5;
    const noiseX = noiseSeed === null
      ? 0
      : Math.sin((index + 1) * 12.9898 + noiseSeed * 78.233) * 0.00018;
    const noiseY = noiseSeed === null
      ? 0
      : Math.sin((index + 1) * 39.3467 + noiseSeed * 11.135) * 0.00018;
    return {
      x: 0.5 + localX * cosine + point.z * 1.4 * sine + noiseX,
      y: point.y + noiseY,
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

function analyze(options = {}) {
  return analyzeFace(observation(options), 1000, 1200, QUALITY);
}

function clone(value) {
  return structuredClone(value);
}

function actionMap(plan) {
  return new Map(plan.actions.map((action) => [action.primitive, action]));
}

function actionMagnitude(plan) {
  return plan.actions.reduce((sum, action) => sum + Math.abs(action.amount), 0);
}

function evidenceIds(plan) {
  return new Set(plan.evidenceFamilies.flatMap((family) => family.validMeasurementIds));
}

function assertComparablePlans(actual, expected, label) {
  const actualActions = actionMap(actual);
  const expectedActions = actionMap(expected);
  assert.deepEqual(
    [...actualActions.keys()].sort(),
    [...expectedActions.keys()].sort(),
    `${label} must not change which primitives the planner selects`,
  );
  for (const [primitive, baseline] of expectedActions) {
    const candidate = actualActions.get(primitive);
    assert.ok(candidate, `${label} lost ${primitive}`);
    assert.equal(
      Math.sign(candidate.amount),
      Math.sign(baseline.amount),
      `${label} reversed ${primitive}`,
    );
    const tolerance = Math.max(0.018, Math.abs(baseline.amount) * 0.18);
    assert.ok(
      Math.abs(candidate.amount - baseline.amount) <= tolerance,
      `${label} changed ${primitive} too much (${baseline.amount.toFixed(3)} -> ${candidate.amount.toFixed(3)})`,
    );
  }
}

test("a supported blended plan reports broad, independent measurement evidence", () => {
  const analysis = analyze({ asymmetric: true });
  const plan = createMorphPlan(analysis, { harmony: 100, symmetry: 100, dimorphism: 100 });

  assert.equal(plan.schemaVersion, "3.0.0-beta.1");
  assert.ok(Array.isArray(plan.evidenceFamilies), "the public plan schema must expose evidenceFamilies");
  assert.ok(plan.evidenceFamilies.length >= 3, "one aggregate ratio is not sufficient evidence for a blended morph");

  const knownMeasurements = new Map(analysis.measurements.map((measurement) => [measurement.id, measurement]));
  for (const family of plan.evidenceFamilies) {
    assert.equal(typeof family.id, "string");
    assert.ok(family.id.length > 0);
    assert.equal(typeof family.region, "string");
    assert.equal(typeof family.primitive, "string");
    assert.ok(Number.isFinite(family.signal) && Math.abs(family.signal) <= 1);
    assert.ok(Number.isFinite(family.confidence) && family.confidence >= 0 && family.confidence <= 1);
    assert.ok(Number.isFinite(family.agreement) && family.agreement >= 0 && family.agreement <= 1);
    assert.equal(typeof family.status, "string");
    assert.ok(Array.isArray(family.measurementIds) && family.measurementIds.length > 0);
    assert.ok(Array.isArray(family.validMeasurementIds));
    for (const id of family.measurementIds) {
      assert.ok(knownMeasurements.has(id), `${family.id} references unknown measurement ${id}`);
    }
    for (const id of family.validMeasurementIds) {
      assert.ok(family.measurementIds.includes(id), `${id} must be a member of ${family.id}`);
      assert.notEqual(knownMeasurements.get(id)?.validity, "invalid", `${family.id} consumed invalid evidence ${id}`);
    }
  }

  const consumedIds = evidenceIds(plan);
  assert.equal(plan.evidenceMeasurementCount, consumedIds.size, "evidenceMeasurementCount must count unique valid inputs");
  assert.ok(plan.evidenceMeasurementCount >= 12, "a supported frontal blend should use at least 12 valid measurements");
  const consumedKinds = new Set([...consumedIds].map((id) => knownMeasurements.get(id)?.kind));
  assert.ok(consumedKinds.size >= 3, `expected at least three evidence kinds, received ${[...consumedKinds].join(", ")}`);

  const regions = new Set(plan.actions.map((action) => action.region));
  assert.ok(regions.size >= 3, "compatible jaw, nose, chin, brow, and symmetry work should coexist in one plan");
  assert.ok(
    plan.actions.some((action) => action.directions.length > 1),
    "compatible direction signals should blend on a primitive instead of replacing one another",
  );
});

test("rich measurements, not the legacy eight-metric shortcut, determine harmony decisions", () => {
  const analysis = analyze({ asymmetric: true });
  const baseline = createMorphPlan(analysis, { harmony: 100, symmetry: 0, dimorphism: 0 });
  assert.ok(baseline.actions.length >= 2, "fixture must produce a multi-region harmony decision");
  assert.ok(baseline.evidenceMeasurementCount >= 8, "the harmony decision must have broad measurement support");

  const poisonedMetrics = clone(analysis);
  poisonedMetrics.metrics = {
    jawToFace: 8,
    noseToFace: 0.001,
    mouthToFace: 4,
    lowerThird: 7,
    pairedDeviation: 100,
    mouthToNose: 12,
    chinToJaw: 0.001,
    faceAspect: 9,
  };
  const withPoisonedMetrics = createMorphPlan(poisonedMetrics, { harmony: 100, symmetry: 0, dimorphism: 0 });
  assertComparablePlans(withPoisonedMetrics, baseline, "changing legacy summary metrics alone");

  const noEvidence = clone(analysis);
  noEvidence.measurements = noEvidence.measurements.map((measurement) => ({
    ...measurement,
    confidence: 0,
    editability: 0,
    validity: "invalid",
    reasons: [...measurement.reasons, "Synthetic evidence removal"],
  }));
  const rejected = createMorphPlan(noEvidence, { harmony: 100, symmetry: 0, dimorphism: 0 });
  assert.equal(rejected.actions.length, 0, "legacy metrics must not bypass an unavailable measurement registry");
  assert.equal(rejected.selectedCandidate, "identity");
});

test("one corrupted measurement cannot dominate a multi-family decision", () => {
  const analysis = analyze({ asymmetric: true });
  const baseline = createMorphPlan(analysis, { harmony: 100, symmetry: 0, dimorphism: 0 });
  assert.ok(baseline.actions.length >= 2);

  const corrupted = clone(analysis);
  const outlierId = "ratio.primary.width.outline.gonionProxy";
  corrupted.measurements = corrupted.measurements.map((measurement) => measurement.id === outlierId
    ? { ...measurement, value: 50, confidence: 1, editability: 1, validity: "valid" }
    : measurement);
  const outlierPlan = createMorphPlan(corrupted, { harmony: 100, symmetry: 0, dimorphism: 0 });

  assertComparablePlans(outlierPlan, baseline, "a single extreme ratio outlier");
  assert.ok(
    actionMagnitude(outlierPlan) <= actionMagnitude(baseline) * 1.25 + 0.02,
    "one input must not inflate the total edit budget",
  );
});

test("one valid measurement is insufficient to authorize a regional morph", () => {
  const analysis = analyze({ asymmetric: true });
  const oneMeasurementId = "ratio.primary.width.outline.gonionProxy";
  const isolated = clone(analysis);
  isolated.measurements = isolated.measurements.map((measurement) => measurement.id === oneMeasurementId
    ? measurement
    : {
        ...measurement,
        confidence: 0,
        editability: 0,
        validity: "invalid",
        reasons: [...measurement.reasons, "Synthetic family isolation"],
      });
  const plan = createMorphPlan(isolated, { harmony: 100, symmetry: 0, dimorphism: 0 });

  assert.ok(
    !plan.actions.some((action) => action.region === "Jaw"),
    "a single ratio must not authorize a jaw warp without independent corroboration and guardrail evidence",
  );
});

test("profile plans consume profile evidence but never perspective-confounded symmetry", () => {
  const rightAnalysis = analyze({ yaw: 58 });
  const leftAnalysis = analyze({ yaw: -58 });
  const mix = { harmony: 80, symmetry: 100, dimorphism: 65 };
  const right = createMorphPlan(rightAnalysis, mix);
  const left = createMorphPlan(leftAnalysis, mix);

  assert.match(rightAnalysis.pose.class, /profile/);
  assert.match(leftAnalysis.pose.class, /profile/);
  assert.ok(right.actions.length > 0 && left.actions.length > 0, "clean profiles should retain visible-side edits");
  for (const plan of [right, left]) {
    assert.ok(!plan.actions.some((action) => action.region === "Symmetry" || action.primitive === "paired-alignment"));
    assert.ok(
      [...evidenceIds(plan)].some((id) => id.includes(".profile.")),
      "a profile decision should consume pose-specific profile measurements",
    );
    assert.ok(
      ![...evidenceIds(plan)].some((id) => id.startsWith("symmetry.")),
      "invalid bilateral symmetry evidence must not leak into a profile plan",
    );
  }
  assertComparablePlans(left, right, "mirroring a supported profile pose");
});

test("small landmark jitter does not flicker actions or materially change their strength", () => {
  const mix = { harmony: 75, symmetry: 60, dimorphism: 50 };
  const baseline = createMorphPlan(analyze({ asymmetric: true }), mix);
  assert.ok(baseline.actions.length >= 3);

  for (let seed = 1; seed <= 6; seed += 1) {
    const jittered = createMorphPlan(analyze({ asymmetric: true, noiseSeed: seed }), mix);
    assertComparablePlans(jittered, baseline, `sub-pixel landmark jitter seed ${seed}`);
    assert.ok(
      Math.abs(jittered.evidenceMeasurementCount - baseline.evidenceMeasurementCount) <= 3,
      `sub-pixel jitter should not churn the evidence inventory (${baseline.evidenceMeasurementCount} -> ${jittered.evidenceMeasurementCount})`,
    );
  }
});

test("zero directions and locked regions remain hard planner gates", () => {
  const analysis = analyze({ asymmetric: true });
  const noOp = createMorphPlan(analysis, { harmony: 0, symmetry: 0, dimorphism: 0 });
  assert.equal(noOp.actions.length, 0);
  assert.equal(noOp.selectedCandidate, "identity");

  const locked = clone(analysis);
  locked.regionEditability = locked.regionEditability.map((entry) => entry.region === "Jaw"
    ? { ...entry, editable: false, reasons: ["Synthetic jaw lock"] }
    : entry);
  const lockedPlan = createMorphPlan(locked, { harmony: 100, symmetry: 100, dimorphism: 100 });
  assert.ok(!lockedPlan.actions.some((action) => action.region === "Jaw"));
  assert.match(lockedPlan.rejectedReasons.join(" "), /Jaw: Synthetic jaw lock/i);
});
