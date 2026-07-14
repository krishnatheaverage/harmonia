import assert from "node:assert/strict";
import test from "node:test";

import {
  MEASUREMENT_CATALOG,
  MEASUREMENT_COUNTS,
  SEMANTIC_LANDMARK_COUNT,
  analyzeFace,
  createMorphPlan,
} from "../lib/face-intelligence.ts";

function fixture(noseX = 0.5) {
  const landmarks = Array.from({ length: 478 }, (_, index) => {
    const angle = (index / 468) * Math.PI * 2;
    return { x: 0.5 + Math.cos(angle) * 0.18, y: 0.5 + Math.sin(angle) * 0.24, z: 0 };
  });
  const set = (index, x, y) => { landmarks[index] = { x, y, z: 0 }; };
  set(10, 0.5, 0.18); set(152, 0.5, 0.82);
  set(234, 0.28, 0.5); set(454, 0.72, 0.5);
  set(1, noseX, 0.46); set(2, noseX, 0.52);
  set(172, 0.34, 0.7); set(397, 0.66, 0.7);
  set(148, 0.46, 0.78); set(377, 0.54, 0.78);
  set(98, noseX - 0.05, 0.51); set(327, noseX + 0.05, 0.51);
  set(61, 0.41, 0.61); set(291, 0.59, 0.61);
  set(33, 0.36, 0.39); set(133, 0.45, 0.39);
  set(263, 0.64, 0.41); set(362, 0.55, 0.41);
  return {
    landmarks,
    blendshapes: {},
    transformationMatrix: {
      rows: 4,
      columns: 4,
      data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    },
  };
}

test("V2 registries stay versioned and internally bounded", () => {
  assert.equal(SEMANTIC_LANDMARK_COUNT, 104);
  assert.equal(MEASUREMENT_CATALOG.length, 408);
  assert.deepEqual(MEASUREMENT_COUNTS, {
    distance: 78,
    ratio: 210,
    angle: 36,
    shape: 20,
    symmetry: 40,
    context: 24,
    total: 408,
  });
  assert.equal(new Set(MEASUREMENT_CATALOG.map(({ id }) => id)).size, 408);
  assert.ok(MEASUREMENT_CATALOG.every((measurement) => measurement.consumerRules.length > 0));
});

test("pose changes which measurements are eligible", () => {
  const frontal = analyzeFace(fixture(0.5), 1000, 1200);
  const threeQuarter = analyzeFace(fixture(0.62), 1000, 1200);
  const profileFixture = fixture(0.72);
  profileFixture.landmarks[1].y = 0.5;
  const profile = analyzeFace(profileFixture, 1000, 1200);
  assert.equal(frontal.pose.class, "frontal");
  assert.match(threeQuarter.pose.class, /three-quarter/);
  assert.match(profile.pose.class, /profile/);
  assert.ok(frontal.validMeasurementCount > profile.validMeasurementCount);
  assert.ok(profile.measurements.filter(({ kind }) => kind === "symmetry").every(({ validity }) => validity === "invalid"));
});

test("directions blend into one plan and no-op remains available", () => {
  const analysis = analyzeFace(fixture(0.5), 1000, 1200);
  const blended = createMorphPlan(analysis, { harmony: 60, symmetry: 55, dimorphism: 35 });
  assert.ok(blended.actions.length > 0);
  assert.ok(blended.actions.some(({ directions }) => directions.includes("dimorphism")));
  assert.ok(blended.actions.some(({ directions }) => directions.includes("symmetry")));
  assert.ok(blended.actions.some(({ directions }) => directions.length > 1));
  assert.equal(blended.candidateCount, 4);

  const noOp = createMorphPlan(analysis, { harmony: 0, symmetry: 0, dimorphism: 0 });
  assert.equal(noOp.actions.length, 0);
  assert.equal(noOp.selectedCandidate, "identity");
});
