import type {
  DirectionMix,
  FaceAnalysis,
  FaceRegion,
  MeasurementValue,
  MorphPrimitive,
  PoseClass,
} from "./face-intelligence.ts";

export const PLANNER_SCHEMA_VERSION = "3.0.0-beta.1";

export type PlannerEvidenceStatus =
  | "supported"
  | "inside-band"
  | "insufficient-evidence"
  | "evidence-conflict";

export type PlannerEvidenceFamily = {
  id: string;
  label: string;
  direction: keyof DirectionMix;
  region: FaceRegion;
  primitive: MorphPrimitive;
  signal: number;
  confidence: number;
  agreement: number;
  status: PlannerEvidenceStatus;
  measurementIds: string[];
  validMeasurementIds: string[];
  /** Maximum normalized primitive pressure before direction and reliability weighting. */
  maxInfluence: number;
  rationale: string;
};

type EvidenceCue = {
  id: string;
  group: string;
  low: number;
  high: number;
};

export type PlannerRuleDefinition = {
  id: string;
  label: string;
  direction: "harmony";
  region: FaceRegion;
  primitive: MorphPrimitive;
  poses: PoseClass[];
  cues: EvidenceCue[];
  minimumIndependentGroups: number;
  primitiveEffectSign: 1 | -1;
  maxInfluence: number;
  guardrailRegions: string[];
};

const FRONT_AND_THREE_QUARTER: PoseClass[] = [
  "frontal",
  "three-quarter-left",
  "three-quarter-right",
];

// These are intentionally broad engineering/style envelopes. They are not a
// universal attractiveness scale. Every directional rule requires independent
// corroboration; the rest of the catalog can support or preserve, but cannot
// pull geometry toward an uncalibrated target.
export const PLANNER_RULES: PlannerRuleDefinition[] = [
  {
    id: "harmony.lower-outline-width",
    label: "Lower-outline proportion",
    direction: "harmony",
    region: "Jaw",
    primitive: "jaw-width",
    poses: FRONT_AND_THREE_QUARTER,
    cues: [
      { id: "ratio.primary.width.outline.gonionProxy", group: "gonion-width", low: 0.6, high: 0.74 },
      { id: "ratio.primary.width.outline.jawAngleLower", group: "lower-angle-width", low: 0.52, high: 0.7 },
      { id: "ratio.primary.width.outline.jawMid", group: "jaw-mid-width", low: 0.4, high: 0.5 },
      { id: "ratio.primary.width.outline.chinLateral", group: "chin-width", low: 0.13, high: 0.22 },
    ],
    minimumIndependentGroups: 2,
    primitiveEffectSign: 1,
    maxInfluence: 0.82,
    guardrailRegions: ["jaw", "chin", "outline"],
  },
  {
    id: "harmony.lower-face-height",
    label: "Lower-face vertical balance",
    direction: "harmony",
    region: "Chin",
    primitive: "chin-length",
    poses: ["frontal", "three-quarter-left", "three-quarter-right"],
    cues: [
      { id: "ratio.primary.midline.lower_lip_chin", group: "lip-to-chin-face", low: 0.34, high: 0.47 },
      { id: "ratio.cross.midline.lower_lip_chin", group: "lip-to-chin-jaw", low: 0.43, high: 0.64 },
      { id: "ratio.partition.midline.lower_lip_chin", group: "lip-to-chin-lower-face", low: 0.46, high: 0.65 },
    ],
    minimumIndependentGroups: 2,
    primitiveEffectSign: 1,
    maxInfluence: 0.7,
    guardrailRegions: ["chin", "jaw", "frame"],
  },
  {
    id: "harmony.nasal-width",
    label: "Nasal-width coherence",
    direction: "harmony",
    region: "Nose",
    primitive: "nose-width",
    poses: FRONT_AND_THREE_QUARTER,
    cues: [
      { id: "ratio.primary.width.nose.alarWidest", group: "alar-face", low: 0.21, high: 0.31 },
      { id: "ratio.cross.width.nose.alarWidest", group: "alar-intercanthal", low: 0.85, high: 1.15 },
      { id: "ratio.primary.width.nose.nostrilLateral", group: "nostril-face", low: 0.19, high: 0.29 },
      { id: "ratio.primary.width.nose.alarBase", group: "alar-base-face", low: 0.065, high: 0.12 },
    ],
    minimumIndependentGroups: 2,
    primitiveEffectSign: -1,
    maxInfluence: 0.68,
    guardrailRegions: ["nose", "profile"],
  },
  {
    id: "harmony.mouth-width",
    label: "Mouth-width coherence",
    direction: "harmony",
    region: "Lips",
    primitive: "mouth-width",
    poses: FRONT_AND_THREE_QUARTER,
    cues: [
      { id: "ratio.primary.width.mouth.commissure", group: "mouth-face", low: 0.27, high: 0.42 },
      { id: "ratio.cross.width.mouth.commissure", group: "mouth-nose", low: 1.35, high: 1.95 },
      { id: "ratio.primary.width.mouth.innerCommissure", group: "inner-mouth-face", low: 0.23, high: 0.36 },
      { id: "ratio.cross.width.mouth.innerCommissure", group: "inner-mouth-mouth", low: 0.72, high: 1.15 },
    ],
    minimumIndependentGroups: 2,
    primitiveEffectSign: 1,
    maxInfluence: 0.5,
    guardrailRegions: ["mouth", "nose"],
  },
  {
    id: "harmony.brow-height",
    label: "Brow-to-eye spacing",
    direction: "harmony",
    region: "Brows",
    primitive: "brow-height",
    poses: FRONT_AND_THREE_QUARTER,
    cues: [
      { id: "ratio.primary.side.brow_eye_inner.right", group: "brow-inner", low: 0.24, high: 0.43 },
      { id: "ratio.primary.side.brow_eye_inner.left", group: "brow-inner", low: 0.24, high: 0.43 },
      { id: "ratio.primary.side.brow_eye_mid.right", group: "brow-mid", low: 0.26, high: 0.46 },
      { id: "ratio.primary.side.brow_eye_mid.left", group: "brow-mid", low: 0.26, high: 0.46 },
      { id: "ratio.primary.side.brow_eye_outer.right", group: "brow-outer", low: 0.22, high: 0.4 },
      { id: "ratio.primary.side.brow_eye_outer.left", group: "brow-outer", low: 0.22, high: 0.4 },
    ],
    minimumIndependentGroups: 2,
    primitiveEffectSign: 1,
    maxInfluence: 0.38,
    guardrailRegions: ["brow", "eye"],
  },
];

const clamp = (value: number, minimum = 0, maximum = 1) =>
  Math.max(minimum, Math.min(maximum, value));

function reliability(measurement: MeasurementValue) {
  return Math.sqrt(clamp(measurement.confidence) * clamp(measurement.editability));
}

function bandSignal(value: number, low: number, high: number) {
  const range = Math.max(high - low, 0.001);
  if (value < low) return clamp((low - value) / range);
  if (value > high) return -clamp((value - high) / range);
  return 0;
}

function median(values: number[]) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function conservativeMagnitude(values: number[]) {
  if (!values.length) return 0;
  const sorted = values.map(Math.abs).sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * 0.25)];
}

function usefulGuardrails(
  measurements: MeasurementValue[],
  regions: string[],
) {
  return measurements
    .filter((measurement) =>
      regions.includes(measurement.region) &&
      measurement.plannerUse === "guardrail" &&
      ["angle", "shape", "distance"].includes(measurement.kind),
    )
    .sort((a, b) => b.confidence * b.editability - a.confidence * a.editability)
    .slice(0, 6);
}

function harmonyFamily(
  analysis: FaceAnalysis,
  rule: PlannerRuleDefinition,
): PlannerEvidenceFamily | null {
  if (!rule.poses.includes(analysis.pose.class)) return null;
  const byId = new Map(analysis.measurements.map((measurement) => [measurement.id, measurement]));
  const guardrails = usefulGuardrails(analysis.measurements, rule.guardrailRegions);
  const measurementIds = [...new Set([
    ...rule.cues.map((cue) => cue.id),
    ...guardrails.map((measurement) => measurement.id),
  ])];
  const validGuardrails = guardrails.filter((measurement) =>
    measurement.validity !== "invalid" && measurement.confidence >= 0.45,
  );
  const groupValues = new Map<string, Array<{ signal: number; confidence: number; id: string }>>();
  for (const cue of rule.cues) {
    const measurement = byId.get(cue.id);
    if (
      !measurement ||
      measurement.validity !== "valid" ||
      measurement.confidence < 0.55 ||
      measurement.value === null ||
      !Number.isFinite(measurement.value)
    ) continue;
    const values = groupValues.get(cue.group) ?? [];
    values.push({
      signal: bandSignal(measurement.value, cue.low, cue.high),
      confidence: reliability(measurement),
      id: measurement.id,
    });
    groupValues.set(cue.group, values);
  }

  const groups = [...groupValues.values()].map((values) => ({
    signal: median(values.map((value) => value.signal)),
    confidence: values.reduce((sum, value) => sum + value.confidence, 0) / values.length,
    ids: values.map((value) => value.id),
  }));
  const active = groups.filter((group) => Math.abs(group.signal) >= 0.035);
  const positive = active.filter((group) => group.signal > 0);
  const negative = active.filter((group) => group.signal < 0);
  const dominant = positive.length >= negative.length ? positive : negative;
  const opposing = dominant === positive ? negative : positive;
  const agreement = active.length ? dominant.length / active.length : 1;
  const enoughGroups = dominant.length >= rule.minimumIndependentGroups;
  const conflict = opposing.length > 0 && agreement < 0.67;
  // A single high-confidence outlier must not raise the authorization budget
  // for its whole family. Median reliability is stable to one corrupted cue.
  const targetConfidence = median(groups.map((group) => group.confidence));
  const guardrailConfidence = validGuardrails.length
    ? validGuardrails.reduce((sum, measurement) => sum + reliability(measurement), 0) / validGuardrails.length
    : targetConfidence;
  const confidence = clamp(targetConfidence * 0.78 + guardrailConfidence * 0.22);
  const validMeasurementIds = [...new Set([
    ...groups.flatMap((group) => group.ids),
    ...validGuardrails.map((measurement) => measurement.id),
  ])];

  let status: PlannerEvidenceStatus = "inside-band";
  if (conflict) status = "evidence-conflict";
  else if (!enoughGroups) status = active.length ? "insufficient-evidence" : "inside-band";
  else status = "supported";
  const sign = dominant[0]?.signal && Math.sign(dominant[0].signal) || 0;
  const signal = status === "supported"
    ? clamp(conservativeMagnitude(dominant.map((group) => group.signal)), 0, 1) * sign * rule.primitiveEffectSign
    : 0;

  return {
    id: rule.id,
    label: rule.label,
    direction: rule.direction,
    region: rule.region,
    primitive: rule.primitive,
    signal,
    confidence,
    agreement,
    status,
    measurementIds,
    validMeasurementIds,
    maxInfluence: rule.maxInfluence,
    rationale: status === "supported"
      ? `${dominant.length} independent measurement groups agreed; ${validGuardrails.length} contour checks supported the region.`
      : status === "evidence-conflict"
        ? "Independent measurement groups pointed in conflicting directions, so this edit is preserved."
        : status === "insufficient-evidence"
          ? "Only one independent measurement group cleared its uncertainty band."
          : "All corroborated measurements remained inside the broad reference envelope.",
  };
}

function symmetryFamily(analysis: FaceAnalysis): PlannerEvidenceFamily | null {
  if (analysis.pose.class !== "frontal") return null;
  const measurements = analysis.measurements.filter((measurement) => measurement.id.startsWith("symmetry."));
  const valid = measurements.filter((measurement) =>
    measurement.validity === "valid" &&
    measurement.value !== null &&
    measurement.confidence >= 0.55,
  );
  const groups = new Map<string, number[]>();
  for (const measurement of valid) {
    const [, operation, ...role] = measurement.id.split(".");
    const deadband = operation === "axial" ? 0.005 : 0.0035;
    const excess = clamp(((measurement.value ?? 0) - deadband) / 0.012);
    if (excess <= 0.02) continue;
    const id = role.join(".");
    const values = groups.get(id) ?? [];
    values.push(excess);
    groups.set(id, values);
  }
  const activeGroups = [...groups.values()].map((values) => median(values));
  const status: PlannerEvidenceStatus = activeGroups.length >= 3 ? "supported" : activeGroups.length ? "insufficient-evidence" : "inside-band";
  const confidence = valid.length
    ? valid.reduce((sum, measurement) => sum + reliability(measurement), 0) / valid.length
    : 0;
  return {
    id: "symmetry.multi-region-residual",
    label: "Multi-region paired residual",
    direction: "symmetry",
    region: "Symmetry",
    primitive: "paired-alignment",
    signal: status === "supported" ? clamp(conservativeMagnitude(activeGroups)) : 0,
    confidence,
    agreement: 1,
    status,
    measurementIds: measurements.map((measurement) => measurement.id),
    validMeasurementIds: valid.map((measurement) => measurement.id),
    maxInfluence: 0.78,
    rationale: status === "supported"
      ? `${activeGroups.length} independent paired roles exceeded the natural-asymmetry deadband.`
      : "Fewer than three independent paired roles exceeded the natural-asymmetry deadband.",
  };
}

type StructuralFamily = {
  id: string;
  label: string;
  region: FaceRegion;
  primitive: MorphPrimitive;
  measurementIds: string[];
  maxInfluence: number;
};

const FRONTAL_DIMORPHISM_FAMILIES: StructuralFamily[] = [
  {
    id: "dimorphism.jaw-structure",
    label: "Jaw contour support",
    region: "Jaw",
    primitive: "jaw-width",
    measurementIds: [
      "angle.jaw_angle.right", "angle.jaw_angle.left",
      "angle.jaw_lower.right", "angle.jaw_lower.left",
      "shape.curvature.jaw.right", "shape.curvature.jaw.left",
      "distance.side.upper_jaw.right", "distance.side.upper_jaw.left",
    ],
    maxInfluence: 0.62,
  },
  {
    id: "dimorphism.chin-structure",
    label: "Chin transition support",
    region: "Chin",
    primitive: "chin-length",
    measurementIds: [
      "angle.chin_transition.right", "angle.chin_transition.left",
      "angle.prechin.right", "angle.prechin.left",
      "distance.midline.lower_lip_chin",
    ],
    maxInfluence: 0.3,
  },
  {
    id: "dimorphism.brow-structure",
    label: "Brow contour support",
    region: "Brows",
    primitive: "brow-height",
    measurementIds: [
      "angle.brow_arch.right", "angle.brow_arch.left",
      "angle.orbital_brow.right", "angle.orbital_brow.left",
      "shape.curvature.brow.right", "shape.curvature.brow.left",
    ],
    maxInfluence: 0.25,
  },
];

const PROFILE_DIMORPHISM_FAMILIES: StructuralFamily[] = [
  {
    id: "dimorphism.profile-jaw-silhouette",
    label: "Visible profile jaw support",
    region: "Jaw",
    primitive: "jaw-width",
    measurementIds: [
      "distance.profile.nasion_chin", "distance.profile.tip_chin",
      "distance.profile.glabella_chin", "distance.profile.bridge_chin",
      "angle.profile_convexity.right", "angle.profile_convexity.left",
    ],
    maxInfluence: 0.55,
  },
  {
    id: "dimorphism.profile-chin-silhouette",
    label: "Visible profile chin support",
    region: "Chin",
    primitive: "chin-length",
    measurementIds: [
      "distance.profile.lower_lip_chin", "distance.profile.tip_chin",
      "distance.profile.nasion_chin", "distance.profile.bridge_chin",
    ],
    maxInfluence: 0.28,
  },
];

function structuralFamily(
  analysis: FaceAnalysis,
  definition: StructuralFamily,
): PlannerEvidenceFamily {
  const byId = new Map(analysis.measurements.map((measurement) => [measurement.id, measurement]));
  const valid = definition.measurementIds
    .map((id) => byId.get(id))
    .filter((measurement): measurement is MeasurementValue => Boolean(
      measurement && measurement.validity !== "invalid" && measurement.confidence >= 0.45,
    ));
  const confidence = valid.length
    ? valid.reduce((sum, measurement) => sum + reliability(measurement), 0) / valid.length
    : 0;
  const status: PlannerEvidenceStatus = valid.length >= 3 ? "supported" : "insufficient-evidence";
  return {
    id: definition.id,
    label: definition.label,
    direction: "dimorphism",
    region: definition.region,
    primitive: definition.primitive,
    signal: status === "supported" ? 1 : 0,
    confidence,
    agreement: status === "supported" ? 1 : 0,
    status,
    measurementIds: definition.measurementIds,
    validMeasurementIds: valid.map((measurement) => measurement.id),
    maxInfluence: definition.maxInfluence,
    rationale: status === "supported"
      ? `${valid.length} pose-valid contour measurements support a restrained style adjustment.`
      : "The pose did not expose enough independent contour measurements for this adjustment.",
  };
}

export function buildPlannerEvidence(analysis: FaceAnalysis) {
  const families: PlannerEvidenceFamily[] = [];
  for (const rule of PLANNER_RULES) {
    const family = harmonyFamily(analysis, rule);
    if (family) families.push(family);
  }
  const symmetry = symmetryFamily(analysis);
  if (symmetry) families.push(symmetry);
  const structural = analysis.pose.class.includes("profile")
    ? PROFILE_DIMORPHISM_FAMILIES
    : FRONTAL_DIMORPHISM_FAMILIES;
  families.push(...structural.map((definition) => structuralFamily(analysis, definition)));
  return families;
}
