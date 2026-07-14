# Harmonia facial analysis and planning specification

Status: V2 implementation contract  
Schema version: `harmonia-face-v2`  
Planner schema version: `3.0.0-beta.1`
Semantic proxy count: **104**  
Measurement catalog count: **408**

## 1. Purpose and non-goals

This specification defines the explainable layer between face detection and the
pixel warp. Its purpose is to help the planner decide whether a region should be
preserved, cautiously adjusted, or locked for the current image. More signals do
not justify more edits: uncertainty, poor visibility, conflicting directions, or
an already coherent face must be able to produce a no-op.

The engine is a photographic geometry tool, not a medical, biometric, or
psychological assessment. Its Harmony, Symmetry, and Dimorphism controls are
editing directions, not objective measures of attractiveness, sex, gender,
health, ethnicity, or character. They must never be exposed as a person score.

The V2 engine does not generate pixels. It cannot create shadows, deepen eye
sockets, invent cheekbone or maxillary projection, remove substantial soft
tissue, synthesize hair or facial hair, reconstruct an occluded region, or
reliably infer underlying bone from one photograph.

## 2. Coordinate layers

The planner keeps three distinct coordinate layers. Mixing their meanings is a
schema error.

1. **Dense image mesh.** MediaPipe Face Mesh returns 468 three-dimensional face
   landmarks. Normalized x/y coordinates locate projected image points; z is a
   relative weak-perspective depth value, not millimetres. All 468 vertices and
   the canonical topology remain available to the warp.
2. **Semantic proxy layer.** `SEMANTIC_LANDMARKS` is Harmonia's versioned mapping
   of 104 planning names to mesh vertices or small vertex groups. A proxy may be
   a contour sample, midpoint, centroid, or curve extremum. “Semantic” means
   meaningful to this product's planner; it does **not** mean that MediaPipe
   officially names or validates the proxy as an anatomical landmark.
3. **Pose-normalized planning frame.** Measurements use image-normalized u/v,
   canonicalized x/y/z, or a visible-contour frame as declared by their
   primitive. The optional face transformation matrix and a 2D fallback estimate
   yaw, pitch, and roll. A transformation output improves normalization but does
   not turn a monocular image into a metric 3D scan.

MediaPipe documents a canonical face model, metric face-geometry space, and a
rigid transformation from the canonical model to the runtime face. Harmonia uses
that transform for normalization and pose context, while retaining fallbacks and
confidence penalties when it is absent or conflicts with the projected mesh.

Primary references: [MediaPipe Face Mesh](https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh),
[Face Landmarker result](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult),
and [Attention Mesh](https://research.google/pubs/attention-mesh-high-fidelity-face-mesh-prediction-in-real-time/).

## 3. Semantic proxy schema

The registry is the only source of truth for semantic proxy count and identity.
Renaming, remapping, adding, or deleting a proxy requires a schema-version
change. A proxy record has this shape:

```ts
type SemanticLandmark = {
  id: string;
  label: string;
  region: FaceRegion;
  side: "left" | "right" | "midline" | "bilateral";
  role: "anchor" | "contour" | "center" | "extremum";
  meshIndices: readonly number[];
  aggregation: "vertex" | "mean" | "midpoint" | "curve-extremum";
  visibleIn: readonly PoseFamily[];
};
```

The 104 proxies cover the face envelope and jaw contour; chin; nasal bridge,
root, tip, columella, alae, and nostril contours; inner and outer eye contours;
brow contours; lip and mouth contours; philtral and oral midline points; and a
small set of pose/reference centers. Hairline and ear proxies are deliberately
absent because the 468-point face mesh does not provide dependable coverage of
those structures. The full registry is exported as `SEMANTIC_LANDMARKS`, and
`SEMANTIC_LANDMARK_COUNT` must equal `104`.

The semantic layer does not replace the dense layer. Edits selected at semantic
proxies are distributed over the 468-vertex mesh, regularized, and then checked
against fixed-topology safety constraints.

## 4. Measurement primitives and catalog

`MEASUREMENT_CATALOG` contains exactly 408 versioned definitions. A definition is
metadata plus an evaluator; it is not automatically an edit target.

```ts
type MeasurementDefinition = {
  id: string;
  label: string;
  family: MeasurementFamily;
  primitive: MeasurementPrimitive;
  landmarkIds: readonly string[];
  denominatorId?: string;
  validPoses: readonly PoseFamily[];
  requiredSides: "none" | "visible" | "bilateral";
  plannerUse: "target" | "guardrail" | "context" | "preserve";
  minimumConfidence: number;
  editableRegions: readonly FaceRegion[];
};

type MeasurementPrimitive =
  | "distance-2d"
  | "distance-3d-relative"
  | "horizontal-span"
  | "vertical-span"
  | "signed-midline-offset"
  | "angle-2d"
  | "angle-3d-relative"
  | "curve-length"
  | "area-2d"
  | "ratio"
  | "pair-difference"
  | "pose-context";
```

Catalog families include facial envelope and thirds; orbital and brow shape;
nasal width, height, contour, and profile context; oral width, height, lip,
philtrum, and expression context; lower-face, jaw, and chin geometry; paired
differences; and pose/visibility context. Ratios use declared, anatomically local
denominators such as face width, visible face height, interpupillary proxy span,
nose width, mouth width, or lower-face height. The engine never constructs every
possible pairwise ratio: combinatorial ratios are highly correlated, unstable,
and difficult to interpret. “408 measurements” means the exact curated registry,
not 408 independent biological facts.

The versioned total has this exact, deterministic family breakdown:

| Catalog family | Count | Planner role |
|---|---:|---|
| Distance and span | 78 | Base lengths, widths, heights, contour spans, and local normalization inputs |
| Ratio | 210 | Declared numerator/denominator relationships with pose and denominator gates |
| Angle | 36 | Ordered profile, contour, eye, brow, nose, mouth, jaw, and chin angles |
| Shape | 20 | Curve, area, envelope, and proportional-shape summaries |
| Symmetry | 40 | Bilateral pair differences and midline drift; front or visibility-gated 3/4 only |
| Context | 24 | Pose, expression, quality, visibility, and preservation/guardrail signals |
| **Total** | **408** | |

IDs are generated deterministically from the family, source proxy or primitive,
and normalizer. The count and IDs therefore change only with an explicit catalog
version change.

The catalog is the machine-readable exhaustive list. Each record must have a
stable ID and must declare all fields above. CI rejects duplicate IDs, missing
semantic proxy references, catalog sizes other than 408, or definitions with no
pose or planner use.

### Primitive rules

- Distances and spans must be normalized before cross-image comparison. Pixel
  distances alone may only be used as local warp radii.
- Relative 3D measurements may compare the detector's internal z values within
  one face; they may not be labelled centimetres or millimetres.
- Bilateral pair differences require both sides to be visible and sufficiently
  supported. Mirroring an occluded side does not create evidence.
- Angles use an explicit vertex and ordered arms. Signed angles state their
  orientation in the definition.
- Curve length follows declared ordered proxy samples rather than the shortest
  straight-line distance.
- A ratio is valid only when both numerator and denominator are valid and the
  denominator exceeds its numerical floor.
- Context and guardrail measurements can veto or attenuate an edit but cannot
  pull a feature toward a target on their own.

## 5. Pose resolution and validity

`PoseEstimate` combines the facial transformation matrix, projected landmark
geometry, and their agreement. The class is one of `frontal`,
`three-quarter-left`, `three-quarter-right`, `profile-left`, `profile-right`, or
`unsupported`. Thresholds are soft: confidence decays near a boundary, and a
measurement carries its own validity rather than inheriting pose class alone.

| Measurement or edit family | Frontal | Moderate 3/4 | Clean profile |
|---|---:|---:|---:|
| Overall vertical proportions | yes | yes, confidence reduced | visible contour only |
| Bilateral symmetry / pair difference | yes | limited to well-visible pairs | no |
| Inter-eye and paired brow relationships | yes | reduced / visibility-gated | no |
| Nose width / alar balance | yes | limited | no |
| Nose profile angle / relative projection | context only | limited | yes, visible side |
| Mouth/lip vertical relations | yes | yes, expression-gated | visible contour only |
| Jaw width | yes | pose-normalized and bounded | no |
| Jaw/chin silhouette | yes | visible-side weighted | yes, visible side |
| Dimorphism direction | bounded | bounded and visibility-gated | silhouette-only subset |

Profile support is intentionally narrower than frontal support. In particular,
the planner must not interpret projective foreshortening as facial asymmetry or
estimate the hidden-side width from a single profile image. Research on
large-pose alignment likewise treats landmark visibility as a first-class
problem rather than assuming that all 2D points remain observable:
[Face Alignment Across Large Poses](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhu_Face_Alignment_Across_CVPR_2016_paper.pdf)
and [3D landmark visibility](https://arxiv.org/abs/1506.03799).

## 6. Confidence

Confidence answers “how much evidence supports this value in this image?” It is
not an attractiveness probability. Every `MeasurementValue` records its numeric
value, validity, confidence, and a reason when invalid.

```ts
type MeasurementValue = {
  definitionId: string;
  value: number | null;
  normalizedValue: number | null;
  valid: boolean;
  confidence: number; // [0, 1]
  invalidReason?: string;
};
```

The implementation combines bounded factors rather than letting a single good
signal erase a hard failure:

```text
measurement confidence =
  quality × poseApplicability × visibility × landmarkSupport
  × expressionCompatibility × transformAgreement
```

Factors are clamped to `[0, 1]`. A hard precondition—missing required side,
unsupported pose, invalid denominator, landmark outside the face crop, or severe
occlusion—sets validity to false. Blur, exposure, crop margin, face resolution,
pose distance, transform/2D disagreement, local mesh spacing, and relevant
blendshape activity attenuate confidence. The UI may summarize confidence but
must retain per-measurement reasons for planner diagnostics.

## 7. Region editability

Editability answers a different question: “even if this relationship is
measurable, can moving this region be done safely with source pixels?” It is
computed per face region and movement axis.

```ts
type RegionEditability = {
  region: FaceRegion;
  score: number; // [0, 1]
  allowedAxes: readonly ("u" | "v" | "normal")[];
  maxDisplacement: number; // fraction of local face width
  locked: boolean;
  reasons: readonly string[];
};
```

Editability includes pose visibility, local landmark support, image margin,
expression, proximity to the protected face boundary, occlusion/coverage proxy,
and the renderer's distortion budget. The plan must use the lower of analysis
editability and geometric safety backoff. Unsupported operations—such as moving
hairline/ears, creating profile depth not present in the photo, or changing a
hidden side—are locked regardless of requested direction strength.

`maxDisplacement` is an auditable per-region handle cap expressed as a fraction
of detected face width. Capture quality and local editability may reduce it, but
the planner and renderer may never raise it above these V2 ceilings:

| Region | Maximum handle displacement |
|---|---:|
| Jaw | 2.4% of face width |
| Chin | 1.8% of face width |
| Face shape / envelope | 1.8% of face width |
| Nose | 1.0% of face width |
| Lips | 1.0% of face width |
| Symmetry alignment | 1.0% of face width |
| Brows | 0.8% of face width |

These are motion ceilings, not targets. A plan can request less, and an
ineligible region remains locked even when global Strength is `100`.

## 8. Harmony rules

A harmony rule is a transparent, bounded planner policy. It declares inputs,
pose scope, deadband, suggested primitive, compatible and conflicting rules, and
maximum influence. It does not directly mutate landmarks.

```ts
type HarmonyRule = {
  id: string;
  measurementIds: readonly string[];
  direction: "harmony" | "symmetry" | "dimorphism";
  validPoses: readonly PoseFamily[];
  deadband: readonly [number, number];
  primitiveId: string;
  maxInfluence: number;
  compatibleWith: readonly string[];
  conflictsWith: readonly string[];
};
```

Rules prefer broad, uncertainty-aware ranges and within-face coherence. A value
inside its deadband produces no pressure. A value outside it produces a proposal
only when the excess is larger than measurement uncertainty. High-confidence,
distinctive geometry is a preservation signal unless multiple independent rules
support a small change. Population norms may calibrate broad plausibility bounds,
but no single dataset is treated as a universal template; published norm sources
can differ materially ([3D Facial Norms](https://pubmed.ncbi.nlm.nih.gov/26492185/),
[norm-source comparison](https://pubmed.ncbi.nlm.nih.gov/31053285/)).

### Active evidence registry

The implemented planner has five calibrated Harmony evidence families: lower
outline width, lower-face height, nasal width, mouth width, and brow-to-eye
spacing. Each family declares its exact catalog IDs, pose scope, broad reference
envelope, independent cue groups, regional contour guardrails, primitive sign,
and maximum influence. A cue is eligible only when its measurement is valid and
has confidence of at least `0.55`; one cue can never authorize a regional warp.

For each family the planner:

1. groups correlated left/right or shared-normalizer cues so they count once;
2. takes the median within a correlated group;
3. requires at least two independent active groups with at least two-thirds
   directional agreement;
4. uses median reliability and a conservative lower-quartile deviation rather
   than the largest deviation;
5. preserves the region when evidence is inside-band, insufficient, or in
   conflict; and
6. exposes every considered and valid measurement ID on the returned plan.

Symmetry is a separate frontal-only family. It consumes all 40 axial and
vertical paired residuals, applies operation-specific natural-asymmetry
deadbands, and requires at least three independent facial roles. The Angularity
direction is authorized by pose-valid jaw, chin, brow, or profile-silhouette
angles, curves, and distances; those measurements authorize a restrained style
vector but are not interpreted as an attractiveness or gender inference.

The legacy `FaceAnalysis.metrics` summary remains temporarily available for UI
compatibility and diagnostics, but `createMorphPlan` does not use it to select
or size edits. Removing the measurement registry therefore produces a true
no-op even if the legacy summary contains extreme values.

## 9. Morph primitives

Planner primitives are small semantic proposals expressed in the pose-normalized
frame. Current V2 primitives cover paired or unilateral contour translation,
bounded horizontal/vertical scale around an anchor, midline recentering,
visible-side silhouette adjustment, and curve-preserving feature translation.
Each primitive declares affected proxies, falloff radius, allowed axes, maximum
normalized displacement, fixed/protected neighbors, and incompatibilities.

Primitives do not paint, inpaint, clone, relight, sharpen, or synthesize depth.
The renderer converts a selected primitive to dense-mesh displacement, applies
neighborhood smoothing with a fixed outer ring, rasterizes the original source
triangles, and validates orientation, area, stretch, shear, and bounds.

## 10. Joint planner

`DEFAULT_DIRECTION_MIX` initializes three independent weights. The UI exposes all
three at once; selecting one never discards the other two.

```ts
type DirectionMix = {
  harmony: number;    // [0, 1]
  symmetry: number;   // [0, 1]
  dimorphism: number; // [0, 1]
};

type PlannedEdit = {
  primitiveId: string;
  region: FaceRegion;
  confidence: number;
  expectedBenefit: number;
  displacementU: number;
  displacementV: number;
  supportingRuleIds: readonly string[];
  guardrailIds: readonly string[];
};

type PlannerEvidenceFamily = {
  id: string;
  direction: "harmony" | "symmetry" | "dimorphism";
  region: FaceRegion;
  primitive: MorphPrimitive;
  signal: number;     // [-1, 1]
  confidence: number; // [0, 1]
  agreement: number;  // [0, 1]
  status: "supported" | "inside-band" | "insufficient-evidence" | "evidence-conflict";
  measurementIds: readonly string[];
  validMeasurementIds: readonly string[];
  maxInfluence: number;
};

type MorphPlan = {
  schemaVersion: "3.0.0-beta.1";
  actions: readonly PlannedEdit[];
  evidenceFamilies: readonly PlannerEvidenceFamily[];
  evidenceMeasurementCount: number;
  selectedCandidate: "identity" | "light" | "balanced" | "full";
};
```

For every valid rule, the planner computes:

```text
priority = deviationBeyondDeadband
  × directionWeight
  × measurementConfidence
  × regionEditability
  × expectedBenefit
  × compatibilityPenalty
```

The planner then:

1. filters invalid or uncertain evidence;
2. forces pose-inapplicable direction components to zero (notably profile
   symmetry);
3. converts supported rules to bounded primitive candidates;
4. merges compatible proposals that affect the same region;
5. rejects contradictory candidates rather than averaging them into a larger
   arbitrary move;
6. limits the number of simultaneously active regions and total displacement;
7. records an evidence label (`light`, `balanced`, or `full`) for explanation,
   without using that label as a hidden render multiplier; and
8. asks the geometry validator for the strongest safe scale independently for
   each affected region, preserving the original geometry for a region when no
   non-zero scale passes.

A valid result may therefore contain zero edits. `createMorphPlan` must return
that no-op explicitly with reasons such as “inside deadbands,” “profile symmetry
unavailable,” “low local confidence,” “region locked,” or “geometric validation
failed.” Increasing a UI strength control changes the allowed cap; it does not
make invalid evidence valid.

### Strength and geometry-backoff contract

The UI exposes global Strength directly on a `0–100` scale. `0` must render the
unchanged source. `100` requests the complete already-bounded plan. Intermediate
values interpolate linearly before geometry validation. There is no additional
candidate-tier, confidence-tier, or mode-specific multiplier in the renderer;
all evidence weighting is already represented in the planned actions.

Geometry validation is the only post-plan attenuator. It evaluates one region at
a time against the safe geometry already accepted, then binary-searches only an
unsafe region toward zero. A nose constraint therefore cannot silently shrink a
jaw edit that already passed. The result reports both the requested Strength and
the applied per-region scales so geometric backoff is visible rather than hidden.

Triangle orientation and output image bounds remain hard constraints: a
foldover, orientation reversal, or target outside the image always fails,
regardless of requested Strength. Area, edge-length, stretch, and shear checks
bound meaningful mesh triangles; numerically tiny mesh slivers cannot veto an
otherwise safe entire-face plan. If no non-zero regional scale satisfies the
hard constraints, that region is rendered as identity.

## 11. Analysis result schema

`analyzeFace` returns one auditable object. UI copy should summarize it without
hiding limitations.

```ts
type FaceIntelligence = {
  schemaVersion: "harmonia-face-v2";
  pose: PoseEstimate;
  semanticLandmarkCount: 104;
  measurementCatalogCount: 408;
  validMeasurementCount: number;
  measurements: Readonly<Record<string, MeasurementValue>>;
  editability: Readonly<Record<FaceRegion, RegionEditability>>;
  qualityConfidence: number;
  preservedRegions: readonly FaceRegion[];
  warnings: readonly string[];
};
```

The capture quality gate, semantic proxy resolution, catalog evaluation,
confidence computation, editability computation, rules, and morph planning are
separate phases. This allows a measurement to remain visible for diagnostics
without accidentally becoming an edit command.

## 12. Validation requirements

Automated tests must verify:

- `SEMANTIC_LANDMARK_COUNT === 104` and registry IDs are unique;
- `MEASUREMENT_CATALOG.length === 408`, IDs are unique, and every proxy reference
  exists;
- every measurement declares pose validity, planner use, and confidence floor;
- profile analysis invalidates bilateral symmetry and hidden-side width targets;
- low quality or editability can produce an explicit no-op;
- blended direction weights can jointly support compatible edits without
  exceeding per-region and total budgets; and
- final dense targets still pass renderer foldover, stretch, shear, area, and
  image-bound checks.

Visual QA must use front, moderate three-quarter, and clean profile examples plus
blur, crop, expression, occlusion, and unusual-proportion stress cases. A useful
test set deliberately includes faces that do not resemble the population samples
used to set broad guardrails.

## 13. Known limitations

- The mesh has no dependable hairline or ear coverage.
- Monocular z is relative and weak-perspective; results are not metric millimetres
  and do not measure true bone projection.
- Occlusion and extreme yaw can move projected landmarks away from visible
  anatomy. Profile editing is therefore a restricted visible-contour mode.
- A single image cannot separate anatomy, expression, lens distortion, camera
  distance, and lighting perfectly.
- Normative facial datasets vary by capture method and sampled population. Their
  values are contextual guardrails, not universal beauty targets.
- Pixel warping cannot create anatomy, texture, lighting, shadows, hair, skin, or
  hidden facial regions. It can only relocate source pixels.
- The system does not infer identity traits, biological sex, health, personality,
  or attractiveness from geometry.

The conservative failure mode for all of these limitations is a smaller edit, a
locked region, or the unchanged original image.
