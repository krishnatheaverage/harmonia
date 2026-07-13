# Harmonia V1

Public app: https://krishnatheaverage.github.io/harmonia/

Harmonia is a private, browser-based portrait reshaping studio. It detects a
single clear face with MediaPipe landmarks and uses a bounded, fixed-topology
deformation mesh to move only pixels from the source image. It does not
generate skin, hair, shadows, color, or background content.

The app reports geometric measurements; it does not claim to calculate an
objective beauty score or a universal ideal face.

## Included

- local JPG, PNG, and WebP upload up to 20 MB
- camera-first automatic scan after explicit browser permission
- synchronized frame capture so landmarks and edited pixels always match
- blur, lighting, resolution, framing, pose, motion, and expression gates
- frontal and conservative three-quarter portrait support
- adaptive Facial Harmony, Symmetry, and Dimorphism morph plans
- geometric jaw, nose, mouth, lower-third, and paired-drift readouts
- MediaPipe's dense canonical face topology with a fixed outer boundary ring
- target smoothing plus flip, stretch, shear, bounds, and area-ratio checks
- automatic strength backoff; unsafe plans return the original geometry
- strength control, before/after preview, and face map
- PNG export without the diagnostic overlay
- no image upload API, persistence, analytics, or authentication

## Morph pipeline

1. Freeze one exact camera frame and detect its dense landmarks.
2. Reject frames that are blurry, poorly lit, moving, strongly posed, or non-neutral.
3. Measure pose-normalized geometry and generate small, feature-grouped targets.
4. Smooth targets across the canonical face mesh while holding a narrow outer ring fixed.
5. Reduce the edit until every triangle passes orientation, area, stretch, shear, and image-bound checks.
6. Replace the complete face region once and leave pixels outside the fixed boundary untouched.

## Research basis

The implementation follows the conservative parts of the published literature:
dense canonical face geometry from [MediaPipe Face Mesh](https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh),
small identity-preserving target changes from the [Facial Reshaping Operator](https://hubertshum.com/publications/eswa2021beautification/files/eswa2021beautification.pdf),
local shape regularization from [Laplacian surface editing](https://diglib.eg.org/items/ee43c2c6-4956-4ff0-9db1-c946306a5b99),
and explicit foldover prevention motivated by [Locally Injective Mappings](https://igl.ethz.ch/projects/LIM/).

## Run locally

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Validate

```bash
npm run lint
npm test
npm run build:pages
```

The face landmark model and MediaPipe WebAssembly runtime are both bundled under
`public/`, so camera frames and photos remain in the browser without a CDN.
