# Harmonia V1

Harmonia is a private, browser-based portrait reshaping studio. It detects a
single face with MediaPipe landmarks and uses a bounded triangular deformation
mesh to move only pixels from the source image. It does not generate skin,
hair, shadows, color, or background content.

## Included

- local JPG, PNG, and WebP upload up to 20 MB
- one-face landmark detection with frontal, three-quarter, and profile support
- Natural Harmony, Chadlite, Refined, and Symmetry morph plans
- text prompt routing, strength control, before/after preview, and face map
- PNG export without the diagnostic overlay
- no image upload API, persistence, analytics, or authentication

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
```

The face landmark model is included under `public/models`. MediaPipe's WebAssembly
runtime is loaded from jsDelivr on first use; portrait pixels remain in the
browser and are never sent with that request.
