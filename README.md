# Harmonia V1

Harmonia is a private, browser-based portrait reshaping studio. It detects a
single face with MediaPipe landmarks and uses a bounded triangular deformation
mesh to move only pixels from the source image. It does not generate skin,
hair, shadows, color, or background content.

## Included

- local JPG, PNG, and WebP upload up to 20 MB
- one-face landmark detection with frontal, three-quarter, and profile support
- Harmony and Angularity morph plans
- strength control, before/after preview, and face map
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

The face landmark model and MediaPipe WebAssembly runtime are both bundled under
`public/`, so the full photo pipeline runs locally in the browser without a CDN.
