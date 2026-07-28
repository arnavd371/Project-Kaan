# Kaan web (`web/`)

Public Next.js website for Project Kaan. Part of the monorepo:
https://github.com/arnavd371/Project-Kaan

Live demo: https://kaan-web.vercel.app

## Features
- Detect clean vs rice weevil / lesser grain borer / red flour beetle
- Runs the trained INT8 CNN (`project-kaan.onnx`) in the browser via ONNX Runtime Web
- Mel spectrogram preprocessing matched to the Python training pipeline
- Severity estimate from acoustic signal density
- English, Hindi, Marathi, Punjabi, Telugu
- Guided tour for first-time users
- Colour + symbol results for accessibility

## Local development
From the monorepo root:
```bash
cd web
npm install
npm run dev
```

## Deploy
Set the Vercel root directory to `web`, or deploy from this folder:
```bash
npm run build
npx vercel --prod
```

## Mobile
See [MOBILE.md](MOBILE.md). Capacitor project folders are `android/` and `ios/` under `web/`.

## Licence
Apache License 2.0. Copyright 2026 Arnav Dhiman.
