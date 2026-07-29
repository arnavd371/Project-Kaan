# Kaan mobile apps (Play Store & App Store)

Kaan wraps the **same pi.website design** (cream background, ink type, Libre Baskerville,
IBM Plex Mono, black-border panels and timeline) in [Capacitor](https://capacitorjs.com/)
for Android and iOS. No separate mobile skin. Copy has no em dashes.

Package ID: `com.arnavdhiman.kaan`

## Prerequisites

- Node.js 20+
- **Android:** Android Studio (SDK 35+, build tools)
- **iOS:** full Xcode from the Mac App Store (Command Line Tools alone are not enough), CocoaPods optional (SPM is used by current Capacitor iOS template)

## Build & sync

```bash
npm install
npm run build:mobile   # next static export -> out/ then cap sync
```

Open native IDEs:

```bash
npm run android   # opens Android Studio
npm run ios       # opens Xcode (requires full Xcode)
```

## Android (Play Store)

1. Install Android Studio and accept SDK licenses.
2. `npm run android`
3. In Android Studio: **Build > Generate Signed Bundle / APK** → Android App Bundle (`.aab`).
4. Create a Play Console listing (app name: **Kaan**, category: Tools / Agriculture).
5. Upload the `.aab`, complete Data safety (audio processed on-device; no account required), content rating, and store listing screenshots.
6. Privacy policy URL: use your GitHub README or a short page stating audio stays on device.

Version is set in `android/app/build.gradle` (`versionCode` / `versionName`).

## iOS (App Store)

1. Install **Xcode** (not only Command Line Tools).
2. `npm run ios`
3. In Xcode: set Team / signing for `com.arnavdhiman.kaan`.
4. Archive → Distribute App → App Store Connect.
5. Complete App Privacy (microphone / media library only if you enable recording; file upload uses the photo/files picker).

## Notes for reviewers

- Inference runs the INT8 CNN in the WebView (ONNX Runtime). First launch may download the WASM runtime from the CDN unless later bundled.
- IRRI sample WAVs are packaged under `public/samples/` for demos.
- License: Apache License 2.0 (see `LICENSE`).
