# Kaan Android app

Kaan wraps the **same website design** (cream `#f7f6f2`, ink type, Libre Baskerville,
IBM Plex Mono, black-border panels and timeline) in [Capacitor](https://capacitorjs.com/).
No separate mobile skin.

Package ID: `com.arnavdhiman.kaan`  
Version: `3.1.1` (`versionCode` 311)

## What works on Android

- Intro + Detect / How / About (same UI as [kaan-web.vercel.app](https://kaan-web.vercel.app))
- **Record 10 seconds** in-app (microphone permission)
- Upload / IRRI sample clips
- On-device INT8 CNN via ONNX Runtime Web in the WebView
- Multilingual UI (en / hi / mr / pa / te)

## Prerequisites

- Node.js 20+
- JDK **21+** recommended for Capacitor 8 / current Android Gradle Plugin (JDK 17 is too old for `sourceCompatibility 21`)
- Android Studio (SDK 35+, build-tools) **or** Homebrew `android-commandlinetools`

## Build & open in Android Studio

```bash
cd web
npm install
npm run build:mobile   # next static export -> out/ then npx cap sync
npm run android        # opens Android Studio on the android/ project
```

In Android Studio:

1. Wait for Gradle sync.
2. Pick an emulator or USB device (enable USB debugging).
3. Run **app**.
4. Allow microphone when prompted, then tap **Record 10 seconds**.

## Debug APK (CLI)

With `ANDROID_HOME` set and SDK platform 36 + build-tools installed:

```bash
cd web
export JAVA_HOME=$(/usr/libexec/java_home -v 21 2>/dev/null || /usr/libexec/java_home)
# example Homebrew SDK root:
# export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
echo "sdk.dir=$ANDROID_HOME" > android/local.properties
npm run android:apk
# APK: android/app/build/outputs/apk/debug/app-debug.apk
```

Install:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Play Store (later)

1. **Build > Generate Signed Bundle / APK** → Android App Bundle (`.aab`).
2. Play Console listing: name **Kaan**, category Tools / Agriculture.
3. Data safety: audio processed on-device; no account required.
4. Privacy: audio stays on device (see About tab / README).

## Notes

- First analysis may take a few seconds while the ONNX WASM runtime loads (CDN unless later bundled).
- IRRI sample WAVs ship under `public/samples/`.
- iOS shell exists under `web/ios/` but is deferred; focus is Android first.
- License: Apache-2.0 (`LICENSE`).
