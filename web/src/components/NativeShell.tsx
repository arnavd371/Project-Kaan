"use client";

import { useEffect } from "react";

export default function NativeShell() {
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { Capacitor } = await import("@capacitor/core");
        if (!Capacitor.isNativePlatform() || cancelled) return;

        document.documentElement.classList.add("kaan-native");

        const { StatusBar, Style } = await import("@capacitor/status-bar");
        await StatusBar.setOverlaysWebView({ overlay: false });
        await StatusBar.setBackgroundColor({ color: "#f7f6f2" });
        await StatusBar.setStyle({ style: Style.Light });

        try {
          const { SplashScreen } = await import("@capacitor/splash-screen");
          await SplashScreen.hide();
        } catch {
          // splash plugin optional
        }
      } catch {
        // ignore on web
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return null;
}
