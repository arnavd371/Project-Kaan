import type { Metadata, Viewport } from "next";
import { Libre_Baskerville, IBM_Plex_Mono } from "next/font/google";
import NativeShell from "@/components/NativeShell";
import "./globals.css";

const baskerville = Libre_Baskerville({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-baskerville",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
});

export const metadata: Metadata = {
  title: "Kaan",
  description: "AI-Powered Acoustic Grain Pest Detector for Indian Farmers",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#f7f6f2",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${baskerville.variable} ${plexMono.variable} font-serif bg-cream text-ink`}>
        <NativeShell />
        {children}
      </body>
    </html>
  );
}
