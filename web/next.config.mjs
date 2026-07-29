/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static export so Capacitor can wrap the same build for Play Store / App Store.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
