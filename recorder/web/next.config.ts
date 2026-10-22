import type { NextConfig } from "next";

const backend = process.env.RECORDER_API ?? "http://127.0.0.1:8010";
const lanIp = process.env.MBAS_LAN_IP;

const nextConfig: NextConfig = {
  reactStrictMode: false, // avoid double-mounting media capture in dev
  // When the page is opened directly on :3010 (same laptop), proxy the API.
  // Through Caddy (:8443) the /api path goes straight to the backend.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
  allowedDevOrigins: lanIp ? [lanIp, "localhost"] : ["localhost"],
};

export default nextConfig;
