import type { NextConfig } from "next";

const backend = process.env.MBAS_API ?? "http://127.0.0.1:8020";
const lanIp = process.env.MBAS_LAN_IP;

const nextConfig: NextConfig = {
  reactStrictMode: false, // avoid double-mounting camera / LiveKit connections in dev
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/metrics", destination: `${backend}/metrics` },
    ];
  },
  allowedDevOrigins: lanIp ? [lanIp, "localhost"] : ["localhost"],
};

export default nextConfig;
