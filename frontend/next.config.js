/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
    return [
      {
        source: "/charts/:path*",
        destination: `${apiBase}/charts/:path*`
      }
    ];
  }
};

module.exports = nextConfig;
