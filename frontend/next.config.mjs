/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Self-contained server bundle for the Docker runtime stage.
  output: "standalone",
};

export default nextConfig;
