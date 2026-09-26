import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The stage API (bin/cyclopsdiary stage) listens on this machine only; the dev server proxies to it.
const API = process.env.STAGE_API ?? "127.0.0.1:8790";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": `http://${API}`,
      "/media": `http://${API}`,
      "/feed": { target: `ws://${API}`, ws: true },
      "/video": { target: `ws://${API}`, ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
