import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  publicDir: "public",
  build: {
    outDir: "dist",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
