import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  plugins: [
    wasm(),
    topLevelAwait(),
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: "", // Copies the files directly to the public directory
        },
      ],
    }),
  ],
  build: {
    target: "esnext",
    rollupOptions: {
      external: ["onnxruntime-node"], // Externalize the Node.js-specific package
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimizeDeps: {
    include: ["onnxruntime-web"], // Make sure 'onnxruntime-web' is handled properly by Vite
    exclude: ["onnxruntime-node"],
  },
});
