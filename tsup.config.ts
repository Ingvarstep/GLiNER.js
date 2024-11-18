import { defineConfig } from "tsup";

export default defineConfig([
  // Web Build
  {
    entry: ["src/web/index.ts"], // Entry for the web build
    format: ["cjs", "esm"], // Output formats
    dts: true, // Generate TypeScript declarations
    outDir: "dist", // Output directory for web
    treeshake: true,
    esbuildOptions(options, context) {
      if (context.format === "esm") {
        options.outExtension = { ".js": ".mjs" };
      } else if (context.format === "cjs") {
        options.outExtension = { ".js": ".cjs" };
      }
    },
  },
  // Node Build
  {
    entry: ["src/node/index.ts"], // Entry for the Node.js build
    format: ["cjs", "esm"], // Output formats
    dts: true, // Generate TypeScript declarations
    outDir: "dist/node", // Output directory for Node.js
    external: ["onnxruntime-web"], // Exclude web-specific runtime
    treeshake: true,
    esbuildOptions(options, context) {
      if (context.format === "esm") {
        options.outExtension = { ".js": ".mjs" };
      } else if (context.format === "cjs") {
        options.outExtension = { ".js": ".cjs" };
      }
    },
  },
]);
