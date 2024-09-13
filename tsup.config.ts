// tsup.config.ts
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],  // Entry file(s)
  format: ['cjs', 'esm'],   // Output both CommonJS and ESM formats
  dts: true,                // Generate TypeScript declarations
  esbuildOptions(options, context) {
    if (context.format === 'esm') {
      // Modify the output for ES modules to use .mjs extension
      options.outExtension = { '.js': '.mjs' };
    } else if (context.format === 'cjs') {
      // Modify the output for CommonJS to use .cjs extension (optional)
      options.outExtension = { '.js': '.cjs' };
    }
  },
});