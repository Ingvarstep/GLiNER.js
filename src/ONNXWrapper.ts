import ortCPU, { InferenceSession } from "onnxruntime-web";
import ortGPU from "onnxruntime-web/webgpu";

export type ExecutionProvider = "cpu" | "webgpu";

export interface IONNXSettings {
  modelPath: string | Uint8Array;
  executionProvider: ExecutionProvider;
  wasmPaths?: string;
  multiThread?: boolean;
  maxThreads?: number;
  fetchBinary?: boolean;
}

// NOTE: Version needs to match installed package!
const ONNX_WASM_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
const DEFAULT_WASM_PATHS = ONNX_WASM_CDN_URL;

export class ONNXWrapper {
  public ort: typeof import("onnxruntime-web") | typeof import("onnxruntime-web/webgpu") = ortCPU;
  private session: ortCPU.InferenceSession | ortGPU.InferenceSession | null = null;

  constructor(public settings: IONNXSettings) {
    if (settings.executionProvider === "webgpu") {
      this.ort = ortGPU;
    }
    this.ort.env.wasm.wasmPaths = settings.wasmPaths ?? DEFAULT_WASM_PATHS;
  }

  public async init() {
    if (!this.session) {
      const { modelPath, executionProvider, fetchBinary } = this.settings;
      if (executionProvider === "cpu") {
        if (fetchBinary) {
          const binaryURL = ONNX_WASM_CDN_URL + "ort-wasm-simd-threaded.wasm";
          const response = await fetch(binaryURL)
          const binary = await response.arrayBuffer();
          this.ort.env.wasm.wasmBinary = binary;
        }

        if (this.settings.multiThread) {
          const maxPossibleThreads = navigator.hardwareConcurrency ?? 1;
          const maxThreads = Math.min(this.settings.maxThreads ?? maxPossibleThreads, maxPossibleThreads);
          this.ort.env.wasm.numThreads = maxThreads;
        } else {
          this.ort.env.wasm.numThreads = 1;
        }
      } else if (executionProvider === "webgpu") {
        this.ort.env.wasm.numThreads = 1; // NOTE: not sure about this setting yet
      } else {
        throw new Error(`ONNXWrapper: Invalid execution provider: ${executionProvider}`);
      }

      // @ts-ignore
      this.session = await this.ort.InferenceSession.create(modelPath, {
        executionProviders: [executionProvider],
      });
    }
  }

  public async run(feeds: InferenceSession.FeedsType, options: InferenceSession.RunOptions = {}): Promise<InferenceSession.ReturnType> {
    if (!this.session) {
      throw new Error("ONNXWrapper: Session not initialized. Please call init() first.")
    }

    return await this.session.run(feeds, options);
  }

}