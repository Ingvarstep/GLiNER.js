import ort_CPU, { InferenceSession } from "onnxruntime-web";
import ort_WEBGPU from "onnxruntime-web/webgpu";
import ort_WEBGL from "onnxruntime-web/webgl";

type OrtLib =
  | typeof import("onnxruntime-web")
  | typeof import("onnxruntime-web/webgpu")
  | typeof import("onnxruntime-web/webgl");

export type ExecutionProvider = "cpu" | "wasm" | "webgpu" | "webgl";

// export type ExecutionContext = "web"

export interface IONNXSettings {
  modelPath: string | Uint8Array | ArrayBufferLike;
  // executionContext: ExecutionContext;
  executionProvider?: ExecutionProvider;
  wasmPaths?: string;
  multiThread?: boolean;
  maxThreads?: number;
  fetchBinary?: boolean;
}

// NOTE: Version needs to match installed package!
const ONNX_WASM_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
const DEFAULT_WASM_PATHS = ONNX_WASM_CDN_URL;

export class ONNXWrapper {
  public ort: OrtLib = ort_CPU;
  private session:
    | ort_CPU.InferenceSession
    | ort_WEBGPU.InferenceSession
    | ort_WEBGL.InferenceSession
    | null = null;

  constructor(public settings: IONNXSettings) {
    const { executionProvider, wasmPaths } = settings;
    switch (executionProvider) {
      case "cpu":
      case "wasm":
        break;
      case "webgpu":
        this.ort = ort_WEBGPU;
        break;
      case "webgl":
        this.ort = ort_WEBGL;
        break;
      default:
        throw new Error(`ONNXWrapper: Invalid execution provider: '${executionProvider}'`);
    }
    this.ort.env.wasm.wasmPaths = wasmPaths ?? DEFAULT_WASM_PATHS;
  }

  public async init() {
    if (!this.session) {
      const { modelPath, executionProvider, fetchBinary, multiThread } = this.settings;
      if (executionProvider === "cpu" || executionProvider === "wasm") {
        if (fetchBinary) {
          const binaryURL = this.ort.env.wasm.wasmPaths + "ort-wasm-simd-threaded.wasm";
          const response = await fetch(binaryURL);
          const binary = await response.arrayBuffer();
          this.ort.env.wasm.wasmBinary = binary;
        }

        if (multiThread) {
          const maxPossibleThreads = navigator.hardwareConcurrency ?? 0;
          const maxThreads = Math.min(
            this.settings.maxThreads ?? maxPossibleThreads,
            maxPossibleThreads,
          );
          this.ort.env.wasm.numThreads = maxThreads;
        }
      }

      // @ts-ignore
      this.session = await this.ort.InferenceSession.create(modelPath, {
        executionProviders: [executionProvider],
      });
    }
  }

  public async run(
    feeds: InferenceSession.FeedsType,
    options: InferenceSession.RunOptions = {},
  ): Promise<InferenceSession.ReturnType> {
    if (!this.session) {
      throw new Error("ONNXWrapper: Session not initialized. Please call init() first.");
    }

    return await this.session.run(feeds, options);
  }
}
