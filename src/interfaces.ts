import { InferenceSession } from "onnxruntime-common";

export type WebOrtLib =
  | typeof import("onnxruntime-web")
  | typeof import("onnxruntime-web/webgpu")
  | typeof import("onnxruntime-web/webgl");

export type NodeOrtLib = typeof import("onnxruntime-node");

export type ExecutionProvider = "cpu" | "wasm" | "webgpu" | "webgl";

// export type ExecutionContext = "web" | "node";

export interface IBaseSettings {
  modelPath: string | Uint8Array | ArrayBufferLike;
  // executionContext: ExecutionContext;
}

export interface IONNXNodeSettings extends IBaseSettings {}

export interface IONNXWebSettings extends IBaseSettings {
  executionProvider?: ExecutionProvider;
  wasmPaths?: string;
  multiThread?: boolean;
  maxThreads?: number;
  fetchBinary?: boolean;
}

export type IONNXSettings = IONNXNodeSettings | IONNXWebSettings;

export interface IONNXBaseWrapper {
  init(): Promise<void>;
  run(
    feeds: InferenceSession.FeedsType,
    options?: InferenceSession.RunOptions,
  ): Promise<InferenceSession.ReturnType>;
}

export interface IONNXWebWrapper extends IONNXBaseWrapper {
  ort: WebOrtLib;
  settings: IONNXWebSettings;
}

export interface IONNXNodeWrapper extends IONNXBaseWrapper {
  ort: NodeOrtLib;
  settings: IONNXNodeSettings;
}

export type ONNXWrapper = IONNXWebWrapper | IONNXNodeWrapper;

export interface ITransformersSettings {
  allowLocalModels: boolean;
  useBrowserCache: boolean;
}

export interface InitConfig {
  tokenizerPath: string;
  onnxSettings: IONNXSettings;
  transformersSettings?: ITransformersSettings;
  maxWidth?: number;
  modelType?: string;
}

export interface IInference {
  texts: string[];
  entities: string[];
  flatNer?: boolean;
  threshold?: number;
  multiLabel?: boolean;
}

export type RawInferenceResult = [string, number, number, string, number][][];

export interface IEntityResult {
  spanText: string;
  start: number;
  end: number;
  label: string;
  score: number;
}
export type InferenceResultSingle = IEntityResult[];
export type InferenceResultMultiple = InferenceResultSingle[];
